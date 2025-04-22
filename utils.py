from einops import rearrange, reduce
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import copy
import time
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
import os
from rdkit.Chem import AllChem, Draw
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
from torch_geometric.data import Data
from rdkit.Chem import AllChem
from itertools import product


from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from rdkit import Chem,DataStructs
import pickle
from pysmiles import read_smiles
# USE THIS TO SUPRESS STEREOCHEMICAL INFORMATION WARNINGS
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

charge2atm = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}
bndname2bndnumber = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':1.5}


def check_cut_cycle(data):
    # if an atom has more than one connection pointing to atms within the cycle, it's cut 
    pass


def canon_smiles_from_mol(atms, bonds):
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC]

    mol = Chem.RWMol()
    for atom in atms:
        a = Chem.Atom(atom)
        mol.AddAtom(a)
    for atms_, bnd_type in bonds.items():
        mol.AddBond(int(atms_[0]),int(atms_[1]),bond_dict[int(bnd_type)])
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))

def form_molecule(atms, bonds, sanitize=True):
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC]

    mol = Chem.RWMol()
    for atom in atms:
        a = Chem.Atom(atom)
        mol.AddAtom(a)
    for atms_, bnd_type in bonds.items():
        mol.AddBond(int(atms_[0]),int(atms_[1]),bond_dict[int(bnd_type)])
    if not sanitize: return Chem.CanonSmiles(Chem.MolToSmiles(mol))
    Chem.SanitizeMol(mol)
    return mol
        

def extract_common_aromatic_cycles(all_atms, all_bnds, all_smiles, fraction=-1, get_all_cycles=False):
    if fraction != -1:
        inds = np.random.choice(len(all_atms), int(len(all_atms)*fraction), replace=False)
        all_atms = [all_atms[i] for i in inds]
        all_bnds = [all_bnds[i] for i in inds]
        all_smiles = [all_smiles[i] for i in inds]


    ring_count_byno_atms = {}
    aromatic_ring_count = {}
    invalid = 0
    for ind, (atms, bnds, sm) in enumerate(zip(all_atms, all_bnds, all_smiles)):
        bnd2index = {(atm[0], atm[1]):ind for ind, atm in enumerate(bnds.keys())}
        bnd2index.update({(atm[1], atm[0]):ind for ind, atm in enumerate(bnds.keys())})
        bnd2type = {(atm[0], atm[1]):bnd_t for atm, bnd_t in bnds.items()}
        bnd2type.update({(atm[1], atm[0]):bnd_t for atm, bnd_t in bnds.items()})
        all_bnds = np.stack([np.array(b) for b in bnds.keys()])
        nodes = np.unique(all_bnds.flatten()) # get all nodes from bonds
        bonds = [(b[0], b[1]) for b in all_bnds]

        mol_graph = nx.Graph()
        mol_graph.add_nodes_from(nodes)
        mol_graph.add_edges_from(bonds)


        cycles = list(nx.simple_cycles(mol_graph))
        if not cycles: continue

        # someetimes atms are ionized and rdkit doesn't actually work
        try:
            mol = form_molecule(atms, bnds)
        except:
            invalid+=1
            continue
        # smallest set of smallest rings
        ssr = Chem.GetSymmSSSR(mol)


        for cycle in ssr:
            if str(mol.GetBondWithIdx(cycle[0]).GetBondType()) == "AROMATIC" or get_all_cycles:
                if len(cycle) not in ring_count_byno_atms: ring_count_byno_atms[len(cycle)] = 1
                else: ring_count_byno_atms[len(cycle)] += 1
                bonds_cycle = {(i,i+1):bnd2type[(cycle[i], cycle[i+1])] for i in range(len(cycle)-1)}
                bonds_cycle.update({(len(cycle)-1, 0):bnd2type[(cycle[-1], cycle[0])]})
                atms_cycle = [atms[c] for c in cycle]
                try:
                    aromatic_smiles = form_molecule(atms_cycle, bonds_cycle, sanitize=False)
                except:
                    breakpoint()
                if aromatic_smiles in aromatic_ring_count: aromatic_ring_count[aromatic_smiles]+=1
                else: aromatic_ring_count[aromatic_smiles]=1
        if ind % 10000 == 0: 
            print(sorted(((v,k) for k,v in aromatic_ring_count.items()), reverse=True)); print(invalid)
    return aromatic_ring_count



def transform_training_batch_into_visualizable(coords, inds, N_list, bnds, explicit_arom, explicit_h, tot_chn, data_type, non_expl_bonds=False):
    tot_bnds = 3 + explicit_arom
    if "GEOM" not in data_type:
        atm_symbs = {0:"C",1:"O",2:"N",3:"F"}
    else:
        atm_symbs = {0:'C',1:'N',2:'O',3:'F',4:'P',5:'S',6:'Cl'}
    if explicit_h: atm_symbs.update( {max(atm_symbs.keys())+ 1 : 'H'} )
    tot_atm = len(atm_symbs)

    atm_start = max(atm_symbs.keys())
    bnd_symbs = {atm_start+1:1, atm_start+2:2, atm_start+3:3}
    if explicit_arom: bnd_symbs.update({atm_start+4:4})
    if tot_chn != len(atm_symbs) + len(bnd_symbs): 
        tot_bnds_atms = len(atm_symbs) + len(bnd_symbs)
        atm_symbs.update({tot_bnds_atms:'X', tot_bnds_atms+1:'Y'})


    atm_symb = []
    atm_pos = []
    actual_bnds = []
    total_coords = 0
    for index, n_list in enumerate(N_list):


        current_inds = inds[total_coords:total_coords+sum(n_list)]


        # if current_inds[-1] > tot_bnds + tot_atm: continue
        # current_coords=  coords[total_coords:total_coords+sum(n_list)]
        atm_symb.append([atm_symbs[i_ % tot_chn] for i_ in current_inds if i_ % tot_chn in atm_symbs])
        # atm_pos.append(torch.stack([crd for i_, crd in enumerate(current_coords) if i_ % tot_chn in atmind2symb]))
        atm_pos.append(torch.stack([coords[total_coords+i] for i, ind in enumerate(current_inds) if ind % tot_chn in atm_symbs]))

        if non_expl_bonds:
            bnds_ = [bnd for bnd in bnds if bnd[1] in np.arange(total_coords, total_coords + sum(n_list))]
            actual_bnds.append([[a1-total_coords,a2-total_coords,b_+1] for b_,a1,a2 in bnds_])
        else:
            # bonds that correspond to this molecule
            bnds_ = [bnd for bnd in bnds if bnd[0] in np.arange(total_coords, total_coords + sum(n_list))]
            bnd_types = [inds[b[0]] % tot_chn - tot_atm + 1 for b in bnds_]

            if 4 in bnd_types and not explicit_arom: breakpoint()
            bonds_atoms = [[b[1]-total_coords, b[2] - total_coords] for b in bnds_]
            actual_bnds.append([[b[0],b[1],b_] for b,b_ in zip(bonds_atoms, bnd_types)])


        total_coords+=sum(n_list)
    return atm_symb, atm_pos, actual_bnds


def plot_one_mol(atm_pos, actual_bnd, plot_bnd=None, threshold=0.7, field=None, x_grid=None,y_grid=None,z_grid=None, atm_symb=None, wo_h=False):

    import matplotlib.pyplot as plt
    
    atm2color = {'C':'black', 'O':'red', 'N':'blue', 'F':'green', 'H':'white'}
    bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [atm2color[a] for a in atm_symb]
    ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300, edgecolor='black')
    print(actual_bnd,atm_pos)
    for bnd_ in actual_bnd:
        if wo_h and (atm_symb[bnd_[0]] == 'H' or atm_symb[bnd_[1]] == 'H'): continue
        bnd_inds = [bnd_[0], bnd_[1]]
        line = atm_pos[bnd_inds]
        if bnd_[2] == 1.5: bnd_[2] = 4
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd_[2]], linewidth=bnd_[2])

        dist = np.linalg.norm(line[0]-line[1])
        bond_position = (line[0]+line[1])/2
        # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
    if plot_bnd is not None:
        x = x_grid[field[plot_bnd]>threshold]
        y = y_grid[field[plot_bnd]>threshold]
        z = z_grid[field[plot_bnd]>threshold]
        scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold])
        plt.colorbar(scatter)
    # for bnd in candidaten_bnds:
    plt.show()

def visualize_failed_mols_w_hydrogens(info, title=None):
    # get all batches
    atm2color = {'C':'black', 'O':'blue', 'N':'green', 'F':'yellow', 'H':'orange'}
    bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange', 1.5:'orange'}
    for failed_batch_info in info:
        if len(failed_batch_info) != 5: continue # some batches might be broken
        field, candidaten_bnds,actual_bnd,atm_symb,atm_pos = failed_batch_info
        if type(actual_bnd) == dict: actual_bnd = [[k[0],k[1],v] for k,v in actual_bnd.items()]
        # create a 3d plot from the Nx3 array atm_bnd_pos
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for atm in atm_pos:
        colors = [atm2color[a] for a in atm_symb]
        ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300)
        for bnd in actual_bnd:
            bnd_inds = [bnd[0], bnd[1]]
            line = atm_pos[bnd_inds]
            dist = np.linalg.norm(line[0]-line[1])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=3)
            bond_position = (line[0]+line[1])/2
            # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
        # for bnd in candidaten_bnds:
        if title is not None: plt.title(title)
        plt.show()

def load_model(self, model_name='model1.pt'):

    accelerator = self.accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(str(self.results_folder + f'/{model_name}.pt'), map_location=device)

    model = self.accelerator.unwrap_model(self.model)
    model.load_state_dict(data['model'])

    self.step = data['step']
    self.opt.load_state_dict(data['opt'])
    self.ema.load_state_dict(data['ema'])

    if 'version' in data:
        print(f"loading from version {data['version']}")

    if exists(self.accelerator.scaler) and exists(data['scaler']):
        self.accelerator.scaler.load_state_dict(data['scaler'])



import networkx as nx
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
import random
from sklearn.mixture import GaussianMixture


max_types = {'QM9':4,'ZINC250K':9}
UNQ_elements = {'QM9':["C","O","N","F"], 'QM9-hydr':["C","O","N","F","H"]}




# constants
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}


def sanity_check_data(all_bonds, args):
    bonds = all_bonds['valid'] if type(all_bonds) == dict else all_bonds
    for b in bonds:
        if 1.5 in b.values() and args.explicit_aromatic: return
        if 1.5 in b.values() and not args.explicit_aromatic: print("!!!WARNING!!! YOU USED explicit_aromatic=FALSE BUT THE "
                                "dataset sepcified in data_file CONTAINS AROMATIC ")
    if args.explicit_aromatic: print("!!!WARNING!!! YOU USED explicit_aromatic=TRUE BUT THE "
                                "dataset sepcified in data_file DOES NOT CONTAIN AROMATIC BONDS EXPLICITLY (most likely kekulize representation)")





def max_axis(x_grid,y_grid,z_grid):
    max_, min_ = np.max(x_grid), np.min(x_grid)
    no_points = x_grid.shape[0]
    y = np.linspace(min_, max_, no_points)
    z = np.linspace(min_, max_, no_points)
    x = np.linspace(min_, max_, no_points)

    x_grid,y_grid,z_grid = np.meshgrid(x,y,z, indexing='ij')
    return x_grid,y_grid,z_grid

def create_simple_chiral_ds(args, data_path):
    if not args.create_simple_chiral_ds: return
    path = os.path.join(data_path, args.data_file + ".bin")
    if os.path.exists(path): return 
    bndname2bndnumber = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':1.5}

    limits = [{"min":np.inf, "max":-np.inf} for _ in range(3)]


    c_modif, n_modif, o_modif = ["C", "CC", "C=C", "CN"], ["N", "NC", "N=C", "NN"], ["O", "OC", "ON"]
    atom_pairs =  list(product(c_modif, n_modif, o_modif))
    all_smiles = []
    for a_p in atom_pairs:
        smiles = "C" + "".join([f"({c})" for c in a_p])
        mol = Chem.MolFromSmiles(smiles); 
        mol = Chem.AddHs(mol)
        all_smiles.append(Chem.MolToSmiles(mol))
    # for s in all_smiles: print(s)


    all_coords, all_edge_attr, all_atom_symbols = [], [], []

    for s in all_smiles:
        coords = []
        mol = Chem.MolFromSmiles(s)
        mol = Chem.AddHs(mol)
        check = AllChem.EmbedMolecule(mol,useRandomCoords=True)
        # if check==-1: removed_smiles+=1; removed_reason2count["AllChem.EmbedMolecule failed"]+=1; continue
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            positions = conf.GetAtomPosition(i)
            coords.append([positions.x,positions.y,positions.z])
        coords = np.array([coords]).reshape(-1,3)
        edge_attr = {}
        if args.not_use_pca: coords_aligned = coords
        else: coords_aligned = align(coords)

        conf,new_coords,atom_symbols = update_mol_positions_whydr(mol,conf,coords_aligned)

        for bnd in mol.GetBonds():
            # print([bnd.GetBeginAtomIdx(),bnd.GetEndAtomIdx()], bnd.GetBondType())
            edge_attr[(bnd.GetBeginAtomIdx(),bnd.GetEndAtomIdx())] = bndname2bndnumber[str(bnd.GetBondType())]
        new_coords = np.array(new_coords)
        new_coords = fix_chiral_atms(atom_symbols, edge_attr, new_coords)
        limits = update_limits(limits, np.min(new_coords, axis=0), np.max(new_coords, axis=0), args.std_atoms)
        
        all_atom_symbols.append(atom_symbols)
        all_coords.append(new_coords)
        all_edge_attr.append(edge_attr)

    x_lim,y_lim,z_lim = limits
    bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(args.resolution*8))*8), [x_lim,y_lim,z_lim])]
    x,y,z = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),
                            bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')

    all_coords_train_test_valid = {'train':all_coords, 'valid':all_coords, 'test':all_coords}
    all_smiles_train_test_valid = {'train':all_smiles, 'valid':all_smiles, 'test':all_smiles}
    all_atom_symbols_train_test_valid = {'train':all_atom_symbols, 'valid':all_atom_symbols, 'test':all_atom_symbols}
    all_edge_attr_train_test_valid = {'train':all_edge_attr, 'valid':all_edge_attr, 'test':all_edge_attr}
    pickle.dump([x,y,z,all_coords_train_test_valid,all_smiles_train_test_valid,all_atom_symbols_train_test_valid,all_edge_attr_train_test_valid], open(path, "wb"))
    return 


def get_classes_cdf(bin2count, cdf=5e-3):
    current_count = 0
    start, end = None, None
    limits2_class = {}
    class_ind = 0
    for bin_, count in bin2count.items():
        if start is None: start = bin_; end = bin_
        end = bin_
        current_count += count
        if current_count > cdf:
            limits2_class[(start,end)] = class_ind
            class_ind+=1;current_count = 0;start = end
        if count < cdf: bin2count[bin_] = 0
    limits2_class[(start,end)] = class_ind
    return limits2_class

def discretize_cond_var(cond_var):
    bins = np.linspace(np.min(cond_var), np.max(cond_var), 1000)
    bin2count = {i:0 for i in range(1000)}
    for c in cond_var:
        # if c > 100: 
        #     bin_ = np.digitize(c, bins, right=True).item()
        bin_ = np.digitize(c, bins, right=True).item()
        bin2count[bin_]+=1
    bin2count = {k:v/sum(bin2count.values()) for k,v in bin2count.items()}
    bin2count = dict(sorted(bin2count.items()))
    limits2_class = get_classes_cdf(bin2count)
    cond_v_limits2class = {}
    for k,v in limits2_class.items():
        cond_v_limits2class[bins[k[0]], bins[k[1]]] = v
    return cond_v_limits2class

def get_cond_var_bin_cls(cond_var, bin2classes):
    bin_lims_clses = np.array([[b[0], b[1], c] for b, c in bin2classes.items()])
    bin_lims_clses[0,0] = -np.inf
    bin_lims_clses[-1,1] = np.inf
    classes = []
    for c_v in cond_var:
        ind = np.argwhere((c_v > bin_lims_clses[:,0]) * (c_v <= bin_lims_clses[:,1])).item()
        classes.append(int(bin_lims_clses[ind,2]))
    return classes

def setup_data_files(args, Smiles,train_valid_test,atoms=None,atom_pos=None,add_atom_num_cls=False, use_subset=-1, center_atm_to_grids=False, data_path=None, consider_arom_chns_as_atms=False, fix_chiral_conf=None, cond_variable=None):


    # * get atom no classes
    if add_atom_num_cls and args.no_sep_bnd_chn: print("!!!WARNING!!! classifier guidance not implemented using dist-based atm type data")
    if add_atom_num_cls and args.data == "GEOM":
        file = os.path.join(data_path, "geom_conf1.bin") if not args.explicit_aromatic \
            else os.path.join(data_path, "geom_conf1_arom.bin")
        bins_upper_lim = get_bin_atm_upper_lims(file=file, geom=True, data_path=data_path, explicit_h=args.explicit_hydrogen, consider_arom_chns_as_atms=consider_arom_chns_as_atms)
        bins_upper_lim.insert(0,0)
        bins = [[bins_upper_lim[i], bins_upper_lim[i+1]] for i in range(len(bins_upper_lim)-1)]
        unique_classes = len(bins)
    elif add_atom_num_cls and os.path.exists(os.path.join(data_path, args.data_file.replace(".bin","") +".bin")):
        bins_upper_lim = get_bin_atm_upper_lims(file=args.data_file.replace(".bin","") +".bin", data_path=data_path, explicit_h=args.explicit_hydrogen, consider_arom_chns_as_atms=consider_arom_chns_as_atms)
        bins_upper_lim.insert(0,0)
        bins = [[bins_upper_lim[i], bins_upper_lim[i+1]] for i in range(len(bins_upper_lim)-1)]
        unique_classes = len(bins)
    
    if args.discrete_conditioning: 
        bin_limits_2_classes = discretize_cond_var(cond_var=cond_variable[0])

    if args.no_limits:
        x = {"min":float('inf'),"max":float('-inf')}
        y = {"min":float('inf'),"max":float('-inf')}
        z = {"min":float('inf'),"max":float('-inf')}

    else:
        # previous small data
        # x = {"min": -4.5, "max": 4.5}
        # y = {"min": -3.5, "max": 3.5}
        # z = {"min": -2.5, "max": 3.5}
        x = {"min":-5.26791692653561, "max":5.10421340654044}
        y = {"min":-4.777702113781722,"max":5.13692682839705}
        z = {"min":-4.8703702402102, "max":4.95109161233209}


    limits = [x, y, z]
    resolution = args.resolution
    if "GEOM" in args.data:
        aromatic_suffix = "_arom" if args.explicit_aromatic else ""
        print(os.path.join(data_path, "conf_1_limits.bin"))
        if not os.path.exists(os.path.join(data_path, f"conf_1_limits.bin")):
            print("!!!WARNING!!! DID NOT FIND conf_1_limits.bin file. ATTEMPTING TO EXTRACT, THEN EXITING")
            if args.explicit_aromatic:
                print("!!!ERROR!!! NOT IMPLEMENTED AUTOMATIC EXTRACTION OF AROMATIC GEOM DATA. EXITING")
                exit(1)
            for i in range(1,31):
                data_f = os.path.join(data_path, "geom_conf{}.bin".format(i))
                save_conf_limits_file = os.path.join(data_path, "geom_data/", "conf_{}_limits.bin".format(i))
                align_extract_geom_data(data_f, save_conf_limits_file)
                print("Finished processing conf {}".format(i))
        dataset_constructor = GeomDatasetConstructor(no_confs=6, cls_bins=bins, 
                        resolution=args.resolution, subset_confs=args.subset_confs,
                        data_path=data_path, run_name=args.run_name,
                        explicit_hydrogen=args.explicit_hydrogen,
                        explicit_aromatic=args.explicit_aromatic)
        return dataset_constructor, unique_classes, bins
    else:
        # * data already extracted and using edm data format (train/val/test splits)
        if not args.force_create_data and args.data_file != "none" and os.path.exists(os.path.join(data_path, args.data_file.replace(".bin","") +".bin")) and train_valid_test:
            dataset_info = pickle.load(open(os.path.join(data_path, args.data_file.replace(".bin","") +".bin"), "rb"))
            if exists(cond_variable) + 7 != len(dataset_info): print("ERROR! YOU HAVE SPECIFIED DATASETS w/wo COND VAR BUT ARE USING DS wo/w COND PROPS! EXITING!"); exit(1)
            if exists(cond_variable): x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds,cond_variables = pickle.load(open(os.path.join(data_path, args.data_file.replace(".bin","") +".bin"), "rb"))
            else: 
                x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds = pickle.load(open(os.path.join(data_path, args.data_file.replace(".bin","") +".bin"), "rb"))
                cond_variables = None
            if args.augment_rotations: x_grid,y_grid,z_grid = max_axis(x_grid,y_grid,z_grid)
            sanity_check_data(all_bonds, args)

            if args.no_sep_bnd_chn:
                data = AtomPositionDataset(all_coords['train'], all_smiles['train'],x_grid, y_grid, z_grid,args.std_atoms)
                data_val = AtomPositionDataset(all_coords['valid'], all_smiles['valid'],x_grid, y_grid, z_grid,args.std_atoms)
            elif args.compact_batch and args.remove_bonds:
                if args.discrete_conditioning and exists(cond_variable) and add_atom_num_cls:
                    print("!!!WARNING!!! Currently I am simply replacing the classes designated to number-of-atoms with conditional variable classes; add_atom_num_cls will not do anything")
                if args.discrete_conditioning and exists(cond_variable):
                    atom_no_classes_train, atom_no_classes_valid = get_cond_var_bin_cls(bins, all_atom_symbols['train'], expl_h=args.explicit_hydrogen), get_atom_no_bin_class(bins, all_atom_symbols['valid'], expl_h=args.explicit_hydrogen)
                    unique_classes = len(bin_limits_2_classes)
                elif add_atom_num_cls: atom_no_classes_train, atom_no_classes_valid = get_atom_no_bin_class(bins, all_atom_symbols['train'], expl_h=args.explicit_hydrogen), get_atom_no_bin_class(bins, all_atom_symbols['valid'], expl_h=args.explicit_hydrogen)
                else: atom_no_classes_train, atom_no_classes_valid = None, None
                data = AtomPositionNoSepBondsDatasetCompact(all_coords['train'], all_smiles['train'],x_grid, y_grid, z_grid,
                                                args.std_atoms, all_atom_symbols['train'],all_bonds['train'],
                                                ignore_aromatic=not args.explicit_aromatic,explicit_hydrogens=args.explicit_hydrogen,
                                                debug_ds=args.debug_ds, subsample_points=args.subsample_points,mixed_prec=args.mixed_prec,
                                                augment_rotations=args.augment_rotations, atom_no_classes=atom_no_classes_train, explicit_aromatic=args.explicit_aromatic, 
                                                use_subset=use_subset, center_atm_to_grids=center_atm_to_grids,
                                                arom_cycle_channel=args.arom_cycle_channel)    
                data_val = AtomPositionNoSepBondsDatasetCompact(all_coords['valid'], all_smiles['valid'],x_grid, y_grid, z_grid,
                                                    args.std_atoms, all_atom_symbols['valid'],all_bonds['valid'],
                                                    ignore_aromatic=not args.explicit_aromatic, explicit_hydrogens=args.explicit_hydrogen,
                                                    debug_ds=args.debug_ds, atom_no_classes=atom_no_classes_valid, explicit_aromatic=args.explicit_aromatic,
                                                    use_subset=use_subset, center_atm_to_grids=center_atm_to_grids, arom_cycle_channel=args.arom_cycle_channel)    

            elif args.compact_batch:
                if args.discrete_conditioning and exists(cond_variable) and add_atom_num_cls:
                    print("!!!WARNING!!! Currently I am simply replacing the classes designated to number-of-atoms with conditional variable classes; add_atom_num_cls will not do anything")
                if args.discrete_conditioning and exists(cond_variable):
                    atom_no_classes_train, atom_no_classes_valid = get_cond_var_bin_cls(cond_var=cond_variables['train'], bin2classes=bin_limits_2_classes), get_cond_var_bin_cls(cond_var=cond_variables['valid'], bin2classes=bin_limits_2_classes)
                    unique_classes = len(bin_limits_2_classes)

                elif add_atom_num_cls: atom_no_classes_train, atom_no_classes_valid = get_atom_no_bin_class(bins, all_atom_symbols['train'], expl_h=args.explicit_hydrogen), get_atom_no_bin_class(bins, all_atom_symbols['valid'], expl_h=args.explicit_hydrogen)
                else: atom_no_classes_train, atom_no_classes_valid = None, None

                data = AtomPositionSepBondsDatasetCompact(all_coords['train'], all_smiles['train'],x_grid, y_grid, z_grid,
                                                args.std_atoms, all_atom_symbols['train'],all_bonds['train'],
                                                ignore_aromatic=not args.explicit_aromatic,explicit_hydrogens=args.explicit_hydrogen,
                                                debug_ds=args.debug_ds, subsample_points=args.subsample_points,mixed_prec=args.mixed_prec,
                                                augment_rotations=args.augment_rotations, atom_no_classes=atom_no_classes_train, explicit_aromatic=args.explicit_aromatic, use_subset=use_subset, center_atm_to_grids=center_atm_to_grids,
                                                arom_cycle_channel=args.arom_cycle_channel,cond_variables=cond_variables['train'] if exists(cond_variable) else None)    
                data_val = AtomPositionSepBondsDatasetCompact(all_coords['valid'], all_smiles['valid'],x_grid, y_grid, z_grid,
                                                    args.std_atoms, all_atom_symbols['valid'],all_bonds['valid'],
                                                    ignore_aromatic=not args.explicit_aromatic, explicit_hydrogens=args.explicit_hydrogen,
                                                    debug_ds=args.debug_ds, atom_no_classes=atom_no_classes_valid, explicit_aromatic=args.explicit_aromatic,
                                                    use_subset=use_subset, center_atm_to_grids=center_atm_to_grids, arom_cycle_channel=args.arom_cycle_channel,
                                                    cond_variables=cond_variables['valid'] if exists(cond_variable) else None)    

            else:
                if add_atom_num_cls: atom_no_classes_train, atom_no_classes_valid = get_atom_no_bin_class(bins, all_atom_symbols['train'], expl_h=args.explicit_hydrogen), get_atom_no_bin_class(bins, all_atom_symbols['valid'], expl_h=args.explicit_hydrogen)
                else: atom_no_classes_train, atom_no_classes_valid = None, None
                data = AtomPositionSepBondsDataset(all_coords['train'], all_smiles['train'],x_grid, y_grid, z_grid,
                                                args.std_atoms, all_atom_symbols['train'],all_bonds['train'],
                                                ignore_aromatic=not args.explicit_aromatic,explicit_hydrogens=args.explicit_hydrogen,
                                                debug_ds=args.debug_ds, subsample_points=args.subsample_points,mixed_prec=args.mixed_prec,
                                                augment_rotations=args.augment_rotations, atom_no_classes=atom_no_classes_train)    
                data_val = AtomPositionSepBondsDataset(all_coords['valid'], all_smiles['valid'],x_grid, y_grid, z_grid,
                                                    args.std_atoms, all_atom_symbols['valid'],all_bonds['valid'],
                                                    ignore_aromatic=not args.explicit_aromatic, explicit_hydrogens=args.explicit_hydrogen,
                                                    debug_ds=args.debug_ds, atom_no_classes=atom_no_classes_valid)    
            # for i in range(50000,100000):
            #     data.__getitem__(i)
            if args.compact_batch: H, W, D = x_grid.shape; C = 7 + args.explicit_aromatic + args.explicit_hydrogen + 2 * args.arom_cycle_channel
            elif args.subsample_points != -1: C, H, W, D = data.__getitem__(0)[0].shape
            elif add_atom_num_cls:  C, H, W, D = data.__getitem__(0)[0].shape
            else:  C, H, W, D = data.__getitem__(0).shape if not args.debug_ds else data.__getitem__(0)['field'].shape
        # * data already extracted
        elif not args.force_create_data and args.data_file != "none" and os.path.exists(os.path.join(data_path, args.data_file +".bin")):
            x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds = pickle.load(open(os.path.join(data_path, args.data_file +".bin"), "rb"))
            if args.augment_rotations: x_grid,y_grid,z_grid = max_axis(x_grid,y_grid,z_grid)
            if args.no_sep_bnd_chn:
                data = AtomPositionDataset(all_coords, all_smiles,x_grid, y_grid, z_grid,args.std_atoms,mixed_prec=args.mixed_prec)
            else:
                data = AtomPositionSepBondsDataset(all_coords, all_smiles,x_grid, y_grid, z_grid,args.std_atoms, all_atom_symbols,all_bonds,
                                                    ignore_aromatic=not args.explicit_aromatic,explicit_hydrogens=args.explicit_hydrogen,
                                                    debug_ds=args.debug_ds, subsample_points=args.subsample_points,augment_rotations=args.augment_rotations)    
            data_val=None
            C, H, W, D = data.__getitem__(0).shape if not args.debug_ds else data.__getitem__(0)[0].shape
        # * when data needs to be extracted
        elif args.data_file != "none":
            # for now, this will simply extract all molecules that are withing arbitrarily chosen limits after (limits)
            x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds,all_cond_variables = extract_aligned_mol_coords(Smiles, args.data, std=args.std_atoms,
                                                                            use_pca=not args.not_use_pca, rescale=args.rescale,
                                                                            limits=limits, resolution=resolution, explicit_aromatic=args.explicit_aromatic,
                                                                            atoms=atoms, atom_pos=atom_pos, explicit_hydrogen=args.explicit_hydrogen, arom_cycle_channel=args.arom_cycle_channel, fix_chiral_conf=fix_chiral_conf, cond_variable=cond_variable)
            if train_valid_test:
                all_smiles = {set:all_smiles[ind] for ind,set in enumerate(['train','valid','test'])}
                all_coords = {set:all_coords[ind] for ind,set in enumerate(['train','valid','test'])}
                all_atom_symbols = {set:all_atom_symbols[ind] for ind,set in enumerate(['train','valid','test'])}
                all_bonds = {set:all_bonds[ind] for ind,set in enumerate(['train','valid','test'])}
                all_cond_variables = {set:all_cond_variables[ind] for ind,set in enumerate(['train','valid','test'])}

            pickle.dump([x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds,all_cond_variables], \
                        open(data_path + args.data_file+".bin", "wb"))
            print("Done extracting the datapoitns. Exiting...")
            exit(1)


        else:
            data,unq_elem,x_grid,y_grid,z_grid = get_mol_reps(Smiles[0:args.nsamples],args.data,std=args.std_atoms,
                                        use_pca=not args.not_use_pca,rescale=args.rescale, limits=limits, resolution=resolution)



            Train_data = DataLoader(data,batch_size=args.batch_size,shuffle=True)
            C,H,W,D = data[0].shape
            data=torch.tensor([data]).view(-1,C,H,W,D).float()
            data_val = None
    if add_atom_num_cls: return data, data_val, C, H, W, D, x_grid, y_grid, z_grid, unique_classes
    return data, data_val, C, H, W, D, x_grid, y_grid, z_grid

def retrieve_smiles(args,train_valid_test,data_path):
    Atoms, Atoms_pos = None, None
    cond_variable = None
    if train_valid_test and (args.data in ['GEOM'] ):
        data_path += "/"
        splits = pickle.load(open(data_path + "splits.pkl", "rb"))
        mol_info = pd.read_csv(data_path + "mol_summary.csv")
        mol_ids, mol_smiles = mol_info['mol_id'].values, mol_info['smiles'].values
        Smiles = []
        for set in ['train', 'val', 'test']:
            set_ids = splits[set]
            set_smiles = mol_smiles[set_ids]
            Smiles.append([sm for sm in set_smiles])

    elif train_valid_test and (args.data in ['QM9','ZINC250K', 'QM9EDM'] or 'QM9EDM' in args.data):
        # for datasets with train_valid_test, Smiles is a list of 3 train/val/test lists of smiles
        cwd = os.getcwd()
        data_path =  data_path + "/qm9_edm/qm9/"
        data_path = data_path.replace("//","/")
        Smiles = []
        Atoms, Atoms_pos = [], []
        cond_variable = []
        for set in ['train', 'valid', 'test']:
            data_ = np.load(os.path.join(data_path,set +".npz"))
            # TODO the stereo-info molecules do not correspond (atom-wise) to the atom positions; ignore for now
            smile_set = data_['mol_wo_stereo'] # if not args.explicit_hydrogen else data_['mol_w_stereo']
            
            if set == 'train' and args.cond_variable:
                np.random.seed(42)
                fixed_perm = np.random.permutation(len(data_['num_atoms']))
                sliced_perm = fixed_perm[len(data_['num_atoms'])//2:]
            else:
                sliced_perm = np.arange(len(data_['num_atoms']))
        

            smile_set = smile_set[sliced_perm]
            
            atom_ = list(map(lambda atm_chrg,atm_len: [charge2atm[c] for c in atm_chrg[:atm_len]], data_['charges'], data_['num_atoms']))
            atom_ = [atom_[i] for i in sliced_perm]
            
            atom_pos_ = list(map (lambda atm_pos,atm_len: atm_pos[:atm_len], data_['positions'], data_['num_atoms']))
            atom_pos_ = [atom_pos_[i] for i in sliced_perm]
            if args.cond_variable:
                cond_props_ = []
                for c_p in args.cond_variable:
                    
                    cond_props_.append(data_[c_p][sliced_perm, None])    
                cond_properties = np.concatenate(cond_props_, axis=1)
            else:
                cond_properties = None

            Smiles.append(smile_set)
            Atoms.append(atom_)
            Atoms_pos.append(atom_pos_)
            cond_variable.append(cond_properties)

    elif not train_valid_test and args.data in ["QM9", "ZINC"]:
        final_data = []
        #loading molecular data
        cwd = os.getcwd()
        data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
        with open(data_path) as f:
            Smiles = f.readlines()
    else:
        print("specified data is not available. Exiting")
        logging.info("specified data is not available. Exiting")
        exit()
    return Smiles, Atoms, Atoms_pos if (args.use_original_atm_pos and args.data != "GEOM") else Smiles , cond_variable

def exists(x):
    return x is not None



def plot_channel(channel,x ,y,z, some_position=None, first_atm_pos=None,second_atm_pos=None, threshold=0.4, title=None,
                 lines=[], figure=None, specific_subplot=None, zoom_in=True):
    # create a 3D plot of the channel
    

    # threshold = (4.15/5 * channel.max() )
    # threshold = threshold.flatten()
    x,y,z = x.flatten(), y.flatten(), z.flatten()
    channel = channel.flatten()
    x_max, x_min, y_max,y_min, z_max, z_min = x.max(), x.min(), y.max(), y.min(), z.max(), z.min()
    x = x[channel>threshold]
    y=  y[channel>threshold]
    z = z[channel>threshold]
    channel = channel[channel>threshold]

        
    fig = plt.figure() if figure is None else figure
    ax = fig.add_subplot(111 if specific_subplot is None else specific_subplot, projection='3d')
    if exists(some_position):
        ax.scatter(some_position[:,0],some_position[:,1],some_position[:,2],c="red", s=200, marker="x")
    
    if exists(first_atm_pos) and exists(second_atm_pos):
        ax.scatter(first_atm_pos[0],first_atm_pos[1],first_atm_pos[2],c="green", s=100, marker="x")
        ax.scatter(second_atm_pos[0],second_atm_pos[1],second_atm_pos[2],c="green", s=100, marker="x")
    scatter = ax.scatter(x,y,z,c=channel)
    if len(lines):
        for line in lines:
            ax.scatter(line[:,0],line[:,1],line[:,2],c="black", s=100, marker="x")
            ax.plot(line[:,0],line[:,1],line[:,2],c="black")

    # set limits according to x,y,z
    if not zoom_in:
        ax.set_xlim3d(x_min,x_max)
        ax.set_ylim3d(y_min,y_max)
        ax.set_zlim3d(z_min,z_max)
    # add colorbar
    if exists(title):
        plt.title(title)
    
    if specific_subplot is None: plt.colorbar(scatter); plt.show(); 



def flip_axes(x,axes):
    """Flips `axes` based on the weight cretirion.

    Args:
        x: Atom coordiantes. Has shape (n_atoms, spatial_dim).
        axes: Axes to flip. Has shape (spatial_dim, spatial_dim).

    Returns:
        Flipped axes.
    """
    flipped_axes = axes.copy()
    n_atoms, spatial_dim = x.shape
    for i in range(spatial_dim):
        proj_coords = x @ axes[[i]].T
        n_neg = np.sum(proj_coords < 0)
        n_pos = n_atoms - n_neg
        if n_pos > n_neg:
            flipped_axes[i] *= -1
    return flipped_axes


def align(x):
    """Aligns the molecule `x` (represented by atom cordinates) to
    the standard cartesian basis.

    Args:
        x: Atom coordinates. Has shape (n_atoms, spatial_dim).

    Returns:
        x_aligned: Atom coordinates aligned to the standard cartesian basis.
    """
    x = x - x.mean(0, keepdims=True)  # remove mean
    C = np.cov(x.T)


    eigval, eigvecs = np.linalg.eig(C)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvecs = eigvecs[:,idx]

    pca_ax = eigvecs  # compute pca axes

    # pca_ax = flip_axes(x, pca_ax)  # Not sure what happened here, but flipping axes results in optical ismores 
    # (destroying the data) - will ignore this for now 
    # print(np.linalg.det(pca_ax))

    det_rot = np.linalg.det(pca_ax)
    # reflections (det=-1 matrices) can lead to optical ismores (destroying the data). Flip one axes
    if det_rot < -0.999: pca_ax = pca_ax * np.array([1, 1, -1]); det_rot = np.linalg.det(pca_ax)
    if (det_rot - 1)**2 > 0.00001: print("!!!WARNING!!! Rotation matrix has a determinant = {}".format(det_rot))
    x_aligned = (pca_ax.T @ x.T).T   # align pca axes to cartesian frame
    return x_aligned


def get_unique(Smiles):
    Unique_elements = []
    dall = []
    for i in Smiles:
        mol = read_smiles(i)
        elements = nx.get_node_attributes(mol, name = "element")
        for k in range(len(elements)):
            dall.append(elements[k])
    
    for val in dall: 
        if val in Unique_elements: 
            continue 
        else:
            Unique_elements.append(val)

    return Unique_elements

def update_mol_positions_whydr(mol,conf,coords):
    coords = np.array(coords, dtype=np.float64)
    coords_ = []
    atom_names = []
    for i, atom in enumerate(mol.GetAtoms()):
        if conf is not None: conf.SetAtomPosition(i,coords[i])
        coords_.append(coords[i].tolist())
        atom_names.append(atom.GetSymbol())

    return conf,coords_,atom_names

def update_mol_positions(mol,conf,coords):
    coords = np.array(coords, dtype=np.float64)
    coords_wo_H = []
    atom_names_wo_H = []
    for i, atom in enumerate(mol.GetAtoms()):
        if conf is not None: conf.SetAtomPosition(i,coords[i])
        if atom.GetSymbol() !='H':
            coords_wo_H.append(coords[i].tolist())
            atom_names_wo_H.append(atom.GetSymbol())

    return conf,coords_wo_H,atom_names_wo_H


def smi2conf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,useRandomCoords=True)
        y = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        if y==-1: 
            return None
        else: 
            conf = mol.GetConformer()
            coords = []
            for i, atom in enumerate(mol.GetAtoms()):
                positions = conf.GetAtomPosition(i)
                coords.append([positions.x,positions.y,positions.z])

            return mol, np.array([coords]).reshape(-1,3),conf
    

def check_conf(smi):
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    y = AllChem.EmbedMolecule(mol_h,useRandomCoords=True) #Checking Bad Conformer ID by RDKIT 
    return y
def clean_smiles(smile):
    return smile.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "")

def update_limits(limits, min_crds,max_crds,std):
    if limits[0]['min'] > min_crds[0] - 2*std: limits[0]['min'] = min_crds[0] - 2*std
    if limits[1]['min'] > min_crds[1] - 2*std: limits[1]['min'] = min_crds[1] - 2*std
    if limits[2]['min'] > min_crds[2] - 2*std: limits[2]['min'] = min_crds[2] - 2*std
    if limits[0]['max'] < max_crds[0] + 2*std: limits[0]['max'] = max_crds[0] + 2*std
    if limits[1]['max'] < max_crds[1] + 2*std: limits[1]['max'] = max_crds[1] + 2*std
    if limits[2]['max'] < max_crds[2] + 2*std: limits[2]['max'] = max_crds[2] + 2*std
    return limits


def within_limits(limits, update_limits,min_crds,max_crds,std):
    # this means I'm determining limits on the go, and will take all molecules
    if update_limits:
        return True
    
    within_min_lims = limits[0]['min'] < min_crds[0]-2*std and\
                      limits[1]['min'] < min_crds[1]-2*std and\
                      limits[2]['min'] < min_crds[2]-2*std
    within_max_lims = limits[0]['max'] > max_crds[0]+2*std and\
                      limits[1]['max'] > max_crds[1]+2*std and\
                      limits[2]['max'] > max_crds[2]+2*std
    return within_min_lims and within_max_lims

def get_current_datetime():
    return str(datetime.datetime.now()).split(".")[0][2:]


def determine_dists(atoms, atom_pos):
    if 'H' not in atoms: return True, None, None
    first_h = min(np.argwhere(np.array(atoms)=='H'))[0]

    if len(atoms)!= first_h + 1 and atoms[first_h+1] != 'H': return False, None, None # this means there's an ion, so just skip it
    hydrogens = atom_pos[first_h:]
    heavy_atms = atom_pos[:first_h]
    distances = cdist(hydrogens, heavy_atms)
    corresp_heavy_atm_inds = np.argmin(distances, axis=1)
    return all(corresp_heavy_atm_inds[i] <= corresp_heavy_atm_inds[i+1] for i in range(len(corresp_heavy_atm_inds) - 1)), \
            corresp_heavy_atm_inds, first_h

def check_flawed_mol(coords, bnds):
    """
        if the atom order (from the data blocks) do not correspond to the one generated by the rdkit from the smile
        (even after reordering hydrogen atoms) - e.g. when heavy atoms have random orders  - then just skip it
        Example of flawed molecule is "C#CC#C"
    """
    if not bnds: return False # (e.g. CH4 is just C cause of implicit H, when explicit_hydrogen=False)
    bonded_atms = np.array([[k[0],k[1]] for k in bnds.keys()])
    atom1,atom2 = coords[bonded_atms[:, 0]], coords[bonded_atms[:, 1]]
    dist = np.linalg.norm(atom1-atom2, axis=1)
    return sum(dist > 2.5)

def fix_chiral_atms(atm_symbs, bnds, coords, config=set(['O', 'N', 'C', 'H'])):
    # fixing chiral configuration for tetrahedral C
    for ind, a in enumerate(atm_symbs):
        current_bonds = list(set(np.array([[bnd] for bnd in bnds.keys() if bnd[0] == ind or bnd[1] == ind]).flatten()) - {ind})
        bonded_atms = [atm_symbs[bnd] for bnd in current_bonds]
        if config != set(bonded_atms): continue
        select_atms = [np.argwhere(np.array(bonded_atms) == 'O')[0,0], np.argwhere(np.array(bonded_atms) == 'C')[0,0], np.argwhere(np.array(bonded_atms) == 'N')[0,0]]
        atm_inds = [current_bonds[sa] for sa in select_atms]
        atms = [atm_symbs[sa] for sa in atm_inds]
        assert atms[0] == 'O' and atms[1] == 'C' and atms[2] == 'N'
        m = np.array([coords[atm_inds[0]], coords[atm_inds[1]], coords[atm_inds[2]]])
        if np.linalg.det(m) < 0: coords = coords @ np.array([[1,0,0],[0,1,0],[0,0,-1]])
        m = np.array([coords[atm_inds[0]], coords[atm_inds[1]], coords[atm_inds[2]]])
        return coords
    return coords

def fix_chiral(coords, bonds, atoms, chiral_conf, smi, positive=True):
    possible_branches2inds = {'SP1C':0, 'SP2C':1, 'SP3C':2, 'C':3, 'N':4, 'O':5, 'H':6, 'F':7}
    inds2possible_branches = {v:k for k,v in possible_branches2inds.items()}

    index2connections = {i:[] for i in range(len(atoms))}
    for atm_pair, bnd_t in bonds.items():
        index2connections[atm_pair[0]].append(atm_pair[1])
        index2connections[atm_pair[1]].append(atm_pair[0])
    tetrahedral_configs = []
    for i, connections in index2connections.items():
        connected_atms, connected_atm_inds = [], []
        if len(connections) == 4:
            for atm in connections:
                atm_ = atoms[atm]
                if atm_=='C': 
                    if len(index2connections[atm]) == 2: atm_ = 'SP1C'
                    elif len(index2connections[atm]) == 3: atm_ = 'SP2C'
                    elif len(index2connections[atm]) == 4: atm_ = 'SP3C'
                    connected_atms.append(atm_)
                    connected_atm_inds.append(atm)
                else: connected_atms.append(atm_); connected_atm_inds.append(atm)
            tetrahedral_configs.append([connected_atms, connected_atm_inds, i])
    number_of_chirals = 0
    for tetra_conf in tetrahedral_configs:
        atoms_, atom_inds, center_atm = tetra_conf
        sorted_inds = np.argsort([possible_branches2inds[atm] for atm in atoms_])
        sorted_positions = coords[atom_inds][sorted_inds]
        connected_atms_inds = np.sort([possible_branches2inds[atm] for atm in atoms_])
        ordered_branch_string = ''.join([inds2possible_branches[ind] for ind in connected_atms_inds])
        if ordered_branch_string == chiral_conf:
            # TODO variation of mean of 4 - central carbon
            coords_inds = [atom_inds[i] for i in sorted_inds]
            coords_ = coords[coords_inds]
            coords_ = coords_ - coords[center_atm]
            reflect = (np.linalg.det(coords_[:3]) > 0) ^ positive
            refl_matrix = np.array([[1,0,0],[0,1,0],[0,0,-1 if reflect else 1]]) # reflect coords or leave as is if correct chiral
            coords = coords @ refl_matrix
            coords.tolist()
            return coords
    return coords



def get_grid_limits(Smiles,use_pca=False,std=0.01,limits=None,explicit_aromatic=False,atoms=None, atom_pos=None,
                    explicit_hydrogen=False,arom_cycle_channel=False, fix_chiral_conf=None, cond_variable=None):
  x = {"min":float('inf'),"max":float('-inf')}
  y = {"min":float('inf'),"max":float('-inf')}
  z = {"min":float('inf'),"max":float('-inf')}
  do_update = limits is None or limits[0]['min'] == float('inf')
  grid_limit = [x,y,z]
  all_coords = []
  all_smiles = []
  all_atom_symbols = []
  all_bonds = []
  removed_smiles = 0
  removed_reason2count = {"Chem.MolFromSmiles retured none":0, "Could not kekulize":0, 
                          "no atms rdkit mol != atom positions":0, "AllChem.EmbedMolecule failed":0,
                          "Ionization":0, "Different rdkit/atm block order":0,"Not within limits":0}
  all_cond_variables = []
  for ind, smi in enumerate(Smiles):
    # if smi != "C[C@@]12C[C@@](C1)(O2)C(=O)N" and smi != "CC12CC(C1)(O2)C(N)=O": continue
    # if smi != "C#C" and smi != "C#C": continue
    if ind % 1000 ==0 and ind != 0:
        print(ind, " smiles elapsed; {} removed because of limits overflow/containing ions. {} remaining".format(removed_smiles, len(all_coords)))
        logging.info("{}: {} smiles elapsed".format(get_current_datetime(), ind))
        for k,v in removed_reason2count.items():
            print(k, v)
    coords = []
    if 'c' in smi and not explicit_aromatic: print("IN get_grid_limits ENCOUNTERED EXPLICIT AROMATIC FORMULATION"
                                                   "explicit_aromatic=False, BUT "); breakpoint()
    mol = Chem.MolFromSmiles(smi); 
    if mol is None:removed_smiles+=1;  removed_reason2count["Chem.MolFromSmiles retured none"] +=1; continue # with explicit hydrogens, some more invalid molecules came up :)
    elif explicit_hydrogen: mol = Chem.AddHs(mol)

    if not explicit_aromatic: 
        try:
            Chem.Kekulize(mol) # swithc to bdn type 1/2 representation
        except:
            removed_smiles+=1; removed_reason2count["Could not kekulize"]+=1; continue

    if (atom_pos is not None and not explicit_hydrogen) and sum([A != 'H' for A in atoms[ind]]) != mol.GetNumAtoms(): removed_smiles+=1;removed_reason2count["no atms rdkit mol != atom positions"]+=1; continue
    elif (atom_pos is not None and explicit_hydrogen) and (mol.GetNumAtoms() != len(atom_pos[ind])): removed_smiles+=1;removed_reason2count["no atms rdkit mol != atom positions"]+=1; continue



    ionized_mol = "+" in smi or "-" in smi # ! (wait, but I still have ionized in data??) if it's ionized, don't touch the positions
    if ionized_mol and not explicit_hydrogen: removed_smiles+=1; removed_reason2count["Ionization"]+=1; continue     # ! when not explicitly modelling hydrogens, it doesn't make sense to use ionized molecules
    if atom_pos is not None and not ionized_mol: 
        coords = atom_pos[ind]; conf=None; 
        if not explicit_hydrogen: coords = np.stack([crd for crd,atm in zip(atom_pos[ind], atoms[ind]) if atm !='H'])
    else:
        # If I don't use data's coordinates, I need to generated (RDKit may not be able to in some cases)
        check = AllChem.EmbedMolecule(mol,useRandomCoords=True)
        if check==-1: removed_smiles+=1; removed_reason2count["AllChem.EmbedMolecule failed"]+=1; continue
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            positions = conf.GetAtomPosition(i)
            coords.append([positions.x,positions.y,positions.z])
        coords = np.array([coords]).reshape(-1,3)

    if use_pca: coords_aligned = align(coords)
    else: coords_aligned = coords

    # * all hydrogen atom positions are usually ordered according to heavy-atm order, first come/first served
    # * some are not - if they are not, re-orderd the positions
    if (explicit_hydrogen and not ionized_mol) and atom_pos is not None:
        correctly_ordered, current_order, first_atm_pos = determine_dists(atoms[ind], atom_pos[ind])
        # first_atm_pos is not None means there's an ion (e.g. H3O+), so let the next part of the code just throw it
        if not correctly_ordered and current_order is None: removed_smiles+=1; removed_reason2count['Ionization']+=1; continue # if the H is literally randomly placed, just skip it...
        if first_atm_pos is not None and not correctly_ordered:
            atm_order = list(range(first_atm_pos)) # leave heavy atms as they are, orders Hs to correspond
            correct_h_order = np.argsort(current_order) + first_atm_pos
            correct_h_order = list(correct_h_order)
            atm_order.extend(correct_h_order)
            coords_aligned = np.array([coords_aligned[i] for i in atm_order])



    conf,new_coords,atom_symbols = update_mol_positions_whydr(mol,conf,coords_aligned) if explicit_hydrogen else update_mol_positions(mol,conf,coords_aligned)
    # mol_ = read_smiles(smi, reinterpret_aromatic=False,explicit_hydrogen=explicit_hydrogen)
    # edge_attr = nx.get_edge_attributes(mol_, name = "order")
    # * GET BONDS without using read_smiles which somehow introduces hydrogen atoms
    edge_attr = {}
    for bnd in mol.GetBonds():
        # print([bnd.GetBeginAtomIdx(),bnd.GetEndAtomIdx()], bnd.GetBondType())
        edge_attr[(bnd.GetBeginAtomIdx(),bnd.GetEndAtomIdx())] = bndname2bndnumber[str(bnd.GetBondType())]
    min_crds, max_crds = np.min(coords_aligned, axis=0), np.max(coords_aligned, axis=0)
    if fix_chiral_conf is not None: new_coords = fix_chiral(np.array(new_coords), edge_attr, atom_symbols, fix_chiral_conf, smi)
    check_flawed = check_flawed_mol(coords_aligned, edge_attr)
    if check_flawed: removed_smiles+=1; removed_reason2count["Different rdkit/atm block order"]+=1;continue


    if not within_limits(limits, do_update, min_crds,max_crds,std): removed_reason2count["Not within limits"]+=1;removed_smiles+=1; continue
    limits = update_limits(limits,min_crds,max_crds, std) if do_update else limits
    coords = np.array(new_coords).reshape(-1,3)
    ssr = Chem.GetSymmSSSR(mol)
    all_smiles.append(smi)
    if arom_cycle_channel:
        Chem.SanitizeMol(mol)
        for cycle in ssr:
            if str(mol.GetBondWithIdx(cycle[0]).GetBondType()) == "AROMATIC":
                aromatic_no = len(cycle)
                if aromatic_no not in [5,6]: continue # do not consider other, way less frequent arom rings
                aromatic_letter = "X" if aromatic_no == 5 else "Y"
                arom_position = [coords[i] for i in list(cycle)]
                arom_position = np.mean(np.stack(arom_position), axis=0)
                coords = np.concatenate([coords, arom_position[None, :]])
                atom_symbols.append(aromatic_letter)

    all_coords.append(coords)
    all_atom_symbols.append(atom_symbols)
    all_bonds.append(edge_attr)
    if cond_variable is not None: all_cond_variables.append(cond_variable[ind])
    # if "H" in smi: breakpoint()

    # * visualize some molecule
    # if not correctly_ordered:
    # if smi == : 

    # TODO get also the bonds to be sure
  print("{} smiles have been removed. Remaining molecules {}.".format(removed_smiles,len(all_coords)))

  

  x,y,z = limits[0], limits[1], limits[2]
  return x,y,z,all_coords,all_smiles,all_atom_symbols, all_bonds, all_cond_variables

def get_min_atom_dists(all_coords_all_molecules):
    from sklearn.metrics.pairwise import euclidean_distances

    for molecule in all_coords_all_molecules:

        coords_ =  np.stack(molecule)
        min_dist = np.min(euclidean_distances(coords_, coords_) + np.eye(coords_.shape[0]) * 100)
        print(len(coords_),min_dist)
    exit(1)


def get_smiles_kekulize_remove_unkekulizable(smiles, atom_pos, atoms):
    """
        smiles: list of smiles or list of (three) lists of smiles (e.g. in case of trian/val/test splits)
        atom_pos: if None, extract atom positiosn using rdkit; else, use the provided atom positions;
                  if it's not none, same structure (i.e. list of lists or list of atom positions) as smiles
        atoms: same as above: idenfitifer atom symbol for each atom position in atom_pos
    """
    explicit_atom_pos = atom_pos is not None
    if type(smiles[0]) == str:
        train_test_val=False
        smiles = [smiles] if type[smiles[0]] == str else smiles
        atom_pos = [atom_pos]
        atoms = [atoms]
    else:
        train_test_val = True

    kekulized_smiles, kekulized_atom_pos, kekulized_atoms = [],[],[]
    for sm, atm_pos, atm in zip(smiles, atom_pos, atoms):
        kekulized_smiles_, kekulized_atom_pos_, kekulized_atoms_ = [],[],[]
        for ind, s in enumerate(sm):
            # some molecules are ionized, so skip them
            mol = Chem.MolFromSmiles(s)
            try:
                kekulized_smiles_.append(Chem.MolToSmiles(mol,kekuleSmiles=True))
                if explicit_atom_pos: kekulized_atom_pos_.append(atm_pos[ind]); kekulized_atoms_.append(atm[ind])
            except:
                continue
        kekulized_smiles.append(kekulized_smiles_)
        kekulized_atom_pos.append(kekulized_atom_pos_)
        kekulized_atoms.append(kekulized_atoms_)

    if not explicit_atom_pos: kekulized_atom_pos = None; kekulized_atoms = None
    if not train_test_val:
        kekulized_smiles = kekulized_smiles[0]
        if explicit_atom_pos: kekulized_atom_pos = kekulized_atom_pos[0]; kekulized_atoms= kekulized_atoms[0]
    return kekulized_smiles, kekulized_atom_pos, kekulized_atoms

def get_smiles_kekulize(smiles):
    # replace lower-case atom symbols in aromatic rings with explicit single/double bonds
    kekulize_smiles = []
    unkekulized=0
    for s in smiles:
        # some molecules are ionized, so skip them
        try:
            mol = Chem.MolFromSmiles(s)
            kekulize_smiles.append(Chem.MolToSmiles(mol,kekuleSmiles=True))
        except:
            unkekulized+=1
            continue
    print("Due to ion-containing or other problems, {} have been removed when trying to kekulize".format(unkekulized))
    return kekulize_smiles

def extract_mulitple_smile_datasets_limit(Smiles, use_pca, std, limits,explicit_aromatic,
                                          atoms=None, atom_pos=None, explicit_hydrogen=False, arom_cycle_channel=False, fix_chiral_conf=None, cond_variable=None):
    """
        use this to extract grid limits for multiple data sets (Smiles is a list of lists of smiles);
        therefore, update limits for all data
    """
    if type(Smiles[0]) == str: return get_grid_limits(Smiles, use_pca=use_pca, std=std, limits=limits,explicit_aromatic=explicit_aromatic,atoms=atoms, atom_pos=atom_pos
                                                      ,explicit_hydrogen=explicit_hydrogen, arom_cycle_channel=arom_cycle_channel, fix_chiral_conf=fix_chiral_conf, cond_variable=cond_variable)

    else:
        x = {"min":float('inf'),"max":float('-inf')}
        y = {"min":float('inf'),"max":float('-inf')}
        z = {"min":float('inf'),"max":float('-inf')}
        do_update = limits is None or limits[0]['min'] == float('inf')
        if not do_update: print("WARNING! YOU ARE IN extract_mulitple_smile_datasets_limit AND HAVE SET LIMITS. ARE YOU SURE?")
        all_coords_, all_smiles_,all_atom_symbols_,all_bonds_, no_of_elements_, all_cond_variables_ = [],[],[],[],[],[]

        for ind, smile_list in enumerate(Smiles):
            atoms_, atom_pos_ = (atoms[ind], atom_pos[ind]) if atoms is not None else (None, None)
            cond_variable_ = cond_variable[ind] if cond_variable is not None else None
            limits_ = [x.copy(),y.copy(),z.copy()] if do_update else copy.deepcopy(limits)
            x_lim, y_lim, z_lim, all_coords, all_smiles,all_atom_symbols,all_bonds,all_cond_variables =\
                  get_grid_limits(smile_list, use_pca=use_pca, std=std, limits=limits_,explicit_aromatic=explicit_aromatic,
                                  atoms=atoms_, atom_pos=atom_pos_,explicit_hydrogen=explicit_hydrogen,arom_cycle_channel=arom_cycle_channel,fix_chiral_conf=fix_chiral_conf, cond_variable=cond_variable_)
            
            min_crds,max_crds = [x_lim["min"],y_lim["min"],z_lim["min"]],[x_lim["max"],y_lim["max"],z_lim["max"]]
            limits = update_limits(limits,min_crds,max_crds, std) if do_update else limits
            all_coords_.append(all_coords)
            all_smiles_.append(all_smiles)
            all_atom_symbols_.append(all_atom_symbols)
            all_bonds_.append(all_bonds)
            all_cond_variables_.append(all_cond_variables)
    return limits[0],limits[1],limits[2], all_coords_, all_smiles_,all_atom_symbols_,all_bonds_,all_cond_variables_

def extract_aligned_mol_coords(Smiles,data_type,std=0.05,use_pca=False,rescale=False,resolution=0.1, limits=None, explicit_aromatic=False,
                               atoms=None, atom_pos=None,explicit_hydrogen=False,arom_cycle_channel=False, fix_chiral_conf=None, cond_variable=None):
    """
        extract all molecules that after translation (pos-pos.mean) and rotation by PCA are within the specified
        limits and save them; alongside, get the grids x,y,z
    """


    # * maybe do the step below directly in the get_grid_limits method
    # if not explicit_aromatic: Smiles, atom_pos, atoms = get_smiles_kekulize_remove_unkekulizable(Smiles,atom_pos,atoms)

    Unique_elements = UNQ_elements.get(data_type)
    x_lim, y_lim, z_lim, all_coords, all_smiles,all_atom_symbols,all_bonds,cond_variable = \
         extract_mulitple_smile_datasets_limit(Smiles, use_pca=use_pca, std=std, limits=limits,
                                               explicit_aromatic=explicit_aromatic,atoms=atoms, atom_pos=atom_pos,
                                               explicit_hydrogen=explicit_hydrogen, arom_cycle_channel=arom_cycle_channel, 
                                               fix_chiral_conf=fix_chiral_conf, cond_variable=cond_variable)
    # x_lim, y_lim, z_lim, all_coords, all_smiles,all_atom_symbols,all_bonds = \
    #     get_grid_limits(Smiles, use_pca=use_pca, std=std, limits=limits,explicit_aromatic=explicit_aromatic)
    # TODO: fixing it this way means you have different resolutions for different axis.
    bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(resolution*8))*8), [x_lim,y_lim,z_lim])]
    x,y,z = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),
                              bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')
    return x,y,z,all_coords,all_smiles, all_atom_symbols,all_bonds,cond_variable


def get_channel_mol_reps(Smiles,data_type,std=0.05,use_pca=False,rescale=False,
                 resolution=0.1, limits=None, return_true_mens=False, data_cutoff=0.1,
                 discretized_center=True, show_molecules=False):
    """
        Generates molecular representations for a list of SMILES strings.

        Args:
            Smiles (List[str]): A list of SMILES strings.
            data_type (str): A string representing the type of data.
            std (float, optional): The standard deviation of the Gaussian distribution used to generate the molecular representations. Defaults to 0.05.
            use_pca (bool, optional): Whether to use PCA to reduce the dimensionality of the molecular representations. Defaults to False.
            rescale (bool, optional): Whether to rescale the molecular representations. Defaults to False.
            resolution (float, optional): The resolution of the 3D grid used to generate the molecular representations. Defaults to 0.1.
            limits (Optional[Dict[str, Union[float, List[float]]]], optional): A dictionary containing the limits of the molecular space. Defaults to None.
            return_true_mens (bool, optional): Whether to return the true molecular mean positions. Defaults to False.
            data_cutoff (float, optional): The cutoff value for the molecular field pdf values. Defaults to 0.1.

        Returns:
            Tuple[List[str], np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the final SMILES strings, the molecular representations, the unique elements in the molecules, the x, y, and z arrays that define the 3D grid, the coordinates of the atoms in the molecules, and the atom lists for each molecule.
    """
    if return_true_mens:
        final_smiles, final_data, Unique_elements,x,y,z,all_coords,all_atom_lists, all_bond_disctionaries = get_mol_reps(Smiles,data_type,std=std,use_pca=use_pca,rescale=rescale,
                    resolution=resolution, limits=limits, return_true_mens=return_true_mens, data_cutoff=data_cutoff,
                    discretized_center=discretized_center)
    else:
        final_smiles, final_data, Unique_elements,x,y,z = get_mol_reps(Smiles,data_type,std=std,use_pca=use_pca,rescale=rescale,
                    resolution=resolution, limits=limits, return_true_mens=return_true_mens, data_cutoff=data_cutoff)
    atm2channel = {"C":0,"O":1,"N":2,"F":3}


    block_shape = x.shape
    x,y,z = x.flatten(),y.flatten(),z.flatten()
    xyz = np.stack([x,y,z],axis=-1)

    new_densities = []

    start_time = time.time()

    # for each input atom, add additional channels for each bond type
    for i,atom_list in enumerate(all_atom_lists):
        atm_density = final_data[i]
        bond_density = np.zeros((3, atm_density.shape[1],atm_density.shape[2],atm_density.shape[3]))


def get_stub_mol_reps(Smiles,data_type,std=0.05,use_pca=False,rescale=False,
                 resolution=0.1, limits=None, return_true_mens=False, data_cutoff=0.1,
                 discretized_center=True, show_molecules=False):

    """
        Generates molecular representations for a list of SMILES strings.

        Args:
            Smiles (List[str]): A list of SMILES strings.
            data_type (str): A string representing the type of data.
            std (float, optional): The standard deviation of the Gaussian distribution used to generate the molecular representations. Defaults to 0.05.
            use_pca (bool, optional): Whether to use PCA to reduce the dimensionality of the molecular representations. Defaults to False.
            rescale (bool, optional): Whether to rescale the molecular representations. Defaults to False.
            resolution (float, optional): The resolution of the 3D grid used to generate the molecular representations. Defaults to 0.1.
            limits (Optional[Dict[str, Union[float, List[float]]]], optional): A dictionary containing the limits of the molecular space. Defaults to None.
            return_true_mens (bool, optional): Whether to return the true molecular mean positions. Defaults to False.
            data_cutoff (float, optional): The cutoff value for the molecular field pdf values. Defaults to 0.1.

        Returns:
            Tuple[List[str], np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the final SMILES strings, the molecular representations, the unique elements in the molecules, the x, y, and z arrays that define the 3D grid, the coordinates of the atoms in the molecules, and the atom lists for each molecule.
    """
    if return_true_mens:
        final_smiles, final_data, Unique_elements,x,y,z,all_coords,all_atom_lists, all_bond_disctionaries = get_mol_reps(Smiles,data_type,std=std,use_pca=use_pca,rescale=rescale,
                 resolution=resolution, limits=limits, return_true_mens=return_true_mens, data_cutoff=data_cutoff,
                 discretized_center=discretized_center)
    else:
        final_smiles, final_data, Unique_elements,x,y,z = get_mol_reps(Smiles,data_type,std=std,use_pca=use_pca,rescale=rescale,
                 resolution=resolution, limits=limits, return_true_mens=return_true_mens, data_cutoff=data_cutoff)
    atm2channel = {"C":0,"O":1,"N":2,"F":3}
    
    # xyz = np.stack([x.flatten(),y.flatten(),z.flatten()],axis=-1)
    # coord_atm = all_coords[-1][0]
    # distances = np.linalg.norm(coord_atm - xyz, axis=1)
    # when gaussians are centered on the true mean positions, difference between voxel+1/voxel-1 is about 0.03 (0.86/0.83)
    # shape = x.shape
    # closest_index = np.argmin(distances)
    # closest_indices = np.unravel_index(closest_index, shape)
    # going +/- 1 in some direction will be equal if parameter discretized_center=True
    # print(final_data[-1][0][closest_indices[0]][closest_indices[1]][closest_indices[2]])
    # print(final_data[-1][0][closest_indices[0]][closest_indices[1]][closest_indices[2]+1])
    # print(final_data[-1][0][closest_indices[0]][closest_indices[1]][closest_indices[2]-1])

    bond_type_2_stud_values = {1:0.1,1.5:1,2:0.2,3:0.3,4:0.4}

    block_shape = x.shape
    x,y,z = x.flatten(),y.flatten(),z.flatten()
    xyz = np.stack([x,y,z],axis=-1)

    new_densities = []

    start_time = time.time()


    all_mols_all_atom_line_points = []
    all_mols_all_corresponding_channel_numbers = []
    all_mols_all_corresponding_stud_values = []
    all_mols_batch_indices = []

    for ind, (smile, density, atoms, atom_bonds,coords) in enumerate(zip(final_smiles, final_data, all_atom_lists, all_bond_disctionaries, all_coords)):
        all_atom_line_points = []
        all_corresponding_channel_numbers = []
        all_corresponding_stud_values = []
        for atm_pair,bond_type in atom_bonds.items():
            atm1,atm2 = atm_pair
            channel_atm1 = atm2channel.get(atoms[atm1])
            channel_atm2 = atm2channel.get(atoms[atm2])

            atm1_coord = coords[atm1]
            atm2_coord = coords[atm2]
            mean_pos = (atm1_coord+atm2_coord)/2
            direction_atm1 = mean_pos-atm1_coord
            direction_atm2 = atm2_coord-mean_pos
            points_on_line1 = np.linspace(0,1,10)[:, None] * direction_atm1 + atm1_coord
            points_on_line2 = np.linspace(0,1,10)[:, None] * direction_atm2 + mean_pos


            all_atom_line_points.extend(list(points_on_line1))
            all_atom_line_points.extend(list(points_on_line2))

            all_corresponding_channel_numbers.extend([channel_atm1]*10)
            all_corresponding_channel_numbers.extend([channel_atm2]*10)

            all_corresponding_stud_values.extend([bond_type_2_stud_values[bond_type]]*20)
            

        
        all_mols_all_atom_line_points.extend(all_atom_line_points)
        all_mols_all_corresponding_channel_numbers.extend(all_corresponding_channel_numbers)
        all_mols_all_corresponding_stud_values.extend(all_corresponding_stud_values)
        all_mols_batch_indices.extend([ind]*len(all_corresponding_stud_values))


    all_mols_all_corresponding_stud_values = np.array(all_mols_all_corresponding_stud_values)
    all_mols_all_atom_line_points = np.array(all_mols_all_atom_line_points)
    all_mols_all_corresponding_channel_numbers = np.array(all_mols_all_corresponding_channel_numbers)
    all_mols_batch_indices = np.array(all_mols_batch_indices)


    
    distances = np.linalg.norm(all_mols_all_atom_line_points[:, None,:] - xyz, axis=2)

    # get the minimum distances
    closest_grid_position = np.argwhere(distances<0.2)


    row_indices = closest_grid_position[:,0]
    col_indices = closest_grid_position[:,1]




    final_data = np.array(final_data)




    all_mols_all_corresponding_channel_numbers = all_mols_all_corresponding_channel_numbers[[row_indices]].flatten()
    all_mols_all_corresponding_stud_values = all_mols_all_corresponding_stud_values[[row_indices]].flatten()
    all_mols_batch_indices = all_mols_batch_indices[[row_indices]].flatten()



    closest_grid_indices = np.unravel_index(col_indices, block_shape)


    dens_and_new_vals = final_data[all_mols_batch_indices,all_mols_all_corresponding_channel_numbers,closest_grid_indices[0],closest_grid_indices[1],closest_grid_indices[2]]
    dens_and_new_vals = np.vstack([dens_and_new_vals,all_mols_all_corresponding_stud_values])

    # get the maximum between the value already there and the stub value
    final_data[all_mols_batch_indices,all_mols_all_corresponding_channel_numbers,
               closest_grid_indices[0],closest_grid_indices[1],closest_grid_indices[2]] = np.max(dens_and_new_vals,axis=0)
    # create a 3D plot of the denisty, having 4 subplots for each of the 4 channels in the denisty
    if show_molecules:
        for fd in final_data:
            fig = plt.figure()
            for i in range(4):
                # ax = fig.add_subplot(111, projection='3d')

                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                flat_dens = fd[i].flatten()
                ax.scatter(x[flat_dens>0.05],y[flat_dens>0.05],z[flat_dens>0.05],c=flat_dens[flat_dens>0.05])
                ax.axes.set_xlim3d(left=x.min(), right=x.max())
                ax.axes.set_ylim3d(bottom=y.min(), top=y.max())
                ax.axes.set_zlim3d(bottom=z.min(), top=z.max())
            plt.suptitle("Density of the molecule\n" + smile)
            plt.show()
    print("Finished adding the stubs to the densities in {:.3f} seconds".format(time.time()-start_time))

    return final_smiles, new_densities, Unique_elements,x,y,z,all_coords,all_atom_lists, all_bond_disctionaries





def discretize_position(all_coords, grid):
    """
        Discretizes a position in a grid.

        Args:
            position (np.ndarray): The position to discretize.
            grid (np.ndarray): The grid.

        Returns:
            np.ndarray: The discretized position.
    """
    all_coords = all_coords


    discretized_coord_positions = []
    for coord in all_coords:
        closest_grid_position = np.argmin(np.linalg.norm(coord[:, None,:] - grid, axis=2),axis=1)
        closest_grid_values = grid[closest_grid_position]
        discretized_coord_positions.append(closest_grid_values)
        

    return discretized_coord_positions


def concatenate_bond_channels(density, bond_types, atom_coords,
                              pos,data_cutoff,cov,ignore_aromatic,normalize01=False):
    """
        Concatenates the bond channels to the density.
    """
    h,w,l = density.shape[1],density.shape[2],density.shape[3]
    found_aromatic=False

    bond_channels = np.zeros((3, h,w,l)) if ignore_aromatic else np.zeros((4, h,w,l))

    bond_type_2_number_of_bonds = {1:0,2:0,3:0,1.5:0}
    for _, bond_type in bond_types.items():
        bond_type_2_number_of_bonds[bond_type] += 1
    for involved_atms in bond_types.keys():
        bond_type = bond_types[involved_atms]
        mean_pos = np.mean(np.stack([atom_coords[involved_atms[0]], atom_coords[involved_atms[1]]]),axis=0)
        rv = multivariate_normal([mean_pos[0],mean_pos[1],mean_pos[2]], cov)
        temp_dens = rv.pdf(pos).reshape(h,w,l)
        temp_dens[temp_dens<data_cutoff] = 0.0
        if bond_type == 1.5 and ignore_aromatic:
            print("!!!WARNING!!! ENCOUNTERED AROMATIC BUT ignore_aromatic=True in concatenate_bond_channels under utils.py"
                  ". SOMETHING IS LIKELY OFF")
        elif bond_type==1.5 and not ignore_aromatic:
            bond_channels[3] += temp_dens/np.max(temp_dens) if normalize01 else temp_dens
        else:
            bond_channels[bond_type-1] += temp_dens/np.max(temp_dens) if normalize01 else temp_dens



    # if found_aromatic:
    #     mean_positions_all = [np.mean(np.stack([atom_coords[involved_atms[0]], \
    #                              atom_coords[involved_atms[1]]]),axis=0) for involved_atms in bond_types.keys()]
    #     # get pairwise distances
    #     pairwise_distances = np.linalg.norm(np.array(mean_positions_all)[:,None,:] - np.array(mean_positions_all)[None,:,:],axis=2)


        
    for bnd_type, num_bnds in bond_type_2_number_of_bonds.items():
        if bnd_type == 1.5 and ignore_aromatic: continue
        if normalize01: continue
        if num_bnds > 0: bond_channels[bnd_type-1] /= num_bnds
    

    # for bt in bond_types.items():
    #     print("CE MORTII MATI MAIV VREI")
    #     ()

    return np.concatenate([density, bond_channels], axis=0)




def get_actual_bonds_through_pdf(candidaten_bnds,field,x,y,z,var=0.05,normalize01=False):
    positions = np.stack([x.flatten(),y.flatten(),z.flatten()],axis=1)
    actual_bonds = []

    # get the bond channels that have connections and determine the mean-max values


    # TODO compute the constant s.t. a pdf with var=0.05 would have a max val determined in the field; then, 
    # compute the pdf value 15 pm away, divide by said constant and that's ur threshold
    per_bnd_thresholds = []
    for bnd_in, bnd_ch in enumerate(field[4:]):
        if normalize01:
            per_bnd_thresholds.append(0.75)
            continue
        mean, max_val = np.mean(bnd_ch), np.max(bnd_ch)
        per_bnd_thresholds.append((max_val - mean)*0.5) if max_val - mean > 0.5 else per_bnd_thresholds.append(10000)

    candidate_excluded=True
    for index_cb, cb in enumerate(candidaten_bnds):
        mean_position = (cb[2] + cb[3])/2
        for bnd_ind, bond_chn in enumerate(field[4:]):
            if per_bnd_thresholds[bnd_ind] > 100: continue
            distances = np.linalg.norm(mean_position - positions,axis=1)
            indices = np.argwhere(distances < 0.45)
            indices = np.unravel_index(indices,field[0].shape)
            distances = distances[distances<0.45]
            # weights = get_pdf_probs(distances,var)
            vals = bond_chn[indices[0], indices[1],indices[2]]
            # total_pdf = sum(np.squeeze(vals)*weights)
            # print(total_pdf)
            total_pdf = np.max(vals)
            # print(total_pdf, bnd_ind, index_cb, )
            if total_pdf > per_bnd_thresholds[bnd_ind]: actual_bonds.append([cb[0],cb[1],bnd_ind+1]); candidate_excluded=False
            # plot_channel(bond_chn, x,y,z,mean_position, cb[2],cb[3])
        candidate_excluded=True
        # print("\n")
    return actual_bonds


def get_sphere_centers_deterministic(tensor,x_grid, y_grid, z_grid,normalize01, set_threshold=None):
    """
        deterministic algorithm to find the centers of spheres in a 3D tensor. Since the average pdf value is generally
        in the ballpark of 0, we can use (max_pdf - avg_pdf) to determine if that distance is say > 0.5, 
        therefore identifying that we indeed have some sort of spheres. Then retrieve elements that cover e.g.
        [4/5 * max_pdf_value, max_pdf_value]
    """
    tensor = torch.tensor(tensor) if type(tensor) != torch.Tensor else tensor
    x_resolution, y_resolution, z_resolution = x_grid[1,0,0] - x_grid[0,0,0], y_grid[0,1,0] - y_grid[0,0,0], z_grid[0,0,1] - z_grid[0,0,0]
    
    avg_pdf, max_pdf = torch.mean(tensor), torch.max(tensor)
    if exists(set_threshold):
        threshold = set_threshold
    elif normalize01:
        threshold = 0.75
    elif max_pdf - avg_pdf  > 0.5:
        threshold = (3.75/5)*(max_pdf-avg_pdf)
    else:
        return []

    starting_points = torch.argwhere(tensor >= threshold).cpu().detach().numpy()

    if len(starting_points) == 0:
        return []

    found_points = starting_points[None,0]
    other_points = starting_points[1:]

    spheres = []
    spheres_pdfs = []

    # when there's only 1 found point, add it as a sphere
    if not len(other_points): 
        spheres = [[found_points[0]]]
        for csf in spheres[0]:
            sphere_pdf = [tensor[csf[0],csf[1],csf[2]].item()]
        spheres_pdfs.append(sphere_pdf)
    while len(other_points):

        found_vicinity_points=True
        current_sphere_points = [found_points[0]]



        while found_vicinity_points:
            # compute pair-wise euclidian distances between points
            distances = np.linalg.norm(found_points[:, None] - other_points, axis=2)
            neighborhood = np.argwhere(distances <= 1.5)
            if len(neighborhood):
                found_vicinity_points = True
                neighborhood = neighborhood[:,1]
                current_sphere_points = np.concatenate([current_sphere_points, other_points[neighborhood]])
                found_points = current_sphere_points
                other_points = np.delete(other_points, neighborhood, axis=0)
            else:
                found_vicinity_points=False
        spheres.append(current_sphere_points)
        sphere_pdf = []
        for csf in current_sphere_points:
            sphere_pdf.append(tensor[csf[0],csf[1],csf[2]].item())
        spheres_pdfs.append(sphere_pdf)
        if len(other_points) == 1:
            spheres.append(other_points)
            sphere_pdf = []
            for csf in other_points:
                sphere_pdf.append(tensor[csf[0],csf[1],csf[2]].item())
            spheres_pdfs.append(sphere_pdf)
            other_points = []
        elif len(other_points):
            found_points = other_points[None, 0]
            other_points = other_points[1:]

    # print("starting_points, spheres",len(starting_points), len(spheres))
    # if len(starting_points) == 1:
    sphere_centers = []
    for sphere_points, sphere_pdf in zip(spheres, spheres_pdfs):

        sphere_weight_points = np.array(sphere_pdf)/np.sum(sphere_pdf)
        center_sphere_ = np.sum(sphere_points.T * sphere_weight_points, axis=1) if len(sphere_points) > 1\
              else sphere_points[0] 
        # sphere_centers.append(center_sphere)        


        # center_sphere = np.round(np.mean(sphere_points, axis=0))


        center_sphere = center_sphere_.astype(int)

        center_sphere_coordinates = x_grid[center_sphere[0],center_sphere[1],center_sphere[2]], \
                                    y_grid[center_sphere[0],center_sphere[1],center_sphere[2]], \
                                    z_grid[center_sphere[0],center_sphere[1],center_sphere[2]]
        
        leftover_decimals = center_sphere_ - center_sphere
        leftover_per_axis = leftover_decimals * np.array([x_resolution, y_resolution, z_resolution])

        center_sphere_coordinates = [center_sphere_coordinates[i] + leftover_per_axis[i] for i in range(3)]


        sphere_centers.append(list(center_sphere_coordinates))
    return np.array(sphere_centers)



def get_potential_atm_pair_positions(atm_bnd_pos, mol_bnd_symb):
    """
        atm_bnd_pos: (N,3) this variable may or may not contain bond positions, depending on the method;
                     see fit_pred_field_sep_chn_batch documentation, determine_bonds argument
    """
    unique_atms = ["C","O","N","F"]
    atom_indices = []
    for ind, symb in enumerate(mol_bnd_symb):
        if symb in unique_atms: 
            atom_indices.append(ind)
    atom_positions = atm_bnd_pos[atom_indices]
    atom_symbols = [mol_bnd_symb[i] for i in atom_indices]

    new2old_index = {new:old for new,old in enumerate(atom_indices)}
    
    distances = np.linalg.norm(atom_positions - atom_positions[:,None],axis=2)
    # get dists in picometers
    distances *= 100

    candidate_bnds = []
    for i in range(len(distances)):
        for j in range(i+1,len(distances)):
            atm1,atm2 = atom_symbols[i],atom_symbols[j]
            #  < bonds3[atm1][atm2]-7
            if bonds1[atm1][atm2]+25 > distances[i,j]: candidate_bnds.append((i,j,atom_positions[i],atom_positions[j]))
    return candidate_bnds, new2old_index

def extract_all_pos(channels,x_grid,y_grid,z_grid):
    unique_atoms = ["C","O","N","F"]
    initial_guesses = []
    atom_symbols = []
    atom_positions = []
    for atom_type in range(0,4):
        initial_guesses=get_sphere_centers_deterministic(channels[atom_type],x_grid,y_grid,z_grid, normalize01=True)
        atom_symbols.extend([unique_atoms[atom_type]]*len(initial_guesses))
        atom_positions.extend(initial_guesses)
    return np.stack(atom_positions), np.array(atom_symbols)

def extract_lines(actual_bnd, atom_positions):
    positions = {1:[],2:[],3:[]}
    for bnd in actual_bnd:
        positions[bnd[2]].append(np.stack([atom_positions[bnd[0]], atom_positions[bnd[1]] ]))
    return positions


def plot_multi_channel(mdl=None, grids=None):
    mdl = "misc/some_imgs_025_small8.bin" if mdl is None else mdl
    x_grid, y_grid, z_grid = get_grids(mdl, args.data_type) if grids is None else grids
    imgs = pickle.load(open(mdl, "rb")) if type(mdl)==str else mdl[None,:]
    channel_symbols = ["C","O","N","F",'BND1', 'BND2','BND3']
    for img_index, img_ in enumerate(imgs):
        atom_positions, atom_symbols = extract_all_pos(img_,x_grid,y_grid,z_grid)
        candidaten_bnds,new2old_index = get_potential_atm_pair_positions(atom_positions, atom_symbols)
        candidate_bnd_positions = np.stack([(cb[2]+cb[3])/2 for cb in candidaten_bnds]) if len(candidaten_bnds) else None
        img_ = img_.detach().cpu().numpy() if type(img_) == torch.Tensor else img_
        actual_bnd = get_actual_bonds_through_pdf(candidaten_bnds, img_,x_grid,y_grid,z_grid,var=0.05,normalize01=True)
        lines = extract_lines(actual_bnd, atom_positions)
        if len(atom_positions) ==0:
            atom_positions = None
            candidate_bnd_positions= None
        fig = plt.figure()
        for ind, ch in enumerate(img_):

            if ind < 4:
                some_positions = atom_positions[atom_symbols == channel_symbols[ind]] if atom_positions is not None else None
                plot_channel(ch, x_grid,y_grid,z_grid, threshold=0.55, title=f"image index {img_index}\nchannel {channel_symbols[ind]}", some_position=some_positions, specific_subplot=240+ind+1, figure=fig)
            else:
                plot_channel(ch, x_grid,y_grid,z_grid, threshold=0.55, title=f"image index {img_index}\nchannel {channel_symbols[ind]}", some_position=candidate_bnd_positions,
                             lines=lines[ind-3],specific_subplot=240 + ind+1, figure=fig)
        plt.show()


def get_mol_reps_w_bonds(Smiles,data_type,std=0.05,use_pca=False,rescale=False,
                        resolution=0.1, limits=None, return_true_mens=False, data_cutoff=0.1,
                        discretized_center=False, bond_information='rdkit_predicted', ignore_aromatic=True,
                        normalize01=False, explicit_aromatic=False, explicit_hydrogen=False):
    """
        with these, the bonds are separately modeled using channels 4-7
    """
    if not explicit_aromatic: Smiles = get_smiles_kekulize(Smiles)
    Unique_elements = UNQ_elements.get(data_type + "-hydr" if explicit_hydrogen else data_type)
    x_lim,y_lim,z_lim,all_coords,all_smiles,all_atom_symbols,all_bonds = get_grid_limits(Smiles,use_pca=use_pca, std=std, 
                                                                    limits=limits, explicit_aromatic=explicit_aromatic,explicit_hydrogen=explicit_hydrogen)

    # TODO: fix the resolution problem
    bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(resolution*8))*8), [x_lim,y_lim,z_lim])]
    x,y,z = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),
                              bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')

    new_resolutions = [(x_lim.get('max')-x_lim.get('min'))/bin_numbers[0], 
                       (y_lim.get('max')-y_lim.get('min'))/bin_numbers[1], 
                       (z_lim.get('max')-z_lim.get('min'))/bin_numbers[2]]

    # x_,y_,z_ = np.meshgrid(np.linspace(-4,4,54), np.linspace(-3,-3, 40), np.linspace(-2,-3, 34), indexing='ij')

    pos = np.column_stack([x.flat,y.flat,z.flat])
    final_data = []
    final_smiles = []
    cov = np.diag([std for i in range(3)])
    all_coords_all_molecules = []
    all_atom_lists = []
    all_bonds_dictionaries = []
    if discretized_center:
        all_coords = discretize_position(all_coords, pos)


    for ind, (coords,smi,atom_symbol,bond) in enumerate(zip(all_coords,all_smiles,all_atom_symbols,all_bonds)):

        # mol = read_smiles(smi, reinterpret_aromatic=explicit_aromatic, explicit_hydrogen=explicit_hydrogen)
        # elements = nx.get_node_attributes(mol, name = "element")
        num_nodes = len(atom_symbol)
        Node_feature_one_hot = np.zeros((num_nodes,len(Unique_elements)))

        elements = {ind:el for ind,el in enumerate(atom_symbol)}
        all_atom_lists.append(atom_symbol)
        all_bonds_dictionaries.append(bond)


        for node in range(num_nodes):
          for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_one_hot[node][unelm] = 1


        X,Y,Z = coords[:,0],coords[:,1],coords[:,2]
        density_map = np.zeros((len(Unique_elements),x.shape[0],x.shape[1],x.shape[2]))
        for idx_node in range(num_nodes):
            rv = multivariate_normal([X[idx_node],Y[idx_node],Z[idx_node]], cov)
            temp_dens = rv.pdf(pos).reshape(x.shape[0],x.shape[1],x.shape[2])
            temp_dens[temp_dens<data_cutoff] = 0.0
            temp_dens = temp_dens/np.max(temp_dens) if normalize01 else temp_dens
            atom_one_hot = Node_feature_one_hot[idx_node]
            density_map[np.argmax(atom_one_hot)] = density_map[[np.argmax(atom_one_hot)]] + temp_dens
        
        for chan in range(len(Unique_elements)):
            if np.max(density_map[chan]) !=0 and not normalize01:
                # 0-1 scale
                # density_map[chan] = density_map[chan] / (density_map[chan] * np.prod(new_resolutions))
                # density_map[chan] = density_map[chan]/np.max(density_map[chan])
                density_map[chan] = density_map[chan]/np.sum(Node_feature_one_hot[:,chan])


                # -1 - 1 scale (as in the paper)/ this is done later in the diffusion model (no need to use this)
                if rescale:
                    density_map[chan] = (density_map[chan] -0.5)*2
        density_map = concatenate_bond_channels(density_map, bond, coords,pos,data_cutoff,
                                                cov,ignore_aromatic,normalize01)

        final_data.append(density_map)
        final_smiles.append(smi.replace("\n", ""))
    # get_min_atom_dists(all_coords_all_molecules)
    if return_true_mens:
        return final_smiles, final_data, Unique_elements,x,y,z,all_coords,all_atom_lists,all_bonds_dictionaries,new_resolutions
    return final_smiles, final_data, Unique_elements,x,y,z

    


def get_mol_reps(Smiles,data_type,std=0.05,use_pca=False,rescale=False,
                 resolution=0.1, limits=None, return_true_mens=False, data_cutoff=0.1,
                 discretized_center=False, explicit_aromatic=False):
    if not explicit_aromatic: Smiles = get_smiles_kekulize(Smiles)
    Unique_elements = UNQ_elements.get(data_type)
    x_lim,y_lim,z_lim,all_coords,all_smiles,all_atom_symbols, all_bonds = get_grid_limits(Smiles,use_pca=use_pca, std=std, limits=limits)
    # TODO: fixing it this way means you have different resolutions for different axis.
    # get closest multiples of 8 for each limit (i.e. extract bin distance d s.t. it has at least resolution specified,
    # while d % 8 ==0

    # x_,y_,z_ = np.mgrid[x_lim.get("min"):x_lim.get("max"):resolution,
    #                  y_lim.get("min"):y_lim.get("max"):resolution,
    #                  z_lim.get("min"):z_lim.get("max"):resolution]


    bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(resolution*8))*8), [x_lim,y_lim,z_lim])]
    x,y,z = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),
                              bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')

    new_resolutions = [(x_lim.get('max')-x_lim.get('min'))/bin_numbers[0], 
                       (y_lim.get('max')-y_lim.get('min'))/bin_numbers[1], 
                       (z_lim.get('max')-z_lim.get('min'))/bin_numbers[2]]

    # x_,y_,z_ = np.meshgrid(np.linspace(-4,4,54), np.linspace(-3,-3, 40), np.linspace(-2,-3, 34), indexing='ij')

    pos = np.column_stack([x.flat,y.flat,z.flat])
    final_data = []
    final_smiles = []
    cov = np.diag([std for i in range(3)])
    all_coords_all_molecules = []
    all_atom_lists = []
    all_bonds_dictionaries = []
    if discretized_center:
        all_coords = discretize_position(all_coords, pos)


    for coords,smi,atom_symbol in zip(all_coords,all_smiles,all_atom_symbols):
        mol = read_smiles(smi)
        elements = nx.get_node_attributes(mol, name = "element")
        edge_attr = nx.get_edge_attributes(mol, name = "order",)
        
        num_nodes = len(elements)
        Node_feature_one_hot = np.zeros((num_nodes,len(Unique_elements)))

        all_atom_lists.append([elements[node] for node in range(num_nodes)])
        all_bonds_dictionaries.append(edge_attr)


        for node in range(num_nodes):
          for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_one_hot[node][unelm] = 1


        X,Y,Z = coords[:,0],coords[:,1],coords[:,2]
        density_map = np.zeros((len(Unique_elements),x.shape[0],x.shape[1],x.shape[2]))
        for idx_node in range(num_nodes):
            rv = multivariate_normal([X[idx_node],Y[idx_node],Z[idx_node]], cov)
            temp_dens = rv.pdf(pos).reshape(x.shape[0],x.shape[1],x.shape[2])
            temp_dens[temp_dens<data_cutoff] = 0.0
            atom_one_hot = Node_feature_one_hot[idx_node]
            density_map[np.argmax(atom_one_hot)] = density_map[[np.argmax(atom_one_hot)]] + temp_dens
        
        for chan in range(len(Unique_elements)):
            if np.max(density_map[chan]) !=0:
                # 0-1 scale
                # density_map[chan] = density_map[chan] / (density_map[chan] * np.prod(new_resolutions))
                # density_map[chan] = density_map[chan]/np.max(density_map[chan])
                density_map[chan] = density_map[chan]/np.sum(Node_feature_one_hot[:,chan])


                # -1 - 1 scale (as in the paper)/ this is done later in the diffusion model (no need to use this)
                if rescale:
                    density_map[chan] = (density_map[chan] -0.5)*2

        final_data.append(density_map)
        final_smiles.append(smi.replace("\n", ""))
    # get_min_atom_dists(all_coords_all_molecules)
    if return_true_mens:
        return final_smiles, final_data, Unique_elements,x,y,z,all_coords,all_atom_lists,all_bonds_dictionaries,new_resolutions
    return final_smiles, final_data, Unique_elements,x,y,z

    


        
def fit_pred_field(pred,cutoff,x,y,z,num_channels,field_atoms):
  pred_field = pred.cpu().numpy()
  Atom_position = []
  for channel in range(num_channels):
    BIC= []
    means = []

    pred_dens = pred_field[channel].flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    x_flat = x_flat[pred_dens >= cutoff]
    y_flat = y_flat[pred_dens >= cutoff]
    z_flat = z_flat[pred_dens >= cutoff]

    if len(x_flat) == 0: continue
    pos_arr = np.concatenate([x_flat.reshape(-1,1),y_flat.reshape(-1,1),z_flat.reshape(-1,1)],axis=1)

    for cl in range(1,10):
       gm = GaussianMixture(n_components=cl, random_state=0).fit(pos_arr)
       means.append(gm.means_)
       bic = gm.bic(pos_arr)
       BIC.append(bic)

    #print(BIC)
    print(np.argmin(BIC)+1," Atoms found in the ",field_atoms[channel]," field")
    Atom_position.append(means[np.argmin(BIC)])


  return Atom_position







    
    

class AtomPositionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms

    def __len__(self):
        return len(self.coords)

    def transform(self, sample):
        mol = read_smiles(sample['smile'])
        coords = sample['coord']

        x,y,z = self.grids
        pos = np.column_stack([x.flat, y.flat, z.flat])
        elements = nx.get_node_attributes(mol, name="element")
        num_nodes = len(elements)
        Unique_elements = UNQ_elements.get("QM9")
        Node_feature_one_hot = np.zeros((num_nodes, len(Unique_elements)))

        for node in range(num_nodes):
            for unelm in range(len(Unique_elements)):
                if elements[node] == Unique_elements[unelm]:
                    Node_feature_one_hot[node][unelm] = 1

        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        density_map = np.zeros((len(Unique_elements), x.shape[0], x.shape[1], x.shape[2]))
        for idx_node in range(num_nodes):
            rv = multivariate_normal([X[idx_node], Y[idx_node], Z[idx_node]], self.std_atoms)
            # coords_molecule.append(np.array([X[idx_node],Y[idx_node],Z[idx_node]]))
            temp_dens = rv.pdf(pos).reshape(x.shape[0], x.shape[1], x.shape[2])
            temp_dens[temp_dens < 0.1] = 0.0
            atom_one_hot = Node_feature_one_hot[idx_node]
            density_map[np.argmax(atom_one_hot)] = density_map[[np.argmax(atom_one_hot)]] + temp_dens
        # all_coords_all_molecules.append(coords_molecule)
        for chan in range(len(Unique_elements)):
            if np.max(density_map[chan]) != 0:
                # 0-1 scale
                density_map[chan] = density_map[chan] / np.max(density_map[chan])
                # -1 - 1 scale (as in the paper)
        return torch.tensor(density_map).to(torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        coord, smile = self.coords[idx], self.smiles[idx]
        sample = {'coord': coord, 'smile': smile}

        if self.transform:
            sample = self.transform(sample)

        return sample



class AtomPositionNoSepBondsDatasetCompact(Dataset):
    """Returns atom and bond positiosn rather than a density map"""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms,all_atom_symbols,all_bonds,ignore_aromatic=True, 
                 explicit_hydrogens=False,debug_ds=False, agnostic_atm_types=False, subsample_points=-1,mixed_prec=False,
                 augment_rotations=False, atom_no_classes=None, explicit_aromatic=False, use_subset=-1, center_atm_to_grids=False,
                 unique_atms = None, arom_cycle_channel=False):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms
        self.all_atom_symbols=all_atom_symbols
        self.all_bonds=all_bonds
        self.ignore_aromatic=ignore_aromatic
        self.explicit_hydrogens=explicit_hydrogens
        self.debug_ds=debug_ds
        self.agnostic_atm_types=agnostic_atm_types
        self.subsample_points = subsample_points
        self.mixed_prec=mixed_prec
        self.augment_rotations=augment_rotations
        self.atom_no_classes=atom_no_classes
        self.explicit_aromatic = explicit_aromatic
        if use_subset != -1:
            self.coords = self.coords[:use_subset]
            self.smiles = self.smiles[:use_subset]
            self.all_atom_symbols = self.all_atom_symbols[:use_subset]
            self.all_bonds = self.all_bonds[:use_subset]
        self.center_atm_to_grids=center_atm_to_grids
        self.unique_atms = unique_atms
        self.arom_cycle_channel=arom_cycle_channel
        if not arom_cycle_channel: 
            for atm_s in self.all_atom_symbols:
                if 'X' in atm_s or 'Y' in atm_s: print("!!!ERROR!!! specified arom_cycle_channel channel False but found X or Y in atom symbols (which are related to aromatic circles)"); exit(1)

    def __len__(self):
        return len(self.coords)

    def append_bond_coords_get_chn_inds(self,coords, bond_type, atm_types):
        if self.unique_atms is not None:
            unq_elems = self.unique_atms
        else:
            unq_elems = np.array(UNQ_elements.get("QM9" if not self.explicit_hydrogens else "QM9-hydr"))
        elem2id = {el:i for i,el in enumerate(unq_elems)}
        arom_circle_start = len(elem2id) + 3 + self.explicit_aromatic
        arom_circle2id = {circle_type:arom_circle_start+ind for ind, circle_type in enumerate(['X', 'Y'])}
        ids = []
        all_coords = []
        N_list = []
        old2new_atm_index = {}
        current_atm_index = 0
        # have the coordinate and indices in the same order (C, O, N, F, H)
        for atm_el in unq_elems:
            current_atms = [a_t==atm_el for a_t in atm_types]
            no_atms = sum(current_atms)
            all_coords.extend(coords[current_atms])
            ids.extend([elem2id[atm_el]] * no_atms)
            if no_atms != 0: N_list.append(sum(current_atms)) # if sum(crrent_bonds) is zero, the corresponding gaussian weight pi will also be 0
            for ind, ca in enumerate(current_atms):
                if ca: old2new_atm_index[ind] = current_atm_index; current_atm_index += 1




        # atms2bndindex = {k:v-1 if v != 1.5 else 3 for k,v in bond_type.items()}
        if type(bond_type) == list: atms2bndindex = np.array([[b[0],b[1],b[2]-1] for b in bond_type] )
        else: atms2bndindex = np.array([(k[0],k[1],v-1) if v != 1.5 else (k[0], k[1], 3) for k,v in bond_type.items()])
        newatm2bndindex = {}

        # in QM9 non-explicit H, some molecules are actually 1 atm
        if len(atms2bndindex):
            for bnd in range(4):
                current_bonds = [b_i==bnd for b_i in atms2bndindex[:,2]]
                for b_i in atms2bndindex[current_bonds]:
                    newatm2bndindex[(old2new_atm_index[b_i[0]], old2new_atm_index[b_i[1]])] = b_i[2]
        all_coords = np.array(all_coords)
        return all_coords, ids, N_list, newatm2bndindex



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coord, smile,bond_type, atm_types = self.coords[idx], self.smiles[idx], self.all_bonds[idx], self.all_atom_symbols[idx]

        coords, ids, N_list, bond_type = self.append_bond_coords_get_chn_inds(coord, bond_type, atm_types)
        coords = torch.tensor(coords).to(torch.float32)
        if self.atom_no_classes is not None:
            no_cls = self.atom_no_classes[idx]
        if self.debug_ds:
            return  smile, bond_type, atm_types, coords, np.array(ids), N_list, no_cls
        if not self.atom_no_classes: return coords, np.array(ids), N_list, bond_type
        return coords, np.array(ids), N_list, no_cls, bond_type

class AtomPositionNoSepBondsDatasetCompactWProps(Dataset):
    """Returns atom and bond positiosn rather than a density map"""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms,all_atom_symbols,all_bonds,ignore_aromatic=True, 
                 explicit_hydrogens=False,debug_ds=False, agnostic_atm_types=False, subsample_points=-1,mixed_prec=False,
                 augment_rotations=False, atom_no_classes=None, explicit_aromatic=False, use_subset=-1, center_atm_to_grids=False,
                 unique_atms = None, arom_cycle_channel=False, properties=None):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms
        self.all_atom_symbols=all_atom_symbols
        self.all_bonds=all_bonds
        self.ignore_aromatic=ignore_aromatic
        self.explicit_hydrogens=explicit_hydrogens
        self.debug_ds=debug_ds
        self.agnostic_atm_types=agnostic_atm_types
        self.subsample_points = subsample_points
        self.mixed_prec=mixed_prec
        self.augment_rotations=augment_rotations
        self.atom_no_classes=atom_no_classes
        self.explicit_aromatic = explicit_aromatic
        if use_subset != -1:
            self.coords = self.coords[:use_subset]
            self.smiles = self.smiles[:use_subset]
            self.all_atom_symbols = self.all_atom_symbols[:use_subset]
            self.all_bonds = self.all_bonds[:use_subset]
        self.center_atm_to_grids=center_atm_to_grids
        self.unique_atms = unique_atms
        self.arom_cycle_channel=arom_cycle_channel
        if not arom_cycle_channel: 
            for atm_s in self.all_atom_symbols:
                if 'X' in atm_s or 'Y' in atm_s: print("!!!ERROR!!! specified arom_cycle_channel channel False but found X or Y in atom symbols (which are related to aromatic circles)"); exit(1)

    def __len__(self):
        return len(self.coords)

    def append_bond_coords_get_chn_inds(self,coords, bond_type, atm_types):
        if self.unique_atms is not None:
            unq_elems = self.unique_atms
        else:
            unq_elems = np.array(UNQ_elements.get("QM9" if not self.explicit_hydrogens else "QM9-hydr"))
        elem2id = {el:i for i,el in enumerate(unq_elems)}
        arom_circle_start = len(elem2id) + 3 + self.explicit_aromatic
        arom_circle2id = {circle_type:arom_circle_start+ind for ind, circle_type in enumerate(['X', 'Y'])}
        ids = []
        all_coords = []
        N_list = []
        old2new_atm_index = {}
        current_atm_index = 0
        # have the coordinate and indices in the same order (C, O, N, F, H)
        for atm_el in unq_elems:
            current_atms = [a_t==atm_el for a_t in atm_types]
            no_atms = sum(current_atms)
            all_coords.extend(coords[current_atms])
            ids.extend([elem2id[atm_el]] * no_atms)
            if no_atms != 0: N_list.append(sum(current_atms)) # if sum(crrent_bonds) is zero, the corresponding gaussian weight pi will also be 0
            for ind, ca in enumerate(current_atms):
                if ca: old2new_atm_index[ind] = current_atm_index; current_atm_index += 1




        # atms2bndindex = {k:v-1 if v != 1.5 else 3 for k,v in bond_type.items()}
        if type(bond_type) == list: atms2bndindex = np.array([[b[0],b[1],b[2]-1] for b in bond_type] )
        else: atms2bndindex = np.array([(k[0],k[1],v-1) if v != 1.5 else (k[0], k[1], 3) for k,v in bond_type.items()])
        newatm2bndindex = {}

        # in QM9 non-explicit H, some molecules are actually 1 atm
        if len(atms2bndindex):
            for bnd in range(4):
                current_bonds = [b_i==bnd for b_i in atms2bndindex[:,2]]
                for b_i in atms2bndindex[current_bonds]:
                    newatm2bndindex[(old2new_atm_index[b_i[0]], old2new_atm_index[b_i[1]])] = b_i[2]
        all_coords = np.array(all_coords)
        return all_coords, ids, N_list, newatm2bndindex



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coord, smile,bond_type, atm_types = self.coords[idx], self.smiles[idx], self.all_bonds[idx], self.all_atom_symbols[idx]

        coords, ids, N_list, bond_type = self.append_bond_coords_get_chn_inds(coord, bond_type, atm_types)
        coords = torch.tensor(coords).to(torch.float32)
        if self.atom_no_classes is not None:
            no_cls = self.atom_no_classes[idx]
        if self.debug_ds:
            return  smile, bond_type, atm_types, coords, np.array(ids), N_list, no_cls
        if not self.atom_no_classes: return coords, np.array(ids), N_list, bond_type
        return coords, np.array(ids), N_list, no_cls, bond_type


class AtomPositionSepBondsDatasetCompact(Dataset):
    """Returns atom and bond positiosn rather than a density map"""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms,all_atom_symbols,all_bonds,ignore_aromatic=True, 
                 explicit_hydrogens=False,debug_ds=False, agnostic_atm_types=False, subsample_points=-1,mixed_prec=False,
                 augment_rotations=False, atom_no_classes=None, explicit_aromatic=False, use_subset=-1, center_atm_to_grids=False,
                 unique_atms = None, arom_cycle_channel=False, cond_variables=None):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms
        self.all_atom_symbols=all_atom_symbols
        self.all_bonds=all_bonds
        self.ignore_aromatic=ignore_aromatic
        self.explicit_hydrogens=explicit_hydrogens
        self.debug_ds=debug_ds
        self.agnostic_atm_types=agnostic_atm_types
        self.subsample_points = subsample_points
        self.mixed_prec=mixed_prec
        self.augment_rotations=augment_rotations
        self.atom_no_classes=atom_no_classes
        self.explicit_aromatic = explicit_aromatic
        self.cond_variables = cond_variables if exists(cond_variables) and len(cond_variables) else None

        if use_subset != -1:
            self.coords = self.coords[:use_subset]
            self.smiles = self.smiles[:use_subset]
            self.all_atom_symbols = self.all_atom_symbols[:use_subset]
            self.all_bonds = self.all_bonds[:use_subset]
        self.center_atm_to_grids=center_atm_to_grids
        self.unique_atms = unique_atms
        self.arom_cycle_channel=arom_cycle_channel
        if not arom_cycle_channel: 
            for atm_s in self.all_atom_symbols:
                if 'X' in atm_s or 'Y' in atm_s: print("!!!ERROR!!! specified arom_cycle_channel channel False but found X or Y in atom symbols (which are related to aromatic circles)"); exit(1)

    def __len__(self):
        return len(self.coords)

    def append_bond_coords_get_chn_inds(self,coords, bond_type, atm_types):
        if self.unique_atms is not None:
            unq_elems = self.unique_atms
        else:
            unq_elems = np.array(UNQ_elements.get("QM9" if not self.explicit_hydrogens else "QM9-hydr"))
        elem2id = {el:i for i,el in enumerate(unq_elems)}
        arom_circle_start = len(elem2id) + 3 + self.explicit_aromatic
        arom_circle2id = {circle_type:arom_circle_start+ind for ind, circle_type in enumerate(['X', 'Y'])}
        ids = []
        all_coords = []
        N_list = []
        old2new_atm_index = {}
        current_atm_index = 0
        # have the coordinate and indices in the same order (C, O, N, F, H)
        for atm_el in unq_elems:
            current_atms = [a_t==atm_el for a_t in atm_types]
            no_atms = sum(current_atms)
            all_coords.extend(coords[current_atms])
            ids.extend([elem2id[atm_el]] * no_atms)
            if no_atms != 0: N_list.append(sum(current_atms)) # if sum(crrent_bonds) is zero, the corresponding gaussian weight pi will also be 0
            for ind, ca in enumerate(current_atms):
                if ca: old2new_atm_index[ind] = current_atm_index; current_atm_index += 1




        # atms2bndindex = {k:v-1 if v != 1.5 else 3 for k,v in bond_type.items()}
        if type(bond_type) == list: atms2bndindex = np.array([[b[0],b[1],b[2]-1] for b in bond_type] )
        else: atms2bndindex = np.array([(k[0],k[1],v-1) if v != 1.5 else (k[0], k[1], 3) for k,v in bond_type.items()])
        newatm2bndindex = {}

        # in QM9 non-explicit H, some molecules are actually 1 atm
        if len(atms2bndindex):
            # add bonds in the same order (bnd 1, 2, 3, 4)
            for bnd in range(4):
                current_bonds = [b_i==bnd for b_i in atms2bndindex[:,2]]
                no_bnds = sum(current_bonds)

                all_coords.extend(  [(coords[b_i[0]] + coords[b_i[1]])/2  for b_i in atms2bndindex[current_bonds]] )
                ids.extend([bnd+len(unq_elems)] * no_bnds)
                if no_bnds != 0: N_list.append(sum(current_bonds))
                for b_i in atms2bndindex[current_bonds]:

                    newatm2bndindex[(old2new_atm_index[b_i[0]], old2new_atm_index[b_i[1]])] = b_i[2]

        if self.arom_cycle_channel:
            for arom_el in ['X', 'Y']:
                current_cycles = [a_t==arom_el for a_t in atm_types]
                no_cycles = sum(current_cycles)
                all_coords.extend(coords[current_cycles])

                ids.extend([arom_circle2id[arom_el]] * no_cycles)
                if no_cycles != 0: N_list.append(sum(current_cycles)) # if sum(crrent_bonds) is zero, the corresponding gaussian weight pi will also be 0
                # for ind, ca in enumerate(current_cycles):
                #     if ca: old2new_atm_index[ind] = current_atm_index; current_atm_index += 1
        # if "X" in atm_types: breakpoint()

        # if "X" in atm_types or "Y" in atm_types: breakpoint()
        
        # if self.arom_cycle_channel:
            # breakpoint()

        all_coords = np.array(all_coords)
        return all_coords, ids, N_list, newatm2bndindex



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coord, smile,bond_type, atm_types = self.coords[idx], self.smiles[idx], self.all_bonds[idx], self.all_atom_symbols[idx]
        cond_var = self.cond_variables[idx] if exists(self.cond_variables) else None
        coords, ids, N_list, bond_type = self.append_bond_coords_get_chn_inds(coord, bond_type, atm_types)
        coords = torch.tensor(coords).to(torch.float32)
        if self.atom_no_classes is not None:
            no_cls = self.atom_no_classes[idx]
        if self.debug_ds:
            return  smile, bond_type, atm_types, coords, np.array(ids), N_list, no_cls
        if not self.atom_no_classes: return coords, np.array(ids), N_list, bond_type
        if exists(cond_var): return coords, np.array(ids), N_list, no_cls, bond_type, cond_var
        return coords, np.array(ids), N_list, no_cls, bond_type


class AtomPositionSepBondsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms,all_atom_symbols,all_bonds,ignore_aromatic=True, 
                 explicit_hydrogens=False,debug_ds=False, agnostic_atm_types=False, subsample_points=-1,mixed_prec=False,
                 augment_rotations=False, atom_no_classes=None):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms
        self.all_atom_symbols=all_atom_symbols
        self.all_bonds=all_bonds
        self.ignore_aromatic=ignore_aromatic
        self.explicit_hydrogens=explicit_hydrogens
        self.debug_ds=debug_ds
        self.agnostic_atm_types=agnostic_atm_types
        self.subsample_points = subsample_points
        self.mixed_prec=mixed_prec
        self.augment_rotations=augment_rotations
        self.atom_no_classes=atom_no_classes

    def __len__(self):
        return len(self.coords)

    def transform(self, sample):
        # mol = Chem.MolFromSmiles(sample['smile'])
        # if self.explicit_hydrogens: mol = Chem.AddHs(mol)
        # elements = []
        # for i in range(mol.GetNumAtoms()): elements.append(mol.GetAtomWithIdx(i).GetSymbol())
        # num_nodes = len(elements)
        elements = sample['atom_type']
        num_nodes = len(elements)
        # breakpoint()

        # * check failure of read_smiles
        # mol_ = read_smiles(sample['smile'],explicit_hydrogen=self.explicit_hydrogens)
        # elements_ = nx.get_node_attributes(mol_, name="element", )
        # print(len(elements), len(elements_), len(sample['coord']))
        # if len(elements) != len(elements_): print("Error: read_smiles failed")

        coords = sample['coord']
        if self.augment_rotations:
            r = Rotation.random()
            coords = r.apply(coords)

        x,y,z = self.grids
        pos = np.column_stack([x.flat, y.flat, z.flat])

        if coords.shape[0] != len(elements): print("Error: number of atoms in the molecule and in the coordinates do not match")

        Unique_elements = UNQ_elements.get("QM9" if not self.explicit_hydrogens else "QM9-hydr")
        Node_feature_one_hot = np.zeros((num_nodes, len(Unique_elements)))
        for node in range(num_nodes):
            for unelm in range(len(Unique_elements)):
                if elements[node] == Unique_elements[unelm]:
                    Node_feature_one_hot[node][unelm] = 1
        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        density_map = np.zeros((len(Unique_elements), x.shape[0], x.shape[1], x.shape[2]))
        for idx_node in range(num_nodes):
            rv = multivariate_normal([X[idx_node], Y[idx_node], Z[idx_node]], self.std_atoms)
            # coords_molecule.append(np.array([X[idx_node],Y[idx_node],Z[idx_node]]))
            temp_dens = rv.pdf(pos).reshape(x.shape[0], x.shape[1], x.shape[2])
            temp_dens[temp_dens < 0.1] = 0.0
            atom_one_hot = Node_feature_one_hot[idx_node]
            temp_dens = temp_dens/np.max(temp_dens)
            density_map[np.argmax(atom_one_hot)] = density_map[[np.argmax(atom_one_hot)]] + temp_dens
        # all_coords_all_molecules.append(coords_molecule)
        pos = np.column_stack([x.flat,y.flat,z.flat])
        cov = np.diag([self.std_atoms for _ in range(3)])
        # TODO below, peaks may be slightly above 1 
        density_map = concatenate_bond_channels(density_map, sample['bond_type'],atom_coords=coords,pos=pos,data_cutoff=0.1,
                                                cov=cov,ignore_aromatic=self.ignore_aromatic,normalize01=True)
        if self.subsample_points != -1:
            atom_bnd_points = np.argwhere(density_map>0.05)
            number_of_zero_points = int(self.subsample_points * np.prod(density_map.shape))
            zero_points = np.argwhere(density_map <= 0.05)
            zero_points = zero_points[np.random.choice(zero_points.shape[0], number_of_zero_points, replace=False)]
            all_points = np.concatenate([atom_bnd_points, zero_points], axis=0)
        # add bond channels

        for chan in range(density_map.shape[0]):
            if np.max(density_map[chan]) != 0:
                # 0-1 scale
                density_map[chan] = np.clip(density_map[chan],0,1)
                # -1 - 1 scale (as in the paper)
        # bond_list = [[k[0],k[1],v] for k,v in sample['bond_type'].items()]
        # plot_one_mol(coords, bond_list, plot_bnd=4, threshold=0.7, atm_symb=elements, 
        #              field=torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32),
        #              x_grid=x,y_grid=y,z_grid=z)

        # # * create 3D plot of density_map
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x, y, z = density_map[0].nonzero()
        # sc = ax.scatter(x, y, z, c=density_map[0,x, y, z], cmap='viridis')
        # plt.colorbar(sc, orientation='horizontal', label='Density')
        # plt.show()

        if self.debug_ds: return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32), elements
        if self.subsample_points != -1: return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32), all_points
        return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        coord, smile,bond_type, atm_types = self.coords[idx], self.smiles[idx], self.all_bonds[idx], self.all_atom_symbols[idx]
        sample = {'coord': coord, 'smile': smile, 'bond_type':bond_type, 'atom_type':atm_types}

        if self.atom_no_classes is not None:
            atm_no_cls = self.atom_no_classes[idx]
            sample_ = self.transform(sample)
            return sample_, atm_no_cls
        if self.debug_ds:
            sample_, atoms = self.transform(sample)
            return {'field':sample_, 'smile':smile, 'bond_type':bond_type, 'coord':coord, 'atoms':atoms}
        elif self.agnostic_atm_types:
            sample_, atoms = self.transform(sample)
            return {'field':sample_, 'smile':smile, 'bond_type':bond_type, 'coord':coord, 'atoms':atoms}
        # elif self.agnostic_atm_types:
        #     sample_, atom_list = self.transform(sample)
        #     elements = []
        #     mol = Chem.MolFromSmiles(sample['smile'])
        #     for i in range(mol.GetNumAtoms()): elements.append(mol.GetAtomWithIdx(i).GetSymbol())

        #     return sample_, smile, bond_type, coord, atom_list, elements
        if self.transform:
            sample = self.transform(sample)

        return sample
    

if __name__=='__main__':
    cwd = os.getcwd()
    data_path =  str(cwd) + '/data/QM9.txt'
    with open(data_path) as f:
         Smiles = f.readlines()
    
    # Unique_elements  = get_unique(Smiles[0:1000])
    # for smi in Smiles[0:10]:
    #      mol, coords,conf = smi2conf(smi)
    #      coords_aligned = align(coords)
    #      new_conf,new_coords = update_mol_positions(mol,conf,coords_aligned)
    #      get_mol_reps(smi,Unique_elements,np.array(new_coords).reshape(-1,3), std=0.05,use_pca=False)

        

def custom_collate_agnostic_atm_types(batch):
    """
    Custom collate function that can handle 5 types of data:
    - Equal size torch tensors
    - Strings
    - Dictionaries
    - List of variable lengths
    """
    # Separate the data into different lists based on their type


    fields, smiles, bond_types, coords, atoms = [],[],[],[],[]

    for b in batch:
        fields.append(b['field'])
        smiles.append(b['smile'])
        bond_types.append(b['bond_type'])
        coords.append(b['coord'])
        atoms.append(b['atoms'])
    return torch.stack(fields), smiles, bond_types, coords, atoms
from torch.nn.utils.rnn import pack_sequence


def collate_fn_compact_expl_ah_clsfreeguid(batch):
    total_atms_and_bnds = 0
    all_coords, all_inds, all_N_lists, all_cls, all_bnds = [], [], [], [], []
    for ind, (crds, inds, n_list, n_cls, bnd) in enumerate(batch):
        no_atms = sum(n_list)-len(bnd)
        all_coords.append(crds)
        all_inds.append(inds + ind * 9)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls,np.array(all_bnds)

def collate_fn_compact_expl_h_clsfreeguid(batch):
    all_coords, all_inds, all_N_lists, all_cls, all_bnds = [], [], [], [], []
    total_atms_and_bnds = 0
    for ind, (crds, inds, n_list, n_cls, bnd) in enumerate(batch):
        no_atms = sum(n_list)-len(bnd)
        all_coords.append(crds)
        all_inds.append(inds + ind * 8)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, np.array(all_bnds)


def collate_fn_compact_expl_h_clsfreeguid_debugds_geom(batch):
    all_coords, all_inds, all_N_lists, all_cls, all_smiles, all_bnd_types, all_atm_types = [], [], [], [], [],[],[]
    for ind, (sml, bnd_t, atm_t, crds, inds, n_list, n_cls) in enumerate(batch):
        all_coords.append(crds)
        all_inds.append(inds + ind * 11)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_smiles.append(sml)
        all_bnd_types.append(bnd_t)
        all_atm_types.append(atm_t)


    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, all_smiles,all_bnd_types,all_atm_types


def collate_fn_compact_expl_h_clsfreeguid_geom(batch):
    all_coords, all_inds, all_N_lists, all_cls, all_bnds = [], [], [], [], []
    total_atms_and_bnds = 0
    for ind, (crds, inds, n_list, n_cls, bnd) in enumerate(batch):
        no_atms = sum(n_list)-len(bnd)
        all_coords.append(crds)
        all_inds.append(inds + ind * 11)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, np.array(all_bnds)


def collate_fn_compact_expl_ah(batch):
    all_coords, all_inds, all_N_lists, all_bnds = [], [], [], []
    total_atms_and_bnds = 0
    for ind, (crds, inds, n_list, bnd) in enumerate(batch):
        no_atms = sum(n_list)-len(bnd)
        all_coords.append(crds)
        all_inds.append(inds + ind * 9)
        all_N_lists.append(n_list)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    return coords,inds,all_N_lists,np.array(all_bnds)



def collate_fn_general_noatmNo(batch,no_channels=8,atms_last_ind=4):
    all_coords, all_inds, all_N_lists, all_bnds = [], [], [], []
    total_atms_and_bnds = 0
    for ind, (crds, inds, n_list, bnd) in enumerate(batch):
        no_atms = sum([i % no_channels <= atms_last_ind for i in inds])
        # no_atms = sum(n_list)-len(bnd)

        all_coords.append(crds)
        all_inds.append(inds + ind * no_channels)
        all_N_lists.append(n_list)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    return coords,inds,all_N_lists, np.array(all_bnds)


def collate_fn_general_cond_var(batch,no_channels=8,atms_last_ind=4):
    all_coords, all_inds, all_N_lists, all_cls, all_bnds, cond_vars = [], [], [], [], [], []
    total_atms_and_bnds = 0
    consider_bonds = no_channels > atms_last_ind + 1
    for ind, (crds, inds, n_list, n_cls, bnd, c_v) in enumerate(batch):
        no_atms = sum([i % no_channels <= atms_last_ind for i in inds])
        all_coords.append(crds)
        all_inds.append(inds + ind * no_channels)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        cond_vars.append(c_v)
        if consider_bonds:
            all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        else:
            # * simply return bond type for visualizations (which otherwise can be inferred through N_list and inds)
            all_bnds.extend([[b_t, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,b_t) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, np.array(all_bnds), np.stack(cond_vars)



def collate_fn_general_debug(batch):
    # ! not so general for now actually
    all_coords, all_inds, all_N_lists, all_cls, all_smiles, all_bnd_types, all_atm_types = [], [], [], [], [], [], []

    for ind, (sml, bnd_t, atm_t, crds, inds, n_list, n_cls) in enumerate(batch):
        all_coords.append(crds)

        all_inds.append (inds + ind * 5)

        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_smiles.append(sml)
        all_bnd_types.append(bnd_t)
        all_atm_types.append(atm_t)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, all_smiles,all_bnd_types,all_atm_types


def collate_fn_general(batch,no_channels=8,atms_last_ind=4):
    all_coords, all_inds, all_N_lists, all_cls, all_bnds = [], [], [], [], []
    total_atms_and_bnds = 0
    consider_bonds = no_channels > atms_last_ind + 1
    for ind, (crds, inds, n_list, n_cls, bnd) in enumerate(batch):
        no_atms = sum([i % no_channels <= atms_last_ind for i in inds])
        all_coords.append(crds)
        all_inds.append(inds + ind * no_channels)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        if consider_bonds:
            all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        else:
            # * simply return bond type for visualizations (which otherwise can be inferred through N_list and inds)
            all_bnds.extend([[b_t, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,b_t) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, np.array(all_bnds)


def collate_fn_compact_expl_h(batch):
    all_coords, all_inds, all_N_lists, all_bnds = [], [], [], []
    total_atms_and_bnds = 0
    for ind, (crds, inds, n_list, bnd) in enumerate(batch):
        no_atms = sum(n_list)-len(bnd)
        all_coords.append(crds)
        all_inds.append(inds + ind * 8)
        all_N_lists.append(n_list)
        all_bnds.extend([[i + no_atms +total_atms_and_bnds, b[0]+total_atms_and_bnds, b[1]+total_atms_and_bnds] for i, (b,_) in enumerate(bnd.items())])
        total_atms_and_bnds += len(inds)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    return coords,inds,all_N_lists,np.array(all_bnds)


def collate_fn_compact_expl_ah_clsfreeguid_debugds(batch):
    all_coords, all_inds, all_N_lists, all_cls, all_smiles, all_bnd_types, all_atm_types = [], [], [], [], [], [], []

    for ind, (sml, bnd_t, atm_t, crds, inds, n_list, n_cls) in enumerate(batch):
        all_coords.append(crds)
        all_inds.append(inds + ind * 9)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_smiles.append(sml)
        all_bnd_types.append(bnd_t)
        all_atm_types.append(atm_t)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, all_smiles,all_bnd_types,all_atm_types

def collate_fn_compact_expl_h_clsfreeguid_debugds(batch):
    all_coords, all_inds, all_N_lists, all_cls, all_smiles, all_bnd_types, all_atm_types = [], [], [], [], [],[],[]
    for ind, (sml, bnd_t, atm_t, crds, inds, n_list, n_cls) in enumerate(batch):
        all_coords.append(crds)
        all_inds.append(inds + ind * 8)
        all_N_lists.append(n_list)
        all_cls.append(n_cls)
        all_smiles.append(sml)
        all_bnd_types.append(bnd_t)
        all_atm_types.append(atm_t)


    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    all_cls = torch.tensor(all_cls)
    return coords,inds,all_N_lists,all_cls, all_smiles,all_bnd_types,all_atm_types

def collate_fn_compact_expl_ah_debugds(batch):
    all_coords, all_inds, all_N_lists = [], [], []
    for ind, (crds, inds, n_list) in enumerate(batch):
        all_coords.append(crds)
        all_inds.append(inds + ind * 9)
        all_N_lists.append(n_list)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    return coords,inds,all_N_lists

def collate_fn_compact_expl_h_debugds(batch):
    all_coords, all_inds, all_N_lists = [], [], []
    for ind, (crds, inds, n_list) in enumerate(batch):
        all_coords.append(crds)
        all_inds.append(inds + ind * 8)
        all_N_lists.append(n_list)

    coords = torch.vstack(all_coords).to(torch.float32)
    inds = np.concatenate(all_inds)
    return coords,inds,all_N_lists






def collate_subsample_points(batch):
    batch_index = []
    all_indexes= []
    all_fields = []
    for ind,(field, indexes) in enumerate(batch):
        batch_index.extend([ind]*indexes.shape[0])
        all_indexes.append(indexes)
        all_fields.append(field)
    all_indexes = np.concatenate(all_indexes, axis=0)
    batch_index = np.array(batch_index)
    all_indexes_w_batch = np.concatenate([batch_index[:, None],all_indexes], axis=1)
    all_fields = torch.stack(all_fields)
    return all_fields, all_indexes_w_batch




class AtomPositionSepBondsAgnosticAtmTypeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, coords, smiles, x_grid,y_grid,z_grid,std_atoms,all_atom_symbols,all_bonds,ignore_aromatic=True, 
                 explicit_hydrogens=False,debug_ds=False, agnostic_atm_types=False, subsample_points=-1,mixed_prec=False):
        self.coords=coords
        self.smiles=smiles
        self.grids = [x_grid,y_grid,z_grid]
        self.std_atoms =std_atoms
        self.all_atom_symbols=all_atom_symbols
        self.all_bonds=all_bonds
        self.ignore_aromatic=ignore_aromatic
        self.explicit_hydrogens=explicit_hydrogens
        self.debug_ds=debug_ds
        self.agnostic_atm_types=agnostic_atm_types
        self.subsample_points = subsample_points
        self.mixed_prec=mixed_prec

    def __len__(self):
        return len(self.coords)

    def transform(self, sample):
        mol = Chem.MolFromSmiles(sample['smile'])
        if self.explicit_hydrogens: mol = Chem.AddHs(mol)
        elements = []
        for i in range(mol.GetNumAtoms()): elements.append(mol.GetAtomWithIdx(i).GetSymbol())
        num_nodes = len(elements)

        # * check failure of read_smiles
        # mol_ = read_smiles(sample['smile'],explicit_hydrogen=self.explicit_hydrogens)
        # elements_ = nx.get_node_attributes(mol_, name="element", )
        # print(len(elements), len(elements_), len(sample['coord']))
        # if len(elements) != len(elements_): print("Error: read_smiles failed")

        coords = sample['coord']
        x,y,z = self.grids
        pos = np.column_stack([x.flat, y.flat, z.flat])

        if coords.shape[0] != len(elements): print("Error: number of atoms in the molecule and in the coordinates do not match")

        Unique_elements = UNQ_elements.get("QM9" if not self.explicit_hydrogens else "QM9-hydr")
        Node_feature_one_hot = np.zeros((num_nodes, len(Unique_elements)))
        for node in range(num_nodes):
            for unelm in range(len(Unique_elements)):
                if elements[node] == Unique_elements[unelm]:
                    Node_feature_one_hot[node][unelm] = 1
        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        density_map = np.zeros(1, x.shape[0], x.shape[1], x.shape[2])
        for idx_node in range(num_nodes):
            rv = multivariate_normal([X[idx_node], Y[idx_node], Z[idx_node]], self.std_atoms)
            temp_dens = rv.pdf(pos).reshape(x.shape[0], x.shape[1], x.shape[2])
            temp_dens[temp_dens < 0.1] = 0.0
            temp_dens = temp_dens/np.max(temp_dens)
            density_map[0] = density_map[0] + temp_dens # all atoms are added in one channel
        # all_coords_all_molecules.append(coords_molecule)
        pos = np.column_stack([x.flat,y.flat,z.flat])
        cov = np.diag([self.std_atoms for _ in range(3)])
        # TODO below, peaks may be slightly above 1 
        density_map = concatenate_bond_channels(density_map, sample['bond_type'],atom_coords=coords,pos=pos,data_cutoff=0.1,
                                                cov=cov,ignore_aromatic=self.ignore_aromatic,normalize01=True)
        if self.subsample_points != -1:
            atom_bnd_points = np.argwhere(density_map>0.05)
            number_of_zero_points = int(self.subsample_points * np.prod(density_map.shape))
            zero_points = np.argwhere(density_map <= 0.05)
            zero_points = zero_points[np.random.choice(zero_points.shape[0], number_of_zero_points, replace=False)]
            all_points = np.concatenate([atom_bnd_points, zero_points], axis=0)
        # add bond channels

        for chan in range(density_map.shape[0]):
            if np.max(density_map[chan]) != 0:
                # 0-1 scale
                density_map[chan] = np.clip(density_map[chan],0,1)
                # -1 - 1 scale (as in the paper)
        if self.debug_ds: return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32), elements
        if self.subsample_points != -1: return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32), all_points
        return torch.tensor(density_map).to(torch.float16 if self.mixed_prec else torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        coord, smile,bond_type = self.coords[idx], self.smiles[idx], self.all_bonds[idx]
        sample = {'coord': coord, 'smile': smile, 'bond_type':bond_type}

        if self.debug_ds:
            sample_, atoms = self.transform(sample)
            return {'field':sample_, 'smile':smile, 'bond_type':bond_type, 'coord':coord, 'atoms':atoms}
        elif self.agnostic_atm_types:
            sample_, atoms = self.transform(sample)
            return {'field':sample_, 'smile':smile, 'bond_type':bond_type, 'coord':coord, 'atoms':atoms}
        # elif self.agnostic_atm_types:
        #     sample_, atom_list = self.transform(sample)
        #     elements = []
        #     mol = Chem.MolFromSmiles(sample['smile'])
        #     for i in range(mol.GetNumAtoms()): elements.append(mol.GetAtomWithIdx(i).GetSymbol())

        #     return sample_, smile, bond_type, coord, atom_list, elements
        if self.transform:
            sample = self.transform(sample)

        return sample
    


def get_atom_no_dist_geom(data_path=None, explicit_h=False):
    data = pickle.load(open(os.path.join(data_path,file), "rb"))

    if geom:
        atoms = []
        for i in range(len(data)): atoms.append(data[str(i)][1])
    else:
        atoms = []
        for set in ['train', 'test', 'valid']: atoms.extend(data[5][set])
    count_w_h, count_wo_h = {}, {}
    for ind, atm in enumerate(atoms):
        no_w_h,no_wo_h = len(atm), sum([a != 'H' for a in atm])
        aromatic_chns_no = sum([a == 'X' or a == 'Y' for a in atm]) # these are "artificial atm chns" that are possibly added to the molecule (see args.aromatic_chns_no)
        no_w_h,no_wo_h = no_w_h-aromatic_chns_no,no_wo_h-aromatic_chns_no
        count_w_h[no_w_h] = count_w_h[no_w_h] + 1 if no_w_h in count_w_h else 1
        count_wo_h[no_wo_h] = count_wo_h[no_wo_h] + 1 if no_wo_h in count_wo_h else 1

    if plot:
        count_plot = count_w_h if explicit_h else count_wo_h
        plt.bar(count_plot.keys(), count_plot.values())
        plt.show()

    return count_w_h, count_wo_h
    
def get_atom_no_dist(file="qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin", plot=False, geom=False, data_path=None, explicit_h=False,consider_arom_chns_as_atms=False):
    # if "geom_data" in file: return get_atom_no_dist_geom(data_path=data_path, explicit_h=explicit_h)
    data = pickle.load(open(os.path.join(data_path,f"{file.replace('.bin','')}.bin"), "rb"))

    if geom:
        atoms = []
        for i in range(len(data)): 
            if str(i) in data: atoms.append(data[str(i)][1])
    else:
        atoms = []
        for set in ['train', 'test', 'valid']: atoms.extend(data[5][set])
    count_w_h, count_wo_h = {}, {}
    for ind, atm in enumerate(atoms):
        no_w_h,no_wo_h = len(atm), sum([a != 'H' for a in atm])
        aromatic_chns_no = sum([a == 'X' or a == 'Y' for a in atm]) * (not consider_arom_chns_as_atms)# when considering these as atoms (a mistake), do not subtract them from the total number of atoms
        no_w_h,no_wo_h = no_w_h-aromatic_chns_no,no_wo_h-aromatic_chns_no
        count_w_h[no_w_h] = count_w_h[no_w_h] + 1 if no_w_h in count_w_h else 1
        count_wo_h[no_wo_h] = count_wo_h[no_wo_h] + 1 if no_wo_h in count_wo_h else 1

    if plot:
        count_plot = count_w_h if explicit_h else count_wo_h
        plt.bar(count_plot.keys(), count_plot.values())
        plt.show()

    return count_w_h, count_wo_h


def remaining_atms(atm_ind, ordered_dict):
    remaining = 0
    for k,v in ordered_dict:
        if k > atm_ind: remaining += v
    return remaining


def get_bin_atm_upper_lims(file="qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin", return_counts=False, geom=False, data_path=None, explicit_h=False, consider_arom_chns_as_atms=False):
    count_w_h, count_wo_h = get_atom_no_dist(file, plot=False, geom=geom, data_path=data_path, explicit_h=explicit_h, consider_arom_chns_as_atms=consider_arom_chns_as_atms)
    count = count_w_h if explicit_h else count_wo_h
    max_atm_no_cnt = max(count.values())
    ordered_amt_cnts = sorted(count.items(), key=lambda x: x[0])

    total_atms = 0
    bins = {}
    bins_upper_lim = []
    bin_no = 0
    for atm_no, cnt in ordered_amt_cnts:
        total_atms += atm_no*cnt
        # if total_atms >= 0.22 * max_atm_no_cnt and atm_no < 26:
        if total_atms >= 0.22 * max_atm_no_cnt and remaining_atms(atm_no, ordered_amt_cnts)  >= 0.02 * max_atm_no_cnt:
            bins[bin_no] = total_atms
            total_atms = 0; bin_no +=1;
            bins_upper_lim.append(atm_no)
    if total_atms != 0:
        bins[bin_no] = total_atms
        total_atms = 0
        bins_upper_lim.append(atm_no)

    if return_counts: return bins_upper_lim, bins
    return bins_upper_lim

def get_atom_no_bin_class(bins, all_atom_symbols, expl_h=False):
    classes = []
    for atms in all_atom_symbols:
        atm_no = len(atms) - sum([a == 'H' for a in atms]) * (not expl_h)
        classes.append(np.argwhere([atm_no > b[0]  and atm_no <= b[1] for b in bins]).flatten()[0])
    return classes


def create_weights(N_list):
    weights = []
    for n_list in N_list:
        for n in n_list:
            if n != 0: weights.extend([1/n] * n)
            else: weights.append(0)
    return weights

def backward_mdl_compat_(x, N_list, coords, gaussian_indices, std, device, no_fields=9, grid_shapes=[32,32,32], threshold_vlals=True):
    tot_fields = no_fields * len(N_list)
    batch_size = len(N_list)
    std = np.sqrt(std)
    pdf =  torch.exp( -torch.sum((x.unsqueeze(1) - coords.to(device))**2,dim=2) / (2 * std**2) ).to(torch.float32)
    pdf = pdf / ( (2 * np.pi)**(3/2)  * std ** 3)
    if threshold_vlals: pdf[pdf<0.1] = 0
    N_list = np.concatenate(N_list)
    add_to = torch.zeros((pdf.shape[0], tot_fields),dtype=torch.float32, device=device)
    norm_factor = torch.max(pdf, dim=0)[0]; norm_factor[norm_factor == 0] = 1 # avoid division by zero
    pdf = pdf / norm_factor
    pdf = add_to.index_add(1, torch.tensor(gaussian_indices, device=device), pdf)
    pdf = torch.clip(pdf, min=0,max=1)
    pdf = (pdf.T)
    pdf = rearrange(pdf, '(b f) (x y z) -> b f x y z', f=no_fields,b=batch_size, x=grid_shapes[0], y=grid_shapes[1], z=grid_shapes[2])
    return pdf



def create_gaussian_batch_pdf_values(x, N_list, coords, gaussian_indices, std, 
                                     device, no_fields=9, grid_shapes=[32,32,32], threshold_vlals=True,
                                     backward_mdl_compat=False, gmm_mixt_weight=None):
    # the implementation below (backward_mdl_compat) is a bit weird and should only be used to continue and finish training these leftover models
    if type(gmm_mixt_weight) == float: gmm_mixt_weight = torch.ones(no_fields) * 1/gmm_mixt_weight
    if backward_mdl_compat: return backward_mdl_compat_(x, N_list, coords, gaussian_indices, std, device, no_fields=no_fields, grid_shapes=grid_shapes, threshold_vlals=threshold_vlals)
    tot_fields = no_fields * len(N_list)
    batch_size = len(N_list)
    std = np.sqrt(std)
    pdf =  torch.exp( -torch.sum((x.unsqueeze(1) - coords.to(device))**2,dim=2) / (2 * std**2) ).to(torch.float32)
    pdf = pdf / ( (2 * np.pi)**(3/2)  * std ** 3)


    # norm_factor = torch.max(pdf, dim=0)[0]; norm_factor[norm_factor == 0] = 1 # avoid division by zero
    # pdf = pdf / norm_factor
    if threshold_vlals: pdf = pdf * (pdf>0.1)
    N_list = np.concatenate(N_list)
    add_to = torch.zeros((pdf.shape[0], tot_fields),dtype=torch.float32, device=device)
    pdf = add_to.index_add(1, torch.tensor(gaussian_indices, device=device), pdf)
    pdf_min, pdf_max = pdf.min(dim=0)[0], pdf.max(dim=0)[0]
    pdf_normalizer= pdf_max - pdf_min if gmm_mixt_weight is None else gmm_mixt_weight.repeat(batch_size).to(device=device)
    pdf_normalizer[pdf_normalizer == 0] = 1e-9 # avoid division by zero
    pdf = (pdf - pdf_min) / pdf_normalizer
    pdf = (pdf.T)
    pdf = rearrange(pdf, '(b f) (x y z) -> b f x y z', f=no_fields,b=batch_size, x=grid_shapes[0], y=grid_shapes[1], z=grid_shapes[2])
    return pdf


def remove_elements_(remove_elements, all_min_x, all_min_y, all_min_z, all_max_x, all_max_y, all_max_z,resolution=0.33):

    # with 5 * len(all_min_x)//1000
    # New max/min x/y/z:  -9.762400874135755 -6.302432982935396 -4.494940428436074 9.803674834274997 6.211237254243912 4.49941307970304
    # Remaining mols 0.9736609504532774
    # x axis is 0.56 with 0.33Angstrom

    # with 2 * len(all_min_x)//1000
    # New max/min x/y/z:  -10.301968354348524 -6.693850295499541 -4.793473827751677 10.345649379862097 6.629486103122121 4.797422531197209
    # Remaining mols 0.9894755519338633
    # x axis is 0.56 only with 0.37Angstrom

    all_min_x = np.array(all_min_x)
    all_min_y = np.array(all_min_y)
    all_min_z = np.array(all_min_z)
    all_max_x = np.array(all_max_x)
    all_max_y = np.array(all_max_y)
    all_max_z = np.array(all_max_z)
    initial_len = len(all_min_x)

    all_min_x_rmvd_ind = np.argsort(np.array(all_min_x))[:remove_elements]
    all_min_y_rmvd_ind = np.argsort(np.array(all_min_y))[:remove_elements]
    all_min_z_rmvd_ind = np.argsort(np.array(all_min_z))[:remove_elements]
    all_max_x_rmvd_ind = np.argsort(np.array(all_max_x))[-remove_elements:]
    all_max_y_rmvd_ind = np.argsort(np.array(all_max_y))[-remove_elements:]
    all_max_z_rmvd_ind = np.argsort(np.array(all_max_z))[-remove_elements:]

    rmv_inds = np.unique(np.concatenate([all_min_x_rmvd_ind, all_min_y_rmvd_ind, all_min_z_rmvd_ind, all_max_x_rmvd_ind, all_max_y_rmvd_ind, all_max_z_rmvd_ind]))

    all_min_x = np.delete(all_min_x, rmv_inds)
    all_min_y = np.delete(all_min_y, rmv_inds)
    all_min_z = np.delete(all_min_z, rmv_inds)
    all_max_x = np.delete(all_max_x, rmv_inds)
    all_max_y = np.delete(all_max_y, rmv_inds)
    all_max_z = np.delete(all_max_z, rmv_inds)
    print("New max/min x/y/z: ", np.min(all_min_x), np.min(all_min_y), np.min(all_min_z), np.max(all_max_x), np.max(all_max_y), np.max(all_max_z))
    print("Gird numbers", (np.max(all_max_x) - np.min(all_min_x))/resolution, 
          (np.max(all_max_y) - np.min(all_min_y))/resolution, 
          (np.max(all_max_z) - np.min(all_min_z))/resolution)
    print("Remaining mols", len(all_min_x)/initial_len)
    return rmv_inds, np.min(all_min_x), np.min(all_min_y), np.min(all_min_z), np.max(all_max_x), np.max(all_max_y), np.max(all_max_z)


def align_extract_geom_data(data_f,save_conf_limits_file):
    data = pickle.load(open(data_f, "rb"))
    data_aligned = {}
    all_min_x, all_min_y, all_min_z, all_max_x, all_max_y, all_max_z = [],[],[],[],[],[]
    for i in range(len(data)):
        if not data[str(i)][2]: # molecule might not have x many conformers (specified in id of data_f)
            data[str(i)][2] = [[0,0,0]]
        crds = np.array(data[str(i)][2])
        crds = align(crds)
        if i % 1000 == 0: print(i)
        data_aligned[str(i)] = [data[str(i)][0], data[str(i)][1], crds]
        min_x,min_y,min_z = np.min(crds, axis=0)
        max_x,max_y,max_z = np.max(crds, axis=0)
        all_min_x.append(min_x)
        all_min_y.append(min_y)
        all_min_z.append(min_z)
        all_max_x.append(max_x)
        all_max_y.append(max_y)
        all_max_z.append(max_z)

    remove_elements = 5*len(all_min_x)//1000
    rmv_inds, min_min_x, min_min_y, min_min_z, max_max_x, max_max_y, max_max_z = remove_elements_(remove_elements, all_min_x, all_min_y, all_min_z, all_max_x, all_max_y, all_max_z)
    pickle.dump([rmv_inds, {'min_min_x':min_min_x, 'min_min_y':min_min_y, 'min_min_z':min_min_z, 'max_max_x':max_max_x, 'max_max_y':max_max_y, 'max_max_z':max_max_z}], open(save_conf_limits_file, "wb"))
    pickle.dump(data_aligned, open(data_f.replace(".bin", "_aligned.bin"), "wb"))


class GeomDatasetConstructor():
    def __init__(self, no_confs = 3, cls_bins=None, resolution=0.33,
                  subset_confs=30, data_path=None, data_paths=None, 
                  run_name=None, explicit_hydrogen=True,
                  explicit_aromatic=False):
        self.total_no_confs = 5 if explicit_aromatic else 30
        self.subset_confs = min(subset_confs, self.total_no_confs)

        self.data_folder = data_path
        self.splits = pickle.load(open(os.path.join(self.data_folder, "splits.pkl"), "rb"))
        df_info = pd.read_csv(os.path.join(self.data_folder,"mol_summary.csv"))
        self.data_path=data_path
        self.run_name=run_name
        # remove mols containing atms outside list below
        self.restrict_atoms = ('C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'H')
        # use only the subset below of atoms (for removing H mainly/implicit H trainng)
        # * expl aromatic stuff
        self.explicit_aromatic=explicit_aromatic
        self.expl_arom_suffix = "_arom" if explicit_aromatic else ""
        self.rmv_unprocced_aroms = pickle.load(open(os.path.join(data_path, "unprocessed_arom.pkl"), "rb"))
        # * ######

        self.atms_considered = self.restrict_atoms if explicit_hydrogen else tuple([ra for ra in self.restrict_atoms if ra != 'H'])
        self.create_atm_type_filter_inds()
        self.explicit_hydrogen = explicit_hydrogen

        self.smiles=list(df_info['smiles'].values)
        self.cls_bins=cls_bins
        self.classes = None
        self.epoch_conformers_val=None
        self.no_confs=no_confs
        self.get_mol2_available_inds()

        if not os.path.exists(os.path.join(data_path, run_name +"_sample_indinces.bin")): self.create_samples()
        else: 
            print("Loaded current indices from ",os.path.join(data_path, run_name +"_sample_indinces.bin"))
            logging.info("Loaded current indices from " + os.path.join(data_path, run_name +"_sample_indinces.bin"))
            self.all_conf_inds, self.current_conf_index = pickle.load(open(os.path.join(self.data_path, self.run_name +"_sample_indinces.bin"), "rb")); self.current_conf_index+=1;
        # if samples is not None:
        #     self.sampled_conf_inds,self.mol_inds = samples
        self.get_sample_inds()
        
        self.extract_conformers()
        self.resolution=resolution
        self.create_x_y_z_grids()

    def create_samples(self):
        sampled_conf_inds = []
        mol_inds = []
        for i in range(len(self.mol2_inds)):
            # i will select them by conf index, so adding -1 below means this moll will not be selected if empty (no 
            # confs found for it)
            if not self.mol2_inds[i]: 
                sampled_conf_inds.append([-1] * self.total_no_confs); continue
            inds_ = self.mol2_inds[i]; inds_.extend([-1] * (self.total_no_confs - len(inds_)))
            sampled_conf_inds.append(np.random.choice(inds_, self.total_no_confs, replace=False))
            mol_inds.append(i)
        self.all_conf_inds = np.stack(sampled_conf_inds)
        self.current_conf_index = 0
        pickle.dump([self.all_conf_inds, self.current_conf_index], open(os.path.join(self.data_path, self.run_name +"_sample_indinces.bin"), "wb"))

    def get_sample_inds(self):
        if self.current_conf_index >= self.all_conf_inds.shape[1]: self.create_samples()
        self.sampled_conf_inds = self.all_conf_inds[:, self.current_conf_index: self.current_conf_index + self.no_confs]

    def get_mol2_available_inds(self):
        mol2conf_no = pickle.load(open(os.path.join(self.data_path, "geom_conf_number_per_molec.bin"), "rb"))
        mol2_inds = [list(range(self.total_no_confs)) for i in range(len(mol2conf_no))]
        all_grid_min_max = []
        for i in range(0,self.total_no_confs):
            delete_inds, grid_min_max = pickle.load(open(os.path.join(self.data_folder, "conf_{}_limits.bin".format(i+1)), "rb"))
            all_grid_min_max.append(grid_min_max)
            for ind in delete_inds: mol2_inds[ind].remove(i)

        # delete the approx 10 molecules unprocessed bc of rdkit
        if self.explicit_aromatic:
            for i in self.rmv_unprocced_aroms:
                mol2_inds[i] = []
        self.all_grid_min_max = all_grid_min_max
        self.mol2_inds=mol2_inds

    def extract_conformers(self):
        epoch_conformers = [[] for _ in range(self.sampled_conf_inds.shape[1])] 
        epoch_conformers_val = []
        train_inds, test_inds = self.splits['train'], self.splits['val']

        for i in range(0,self.subset_confs):
            conf_file = os.path.join(self.data_path, f"geom_conf{i+1}_aligned{self.expl_arom_suffix}.bin")
            conf_mols = pickle.load(open(conf_file, "rb"))
            if self.classes is None: self.classes = get_atom_no_bin_class(self.cls_bins, [conf[1] for conf in conf_mols.values()], expl_h=self.explicit_hydrogen)

            for batch_ind_epoch in range(self.sampled_conf_inds.shape[1]):
                inds = np.argwhere(self.sampled_conf_inds[:,batch_ind_epoch] == i).flatten() # conf inds for each mol indexed by rows
                inds = np.intersect1d(inds, train_inds)
                inds = np.intersect1d(inds, self.restrict_atm_inds)
                # TODO why do i need to manually check below that conf_mols[str(i)][1] is not empty?
                #! samples should be according to aligned number of conformers 
                #! but now they are according to just non-aligned conformers
                # (that's probably what the issue was)
                # * check without the fix above in TODO
                # epoch_conformers[batch_ind_epoch].extend([[*conf_mols[str(i)], self.smiles[i], self.classes[i]] for i in inds if conf_mols[str(i)][1] ])
                epoch_conformers[batch_ind_epoch].extend([[*conf_mols[str(i)], self.smiles[i], self.classes[i]] for i in inds])

                # append all test conformers (if they exists, i.e. if  conf_mols[str(i)]), if necessary
                if self.epoch_conformers_val is None and batch_ind_epoch == 0: epoch_conformers_val.extend([ [*conf_mols[str(i)],self.smiles[i], self.classes[i]]  for i in test_inds if conf_mols[str(i)][1]])
        self.epoch_conformers_train = self.remove_nonconsidered_atoms(epoch_conformers)
        if self.epoch_conformers_val is None: self.epoch_conformers_val = self.remove_nonconsidered_atoms([epoch_conformers_val])[0] if self.restrict_atoms != self.atms_considered else epoch_conformers_val

    def just_align(self, epoch_conformers):
        discard_atms = []
        for epch_confs in epoch_conformers:
            for ind, conf in enumerate(epch_confs):
                discard_inds = [ind for ind, atm in enumerate(conf[1]) if atm in discard_atms]
                kept_inds = [ind for ind, atm in enumerate(conf[1]) if atm not in discard_atms]
                oold2newinds = {i2:i1  for i1,i2 in enumerate(kept_inds)}
                bonds = [[oold2newinds[b[0]], oold2newinds[b[1]], b[2]] for b in conf[0] if b[0] not in discard_inds and b[1] not in discard_inds]
                atm_positions = align(conf[2][kept_inds])
                atom_symbols = [conf[1][ind] for ind in kept_inds]
                epch_confs[ind] = [bonds, atom_symbols, atm_positions, conf[3], conf[4]]
        return epoch_conformers

    def remove_nonconsidered_atoms(self, epoch_conformers):
        if self.restrict_atoms == self.atms_considered: return self.just_align(epoch_conformers)
        discard_atms = [atm for atm in self.restrict_atoms if atm not in self.atms_considered]
        for epch_confs in epoch_conformers:
            for ind, conf in enumerate(epch_confs):
                discard_inds = [ind for ind, atm in enumerate(conf[1]) if atm in ['H']]
                kept_inds = [ind for ind, atm in enumerate(conf[1]) if atm not in discard_atms]
                oold2newinds = {i2:i1  for i1,i2 in enumerate(kept_inds)}
                bonds = [[oold2newinds[b[0]], oold2newinds[b[1]], b[2]] for b in conf[0] if b[0] not in discard_inds and b[1] not in discard_inds]
                atm_positions = align(conf[2][kept_inds])
                atom_symbols = [conf[1][ind] for ind in kept_inds]
                epch_confs[ind] = [bonds, atom_symbols, atm_positions, conf[3], conf[4]]
        return epoch_conformers
    
    def create_x_y_z_grids(self):
        min_min_min_x, min_min_min_y, min_min_min_z, max_max_max_x, max_max_max_y, max_max_max_z = 0,0,0,0,0,0
        for min_max_vals in self.all_grid_min_max:
            min_min_x, min_min_y, min_min_z, max_max_x, max_max_y, max_max_z = min_max_vals['min_min_x'], min_max_vals['min_min_y'], min_max_vals['min_min_z'], min_max_vals['max_max_x'], min_max_vals['max_max_y'], min_max_vals['max_max_z']
            if min_min_x < min_min_min_x: min_min_min_x = min_min_x
            if min_min_y < min_min_min_y: min_min_min_y = min_min_y
            if min_min_z < min_min_min_z: min_min_min_z = min_min_z
            if max_max_x > max_max_max_x: max_max_max_x = max_max_x
            if max_max_y > max_max_max_y: max_max_max_y = max_max_y
            if max_max_z > max_max_max_z: max_max_max_z = max_max_z
        x_lim,y_lim,z_lim = {"min":min_min_min_x, "max":max_max_max_x}, {"min":min_min_min_y, "max":max_max_max_y}, {"min":min_min_min_z, "max":max_max_max_z}
        resolution = self.resolution
        bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(resolution*8))*8), [x_lim,y_lim,z_lim])]
        x,y,z = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')
        self.x_grid, self.y_grid, self.z_grid = x,y,z
        
    def create_atm_type_filter_inds(self):
        data = pickle.load(open(os.path.join(self.data_folder,f"geom_conf1_aligned{self.expl_arom_suffix}.bin"),"rb"))
        restrict_atm_inds = []
        for d_ind in range(len(data)):
            # explicit aromatic removes 10 or so mols, skip
            if str(d_ind) not in data and self.explicit_aromatic: continue
            atms = data[str(d_ind)][1]
            if sum([atm in self.restrict_atoms for atm in atms]) == len(atms): restrict_atm_inds.append(d_ind)
        self.restrict_atm_inds = np.array(restrict_atm_inds)

    def get_next_epoch_dl(self, train_data_args, val_data_args):
        if not self.epoch_conformers_train: self.get_sample_inds(); self.extract_conformers()
        train_data = self.epoch_conformers_train.pop()
        val_data = self.epoch_conformers_val
        self.current_conf_index += 1
        bonds_train, atm_symb_train, coords_train, smiles_train, no_atms_cls_train = [td[0] for td in train_data], [td[1] for td in train_data], [td[2] for td in train_data], [td[3] for td in train_data], [td[4] for td in train_data]
        bonds_val, atm_symb_val, coords_val, smiles_val, no_atms_cls_val = [vd[0] for vd in val_data], [vd[1] for vd in val_data], [vd[2] for vd in val_data], [vd[3] for vd in val_data], [vd[4] for vd in val_data]
        data = AtomPositionSepBondsDatasetCompact(coords=coords_train, smiles=smiles_train, all_atom_symbols=atm_symb_train, all_bonds=bonds_train, atom_no_classes=no_atms_cls_train, **train_data_args)
        data_val = AtomPositionSepBondsDatasetCompact(coords=coords_val, smiles=smiles_val, all_atom_symbols=atm_symb_val,  all_bonds=bonds_val,atom_no_classes=no_atms_cls_val, **val_data_args)
        return data, data_val


def get_next_ep_data(data_generator, train_dataset_args, val_dataset_args):
    try:
        train_data = data_generator.epoch_conformers_train.pop()
        val_data = data_generator.epoch_conformers_val
    except:
        data_generator.create_sample()
        data_generator.extract_conformers()
        train_data = data_generator.epoch_conformers_train.pop()
        val_data = data_generator.epoch_conformers_val

    bonds_train, atm_symb_train, coords_train, smiles_train, no_atms_cls_train = [td[0] for td in train_data], [td[1] for td in train_data], [td[2] for td in train_data], [td[3] for td in train_data], [td[4] for td in train_data]
    bonds_val, atm_symb_val, coords_val, smiles_val, no_atms_cls_val = [vd[0] for vd in val_data], [vd[1] for vd in val_data], [vd[2] for vd in val_data], [vd[3] for vd in val_data], [vd[4] for vd in val_data]
    data = AtomPositionSepBondsDatasetCompact(coords=coords_train, smiles=smiles_train, all_atom_symbols=atm_symb_train, all_bonds=bonds_train, atom_no_classes=no_atms_cls_train, **train_dataset_args)
    data_val = AtomPositionSepBondsDatasetCompact(coords=coords_val, smiles=smiles_val, all_atom_symbols=atm_symb_val,  all_bonds=bonds_val,atom_no_classes=no_atms_cls_val, **val_dataset_args)
    return data, data_val




def visualize_mol(plot_bnd=0, threshold=0.7, plot_all_atms=False, plot_all_bnd=False, field=None, atm_pos=None, atm_symb=None,actual_bnd=None,
                  atms_required=None, bins=None, batch_ind=None, x_grid=None,y_grid=None,z_grid=None, annotate_atm_no=False, timestep=None, restrict_mol=False,
                  avg_z=True, data='QM9'):
        # create a 3d plot from the Nx3 array atm_bnd_pos
    
    fig = plt.figure()
    # set figure size
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111, projection='3d')
    # for atm in atm_pos:

    min_, max_ = np.min(atm_pos, axis=0) - 0.2, np.max(atm_pos, axis=0) + 0.2


    if "GEOM" in data:
        atm2color = {'C':'black', 'N':'blue', 'O':'red', 'F':'green', 'P':'purple', 'S':'olive', 'Cl':'mediumseagreen', 'H':'white', 'X':'pink', 'Y':'darkviolet'}
        bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}
    else:
        atm2color = {'C':'black', 'O':'red', 'N':'blue', 'F':'green', 'H':'white', 'X':'pink', 'Y':'darkviolet'}
        bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}        

    colors = [atm2color[a] for a in atm_symb]
    ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300, edgecolor='black', alpha=0.5)
    if annotate_atm_no:
        for ind, atm_p in enumerate(atm_pos):
            ax.text(atm_p[0], atm_p[1], atm_p[2], "{}".format(ind),  fontsize=15)

    # actual_bnd = [actual_bnd]
    for bnd in actual_bnd:
        bnd_inds = [bnd[0], bnd[1]]
        line = atm_pos[bnd_inds]
        dist = np.linalg.norm(line[0]-line[1])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=3)
        bond_position = (line[0]+line[1])/2
        # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
    if plot_all_atms:
        for plot_bnd in range(len(atm2color) -2 if 'X' in atm2color else len(atm2color)):
            if restrict_mol:
                plot_inds =  (x_grid > min_[0]) * (x_grid < max_[0]) * (y_grid > min_[1]) * (y_grid < max_[1]) * (z_grid > min_[2]) * (z_grid < max_[2])
                if threshold is None: threshold = 0.7 * np.max(field[plot_bnd][plot_inds])
                plot_inds = plot_inds * (field[plot_bnd]>threshold)

                # (field[plot_bnd]>threshold) *
                x = x_grid[plot_inds]
                y = y_grid[plot_inds]
                z = z_grid[plot_inds]
                scatter =ax.scatter(x,y,z,c=field[plot_bnd][plot_inds], edgecolor='black')
            else:
                x = x_grid[field[plot_bnd]>threshold]
                y = y_grid[field[plot_bnd]>threshold]
                z = z_grid[field[plot_bnd]>threshold]
                alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
                size = alpha * 50
                scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold],  s=20, )
        plt.colorbar(scatter)
    elif plot_all_bnd:
        for plot_bnd in range(len(atm2color), field.shape[0]):
            x = x_grid[field[plot_bnd]>threshold]
            y = y_grid[field[plot_bnd]>threshold]
            z = z_grid[field[plot_bnd]>threshold]
            alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
            size = alpha * 50
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold],s=20, )
        plt.colorbar(scatter)
    elif plot_bnd != -1:
        if restrict_mol:
            plot_inds =  (x_grid > min_[0]) * (x_grid < max_[0]) * (y_grid > min_[1]) * (y_grid < max_[1]) * (z_grid > min_[2]) * (z_grid < max_[2])
            plot_inds = (x_grid < 1000)
            if threshold is None: 
                threshold = 0.1 * np.max(field[plot_bnd][plot_inds])
            plot_inds = plot_inds * (field[plot_bnd]>threshold)

            # (field[plot_bnd]>threshold) *
            x = x_grid[plot_inds]
            y = y_grid[plot_inds]
            z = z_grid[plot_inds]
            # alpha=field[plot_bnd][plot_inds]**2 if len(field[plot_bnd][plot_inds]) else 1
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][plot_inds], s=20)
        else:
            x = x_grid[field[plot_bnd]>threshold]
            y = y_grid[field[plot_bnd]>threshold]
            z = z_grid[field[plot_bnd]>threshold]
            alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
            size = alpha * 50
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold], s=20)
        plt.colorbar(scatter)

    ax.set_xlim3d(x_grid.min(), x_grid.max())
    ax.set_ylim3d(x_grid.min(), x_grid.max())
    ax.set_zlim3d(x_grid.min(), x_grid.max())

    # for bnd in candidaten_bnds:
    timestep_fn_save = "" if timestep is None else "x_{}".format(timestep)
    timestep = "" if timestep is None else ". $x_{{{}}}$".format(timestep)
    if atms_required is not None: 
        bin_class = atms_required[batch_ind]
        bin = "{} < n <= {}".format(bins[bin_class-1], bins[bin_class])
        plt.title(plt.title("Moleule {} failed. Required atom number {}".format(batch_ind,bin)))
    else: plt.title("Moleule {} failed".format(batch_ind) + timestep)
    plt.show()
    # plt.savefig("/home/alex/Desktop/molecule_{}".format(batch_ind)+ timestep_fn_save+ ".png")



def check_containing_H(data):
    for a_s in data.all_atom_symbols:
        if 'H' in a_s: return True
    return False


def get_cond_var_normalizing_factors(dataset, args):
    if not args.cond_variable: return None
    all_props = dataset.cond_variables
    return np.min(all_props, axis=0), np.max(all_props, axis=0)


class ConditioningVarSampler():
    def __init__(self, data, norm_factors):

        cond_var_bins = np.linspace(norm_factors[0], norm_factors[1], 1001).flatten()  # 1001 because it includes both end points
        cond_var_count_bins = {}
        cond_var_ind_2_bins = {i: [cond_var_bins[i], cond_var_bins[i + 1]] for i in range(len(cond_var_bins) - 1)}
        cond_var_ind_2_no_atm_clses = {i:[] for i in range(len(cond_var_bins) - 1)}
        cond_var_bins[-1] += 0.001
        for no_atm_cls, cond_v in zip(data.atom_no_classes, data.cond_variables):
            cond_v = cond_v[0]
            # if it's the minimum (lowest in cond_var_bins, set ind=0)
            cond_v_ind = np.max([np.argwhere(cond_v <= cond_var_bins)[0][0] -1 , 0])
            cond_var_count_bins[cond_v_ind] = 1 if cond_v_ind not in cond_var_count_bins else cond_var_count_bins[cond_v_ind] + 1
            cond_var_ind_2_no_atm_clses[cond_v_ind].append(no_atm_cls)

        num_datapoints = sum([len(v) for v in cond_var_ind_2_no_atm_clses.values()])
        cond_var_bins_dist = [[k for k in cond_var_count_bins.keys()]]
        cond_var_bins_dist.append([cond_var_count_bins[k] / num_datapoints for k in cond_var_bins_dist[0]])

        cond_var_ind_2_atm_no_cls_dist = {}

        for cond_ind, cond_class_nos in cond_var_ind_2_no_atm_clses.items():
            cond_cls_no_count = {}
            total_ = len(cond_class_nos)
            for cls_no in cond_class_nos:
                cond_cls_no_count[cls_no] = 1 if cls_no not in cond_cls_no_count else cond_cls_no_count[cls_no] + 1
            cond_cls_no_count = {k:v/total_ for k,v in cond_cls_no_count.items()}
            cond_var_ind_2_atm_no_cls_dist[cond_ind] = [[k for k in cond_cls_no_count]]
            cond_var_ind_2_atm_no_cls_dist[cond_ind].append([cond_cls_no_count[k] for k in cond_var_ind_2_atm_no_cls_dist[cond_ind][0]])
        self.cond_var_bins_dist = cond_var_bins_dist
        self.cond_var_ind_2_atm_no_cls_dist = cond_var_ind_2_atm_no_cls_dist
        self.cond_var_ind_2_bins = cond_var_ind_2_bins

    def sample(self, num_points=1):

        classes, cond_vars = [], []

        for _ in range(num_points):
            cond_var_bin = np.random.choice(self.cond_var_bins_dist[0], p=self.cond_var_bins_dist[1])
            no_of_atms_class = np.random.choice(self.cond_var_ind_2_atm_no_cls_dist[cond_var_bin][0], p=self.cond_var_ind_2_atm_no_cls_dist[cond_var_bin][1])
            cnd_var_bin_val = np.random.uniform(self.cond_var_ind_2_bins[cond_var_bin][0], self.cond_var_ind_2_bins[cond_var_bin][1])
            classes.append(no_of_atms_class)
            cond_vars.append(cnd_var_bin_val)

        return np.array(classes), np.array(cond_vars)
        # cond_var_bins_dist sample p(cond_var)
        # cond_var_ind_2_atm_no_cls_dist sample p(no_atms|cond_var)
        # cond_var_ind_2_bins get the interval for cond_var_ind sampled
        # sample uniformly from the bin
            
        

