# create a torch model containing N number of mean atom positions (as learnable parameters), with equal and learnable weight parameter (for the gaussian components) and a fixed variance parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from pysmiles import read_smiles

from collections import namedtuple
import networkx as nx

from sklearn.mixture import GaussianMixture

from utils import align, update_mol_positions

import logging
logging.getLogger('some_logger')

# distances to bond types


def plot_one_mol(atm_pos, actual_bnd, plot_bnd=None, threshold=0.7, field=None, x_grid=None,y_grid=None,z_grid=None, atm_symb=None, title=None, unique_atm_symbols=None):

    import matplotlib.pyplot as plt
    
    atm2color = {'C':'black', 'O':'red', 'N':'blue', 'F':'green', 'H':'white'}
    bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}
    if len(unique_atm_symbols) > 5:
        atm2color = {'C':'black', 'N':'blue', 'O':'red', 'F':'green', 'P':'purple', 'S':'olive', 'Cl':'mediumseagreen', 'H':'white'}
        bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [atm2color[a] for a in atm_symb]
    ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300, edgecolor='black')
    for bnd in actual_bnd:
        bnd_inds = [bnd[0], bnd[1]]
        line = atm_pos[bnd_inds]
        dist = np.linalg.norm(line[0]-line[1])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=3)
        bond_position = (line[0]+line[1])/2
        # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
    if plot_bnd is not None:
        x = x_grid[field[plot_bnd]>threshold]
        y = y_grid[field[plot_bnd]>threshold]
        z = z_grid[field[plot_bnd]>threshold]
        scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold])
        plt.colorbar(scatter)
    # for bnd in candidaten_bnds:
    if title is not None:
        plt.title(title)
    plt.show()


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

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])



def get_bond_order(atom1,atom2,distance,check_exists=False, return_dists=False):
    distance = 100 * distance 

    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0
        
    if distance < bonds1[atom1][atom2] + margin1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        if return_dists: return distance, 3
                        else: return 3        # Triple
                if return_dists: return distance, 2
                else: return 2
        if return_dists: return distance, 1                # Single
        else: return 1
    if return_dists: return distance, 0                    
    return 0


def get_sphere_centers_deterministic(tensor,x_grid, y_grid, z_grid,normalize01, set_threshold=None):
    """
        deterministic algorithm to find the centers of spheres in a 3D tensor. Since the average pdf value is generally
        in the ballpark of 0, we can use (max_pdf - avg_pdf) to determine if that distance is say > 0.5, 
        therefore identifying that we indeed have some sort of spheres. Then retrieve elements that cover e.g.
        [4/5 * max_pdf_value, max_pdf_value]
    """

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
            # breakpoint()
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


    sphere_centers = []
    for sphere_points, sphere_pdf in zip(spheres, spheres_pdfs):

        sphere_weight_points = np.array(sphere_pdf)/np.sum(sphere_pdf)
        center_sphere_ = np.sum(sphere_points.T * sphere_weight_points, axis=1) if len(sphere_points) > 1\
              else sphere_points[0] 

        center_sphere = center_sphere_.astype(int)

        center_sphere_coordinates = x_grid[center_sphere[0],center_sphere[1],center_sphere[2]], \
                                    y_grid[center_sphere[0],center_sphere[1],center_sphere[2]], \
                                    z_grid[center_sphere[0],center_sphere[1],center_sphere[2]]
        
        leftover_decimals = center_sphere_ - center_sphere
        leftover_per_axis = leftover_decimals * np.array([x_resolution, y_resolution, z_resolution])

        center_sphere_coordinates = [center_sphere_coordinates[i] + leftover_per_axis[i] for i in range(3)]


        sphere_centers.append(list(center_sphere_coordinates))
    return np.array(sphere_centers)

def extract_positions_mulitple_cutoffs(pred_dens, cutoff_steps, cutoff,x_flat,y_flat,z_flat):
    lower_limits = np.linspace(cutoff, np.max(pred_dens), cutoff_steps+1)[:-1]

    all_x, all_y, all_z = [], [], []

    for lower_lim in lower_limits:
        coord_inds = np.where(pred_dens > lower_lim)
        all_x.extend(x_flat[coord_inds])
        all_y.extend(y_flat[coord_inds])
        all_z.extend(z_flat[coord_inds])

    return np.array(all_x), np.array(all_y), np.array(all_z)


def check_validity(pos,atoms,channel_atoms):
    mol = build_molecule(pos,atoms,channel_atoms)
    smiles = mol2smiles(mol)
    if smiles is not None:
        return 1, smiles
    else:
        return 0, None


def build_molecule_from_position(pos,atoms, return_dists=False):
    n = pos.shape[0]
    X = atoms
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)
    D = torch.zeros((n, n), dtype=torch.float)

    pos = pos.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atoms[i], atoms[j]])
            if return_dists:
                dist, order = get_bond_order(pair[0], pair[1], dists[i, j], return_dists=return_dists)
            else:
                order = get_bond_order(pair[0], pair[1], dists[i, j], return_dists=return_dists)
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
                if return_dists: D[i, j] = dist
    if return_dists:

        return X,A,E,D
    else:
        return X,A,E



def build_molecule_return_bonds(pos,atoms,channel_atoms, return_dists=False, return_bonds=False):

    if return_dists:
        X,A,E,D = build_molecule_from_position(pos,atoms, return_dists=return_dists)
    else:
        X,A,E = build_molecule_from_position(pos,atoms)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom)
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    all_bonds_and_types = []
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
        all_bonds_and_types.append([bond[0].item(), bond[1].item(), E[bond[0], bond[1]].item()])
    return mol, all_bonds_and_types


def build_molecule(pos,atoms,channel_atoms, return_dists=False, return_bonds=False):

    if return_dists:
        X,A,E,D = build_molecule_from_position(pos,atoms, return_dists=return_dists)
    else:
        X,A,E = build_molecule_from_position(pos,atoms)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom)
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    if return_dists:
        return mol,D
    return mol


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol,kekuleSmiles=True)


def exists(x):
    return x is not None

def compute_uniqueness(valid):
    return len(set(valid)) / len(valid)


def compute_novelty(unique,smiles_list):
    num_novel = 0
    for smiles in unique:
        if smiles not in smiles_list:
            num_novel +=1
    
    return num_novel / len(unique)

def extract_N_inds_from_atms(all_atm_symbs, explicit_hydrogen=False, unique_atm_symbols=None):
    if unique_atm_symbols is None:
        unique_atm_symbols = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]
    N_list = []
    for atm_symbs in all_atm_symbs:
        for u_a in unique_atm_symbols:
            N_list.append(atm_symbs.count(u_a))
    return N_list





class GMMBatchModelPositions(nn.Module):
    def __init__(self, N_list, variance, initial_guesses, learn_weights=False, learn_nothing=False, gaussian_indices=None) -> None:
        super(GMMBatchModelPositions, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N_list = N_list
        self.std = np.sqrt(variance) 
        

        if exists(initial_guesses):
            initial_guess = torch.tensor(np.concatenate(initial_guesses)) if type(initial_guesses) == list else initial_guesses
            self.mean_positions = nn.Parameter(initial_guess) if not learn_weights else initial_guess
        else:
            self.mean_positions = nn.Parameter(torch.randn(N, 3)) if not learn_weights else torch.randn(N, 3)
        # self.mean_positions = nn.Parameter(torch.randn(N, 3)) 
        weights = np.concatenate( [ [0.2 for _ in range(N)] for N in N_list])
        self.weights = torch.tensor(weights, device=self.device)
        self.weights = torch.nn.Parameter(self.weights)

        # after weighted pdf computation, we need to sum over the gaussian components (and there are multiple 
        # GMMs, so gaussian_indices will be a vector like [0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2] of same dim as self.weights
        # specifying where each component goes (first GMM 0, second 1, etc)
        self.gaussian_indices = torch.tensor(np.concatenate([ [i for _ in range(N)] for i,N in enumerate(N_list)])).to(self.device, dtype=torch.int32)

    def forward(self, x):
        pdf =  torch.exp( -torch.sum((x.unsqueeze(1) - self.mean_positions)**2,dim=2) / (2 * self.std**2) )
        pdf = pdf / ( (2 * np.pi)**(3/2)  * self.std ** 3 )
        pdf = pdf * self.weights
        

        add_to = torch.zeros((pdf.shape[0],len(self.N_list)),dtype=torch.float64, device=self.device)
        pdf = torch.index_add(add_to, 1, self.gaussian_indices, pdf)
        return pdf

class GMMBatchModel(nn.Module):
    def __init__(self, N_list, variance, initial_guesses, learn_weights=False, learn_nothing=False, gaussian_indices=None) -> None:
        super(GMMBatchModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N_list = N_list
        self.std = np.sqrt(variance) 
        

        if exists(initial_guesses):
            initial_guess = torch.tensor(np.concatenate(initial_guesses)) if type(initial_guesses) == list else initial_guesses
            self.mean_positions = nn.Parameter(initial_guess) if not learn_weights else initial_guess
        else:
            self.mean_positions = nn.Parameter(torch.randn(N, 3)) if not learn_weights else torch.randn(N, 3)
        # self.mean_positions = nn.Parameter(torch.randn(N, 3)) 
        self.weights = np.concatenate( [ [1/5.679043443503446 for _ in range(N)] for N in N_list])
        self.weights = torch.tensor(self.weights, device=self.device)
        if learn_weights:
            self.weights = torch.nn.Parameter(self.weights)
        if learn_nothing:
            self.weights = initial_guesses.to(self.device) if type(initial_guesses) == torch.tensor else torch.tensor(initial_guesses, device=self.device)
            gaussian_indices = torch.tensor(np.concatenate([ [i for _ in range(N)] for i,N in enumerate(N_list)])).to(self.device) if gaussian_indices is None else gaussian_indices
            self.gaussian_indices = gaussian_indices

        # after weighted pdf computation, we need to sum over the gaussian components (and there are multiple 
        # GMMs, so gaussian_indices will be a vector like [0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2] of same dim as self.weights
        # specifying where each component goes (first GMM 0, second 1, etc)
        self.gaussian_indices = torch.tensor(np.concatenate([ [i for _ in range(N)] for i,N in enumerate(N_list)])).to(self.device)

    def forward(self, x):
        pdf =  torch.exp( -torch.sum((x.unsqueeze(1) - self.mean_positions)**2,dim=2) / (2 * self.std**2) )
        pdf = pdf / ( (2 * np.pi)**(3/2)  * self.std ** 3 )
        pdf = pdf * self.weights
        

        add_to = torch.zeros((pdf.shape[0],len(self.N_list)),dtype=torch.float64, device=self.device)
        pdf = torch.index_add(add_to, 1, self.gaussian_indices, pdf)
        return pdf

class GaussianModel(nn.Module):
    def __init__(self, N, variance,initial_guess):
        super(GaussianModel, self).__init__()
        self.N = N
        self.std = np.sqrt(variance) 
        

        if exists(initial_guess):
            initial_guess = torch.tensor(initial_guess)
            # initial_guess = initial_guess + torch.randn_like(initial_guess)*0.15
            self.mean_positions = nn.Parameter(initial_guess)
        else:
            self.mean_positions = nn.Parameter(torch.randn(N, 3)) 
        # self.mean_positions = nn.Parameter(torch.randn(N, 3)) 
        self.weights = 1/N
        
    def forward(self, x):


        # get multivariate gaussian pdf of all inputs x given the mean positions and the variance
        


        # get the multivariate normal pdf for each atom position
        # x is a tensor of shape (batch_size, 3)
        # mean_positions is a tensor of shape (N, 3)
        # breakpoint()

        # non-log
        pdf =  torch.exp( -torch.sum((x.unsqueeze(1) - self.mean_positions)**2,dim=2) / (2 * self.std**2) )
        pdf = pdf / ( (2 * np.pi)**(3/2)  * self.std ** 3 )
        pdf = torch.sum(pdf * self.weights,dim=1)
        return pdf


        # log_pdf = torch.sum((x.unsqueeze(1) - self.mean_positions)**2,dim=2)
        # log_pdf = -log_pdf - np.log( (2 * np.pi)**(3/2)  * self.std ** 3 ) 
        # return log_pdf.sum(dim=1)

def check_same_bond(all_atms, all_bonds, all_bonds_dist_based):
    same_mols = 0
    total_atoms = 0
    total_same_atoms = 0
    all_atm2bnd_dist, all_atm2bnd_sep_chn = [],[] 
    for atm_list, bnd_sep_chn, bnd_dist_based in zip(all_atms, all_bonds, all_bonds_dist_based):
        same_atoms = 0
        atm2bnds_sep_chn, atm2bnds_dist_based = {a_ind:[] for a_ind in range(len(atm_list))}, {a_ind:[] for a_ind in range(len(atm_list))}
        for bnd_sep in bnd_sep_chn: 
            atm2bnds_sep_chn[bnd_sep[0]].append(tuple(bnd_sep.tolist()))
            atm2bnds_sep_chn[bnd_sep[1]].append(tuple(bnd_sep.tolist()))
            atm2bnds_sep_chn[bnd_sep[0]].append(tuple([bnd_sep[1], bnd_sep[0], bnd_sep[2]]))
            atm2bnds_sep_chn[bnd_sep[1]].append(tuple([bnd_sep[1], bnd_sep[0], bnd_sep[2]]))
        for bnd_dist in bnd_dist_based: 
            atm2bnds_dist_based[bnd_dist[0]].append(tuple(bnd_dist))
            atm2bnds_dist_based[bnd_dist[1]].append(tuple(bnd_dist))
            atm2bnds_dist_based[bnd_dist[0]].append(tuple([bnd_dist[1],bnd_dist[0],bnd_dist[2]]))
            atm2bnds_dist_based[bnd_dist[1]].append(tuple([bnd_dist[1],bnd_dist[0],bnd_dist[2]]))
        all_atm2bnd_dist.append(atm2bnds_dist_based)
        all_atm2bnd_sep_chn.append(atm2bnds_sep_chn)
        for a_ind in range(len(atm_list)): 
            if set(atm2bnds_dist_based[a_ind]) == set(atm2bnds_sep_chn[a_ind]): same_atoms +=1
        if same_atoms == len(atm_list): same_mols +=1
        total_atoms += len(atm_list)
        total_same_atoms += same_atoms
    print(total_same_atoms/total_atoms, same_mols/len(all_atms), total_same_atoms, total_atoms, same_mols)
    stable_atms_dist, stable_mols_dist = check_mol_atm_stability(all_atms, all_atm2bnd_dist)
    stable_atms_sepchn, stable_mols_spechn = check_mol_atm_stability(all_atms, all_atm2bnd_sep_chn)
    print("Stable sepchan atoms/mols: {}/{}".format(stable_atms_sepchn/total_atoms, stable_mols_spechn/len(all_atms)))
    print("Stable distbased atoms/mols: {}/{}".format(stable_atms_dist/total_atoms, stable_mols_dist/len(all_atms)))
    breakpoint()

def check_mol_atm_stability(all_atms, bonds):
    atm2valency = {'C':4, 'O':2, 'N':3, 'F':1, 'H':1}
    stable_mols = 0
    tot_stable_atms = 0
    for atm_list, bnds in zip(all_atms, bonds):
        stable_atms = 0
        atm_stability = []
        for atm, bnds_ in bnds.items():
            correct_val = atm2valency[atm_list[atm]]
            pred_val = sum([b[2] for b in bnds_]) // 2
            stable_atms += pred_val == correct_val; atm_stability.append(pred_val == correct_val)
        if stable_atms == len(atm_list): stable_mols +=1
        tot_stable_atms += stable_atms
    return tot_stable_atms, stable_mols

def extract_positions_batch(pred_dens,x_flat,y_flat,z_flat, num_atoms, std, initial_guess):


    model = GMMBatchModelPositions(num_atoms, std, initial_guess)
    optimizer = torch.optim.SGD(model.parameters(), lr=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_flat,y_flat,z_flat = torch.tensor(x_flat.flatten(),device=device), \
        torch.tensor(y_flat.flatten(),device=device), torch.tensor(z_flat.flatten(),device=device)
    input_ = torch.stack([x_flat,y_flat,z_flat],dim=1)

    # scaling is based also on the number of atoms (it shouldn't integrate to 1 but to smth else)
    # I ALSO CHANGED THE RESOLUTIONS
    model = model.to(device)
    pred_dens = np.stack(pred_dens)
    pred_dens = np.reshape(pred_dens, (-1, np.prod(pred_dens.shape[2:])))

    pred_dens = torch.tensor(pred_dens).to(device)
    input_ = input_.to(device)

    # pred_dens = pred_dens / (torch.sum(pred_dens) *np.prod(new_resolutions))

    for i in range(500):
        # print(model.mean_positions)
        optimizer.zero_grad()
        output = model(input_)
        loss = torch.mean((output.T - pred_dens)**2, dim=1)
        # if i % 20 == 0:
        #     print("Step {} loss was {}".format(i,torch.max(loss).item()))
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
    return model.mean_positions

def extract_positions(pred_dens,x_flat,y_flat,z_flat, num_atoms, std, new_resolutions,initial_guess):

    model = GaussianModel(num_atoms, std, initial_guess)
    optimizer = torch.optim.SGD(model.parameters(), lr=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_flat,y_flat,z_flat = torch.tensor(x_flat.flatten(),device=device), \
        torch.tensor(y_flat.flatten(),device=device), torch.tensor(z_flat.flatten(),device=device)
    input_ = torch.stack([x_flat,y_flat,z_flat],dim=1)

    # scaling is based also on the number of atoms (it shouldn't integrate to 1 but to smth else)
    # I ALSO CHANGED THE RESOLUTIONS
    model = model.to(device)
    pred_dens = torch.tensor(pred_dens.flatten(),device=device)


    # pred_dens = pred_dens / (torch.sum(pred_dens) *np.prod(new_resolutions))

    for i in range(10000):

        optimizer.zero_grad()
        output = model(input_)
        # breakpoint
        loss = torch.mean((output - pred_dens)**2)

        print(loss.item())
        loss.backward()
        optimizer.step()
    return model.mean_positions



def clean_smiles(smile):
    if smile is None:
        return smile
    for i in range(2,5):
        smile = smile.replace("H{}".format(i),"")
    smile = smile.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "").replace("+","").replace("-","").upper().replace("()","")
    # these are also equivalent
    if "21" in smile:
        smile = smile.replace("21","12")
    return smile

def get_potential_atm_pair_positions(atm_bnd_pos, mol_bnd_symb,explicit_hydrogen=False,unique_atm_symbols=None):
    """
        atm_bnd_pos: (N,3) this variable may or may not contain bond positions, depending on the method;
                     see fit_pred_field_sep_chn_batch documentation, determine_bonds argument
    """
    if unique_atm_symbols is None:
        unique_atms = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]
    else:
        unique_atms = unique_atm_symbols
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
            if bonds1[atm1][atm2]+35 > distances[i,j]: candidate_bnds.append((i,j,atom_positions[i],atom_positions[j]))
    return candidate_bnds, new2old_index

def get_actual_bnds(candidaten_bnds,atoms_and_bond_positions, mol_bond_symbols):

    # get all bond types and their positions
    all_bond_positions, all_bond_types = [], []
    for atm_or_bnd, atm_or_bnd_pos in zip(mol_bond_symbols,atoms_and_bond_positions):
        if atm_or_bnd in [1,2,3]: # if it's a bond
            all_bond_positions.append(atm_or_bnd_pos)
            all_bond_types.append(atm_or_bnd)

    all_bond_positions = np.stack(all_bond_positions)

    final_bonds = []
    for cnd_bnd in candidaten_bnds:

        center_candidate_bond = (cnd_bnd[2] + cnd_bnd[3])/2
        candidate_2_actual_bnd_dists = np.linalg.norm(all_bond_positions - center_candidate_bond,axis=1)
        bnd_ind, bnd_dist = np.argmin(candidate_2_actual_bnd_dists), np.min(candidate_2_actual_bnd_dists)
        
        if bnd_dist < 0.25:
            bnd_type = all_bond_types[bnd_ind]
            final_bonds.append((cnd_bnd[0],cnd_bnd[1],bnd_type))

    return final_bonds

def get_pdf_probs(distances,var):
    return np.exp(-distances**2/(2*var))/np.sum(np.exp(-distances**2/(2*var)))


def get_actual_bonds_through_pdf(candidaten_bnds,field,x,y,z,var=0.05,normalize01=False, explicit_hydrogen=False,threshold_bond=0.75,unique_atm_symbols=None):
    """
        for all atom distances that are within certain thresholds, determine the pdf value over bond
        channels to get "actual bonds"
    """
    positions = np.stack([x.flatten(),y.flatten(),z.flatten()],axis=1)
    actual_bonds = []

    # get the bond channels that have connections and determine the mean-max values
    removed = []

    # TODO compute the constant s.t. a pdf with var=0.05 would have a max val determined in the field; then, 
    # compute the pdf value 15 pm away, divide by said constant and that's ur threshold
    per_bnd_thresholds = []
    
    start_bnd = 4+explicit_hydrogen if unique_atm_symbols is None else len(unique_atm_symbols)
    for bnd_in, bnd_ch in enumerate(field[start_bnd:]):
        per_bnd_thresholds.append(threshold_bond); # continue
        # if normalize01: per_bnd_thresholds.append(0.75); continue
        # mean, max_val = np.mean(bnd_ch), np.max(bnd_ch)
        # per_bnd_thresholds.append((max_val - mean)*0.5) if max_val - mean > 0.5 else per_bnd_thresholds.append(10000)

    candidate_excluded=True
    for index_cb, cb in enumerate(candidaten_bnds):
        mean_position = (cb[2] + cb[3])/2
        for bnd_ind, bond_chn in enumerate(field[start_bnd:]):
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
        if candidate_excluded: removed.append(index_cb)
        candidate_excluded=True
        # print("\n")
    return actual_bonds


def rmv_gmm_low_bnd_weights(all_actual_bnd_positions, all_actual_bonds, indices, field, x_flat,y_flat,z_flat,corresp_field, return_last_loss=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_actual_bnd_positions = torch.tensor(np.stack(all_actual_bnd_positions)).to(device)

    model = GMMBatchModel(N_list=indices, variance=0.05, initial_guesses=all_actual_bnd_positions, learn_weights=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_flat,y_flat,z_flat = torch.tensor(x_flat.flatten(),device=device), \
        torch.tensor(y_flat.flatten(),device=device), torch.tensor(z_flat.flatten(),device=device)
    input_ = torch.stack([x_flat,y_flat,z_flat],dim=1)

    # scaling is based also on the number of atoms (it shouldn't integrate to 1 but to smth else)
    # I ALSO CHANGED THE RESOLUTIONS
    model = model.to(device)
    pred_dens = np.stack(corresp_field)
    pred_dens = np.reshape(pred_dens, (-1, np.prod(pred_dens.shape[1:])))

    pred_dens = torch.tensor(pred_dens).to(device)
    input_ = input_.to(device)

    # pred_dens = pred_dens / (torch.sum(pred_dens) *np.prod(new_resolutions))
    for i in range(200):

        optimizer.zero_grad()
        output = model(input_)

        loss = (output.T - pred_dens)**2
        last_loss = torch.max(loss,dim=1)[0]
        loss = torch.mean(loss, dim=1)
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
    if return_last_loss: return model.weights.data, last_loss.detach().cpu().numpy()
    return model.weights.data



def reorder_inds_get_bnd_numbers(unique_bnds, actual_bnd):
    ind = 0 
    indices = []
    bnd2count = {u:0 for u in unique_bnds}
    for b in unique_bnds:
        for ind, b_ in enumerate(actual_bnd):
            if b_[2] == b: bnd2count[b]+=1; indices.append(ind)
    return bnd2count, indices
    
def remove_low_weight_bonds(corresp_weights, actual_bonds, indices, pi_threshold):



    mol_bnd_type_numbers = []
    current_bond_numbers_per_ch = []
    current_no_bonds = 0
    current_molecule = 0
    current_bond_types = []
    bond_types = []
    for ind in indices:
        current_bond_numbers_per_ch.append(ind)
        # * sanity check that all bnds come in the order bnd type 1,2,3...
        # for ind_ in range(1,len(actual_bonds[current_molecule])):
        #     if actual_bonds[current_molecule][ind_][2] < actual_bonds[current_molecule][ind_-1][2]: 
        #         breakpoint()
        if len(actual_bonds[current_molecule]) == ind + current_no_bonds:
            current_bond_types.append(actual_bonds[current_molecule][-1][2])
            bond_types.append(current_bond_types)
            mol_bnd_type_numbers.append(current_bond_numbers_per_ch)
            current_bond_numbers_per_ch = []
            current_molecule += 1
            current_no_bonds=0
            current_bond_types = []
        else:
            current_bond_types.append(actual_bonds[current_molecule][current_no_bonds][2])
            current_no_bonds += ind

    current_elapsed_weights = 0
    removed_bnds_mask = []
    all_new_bonds = []
    weights_per_molecule = []
    all_rmvd_bonds = []
    for bnd, ch_based_inds, bnd_type in zip(actual_bonds, mol_bnd_type_numbers, bond_types):
        current_mol_elapsed_bonds = 0
        current_remove_bnds = []
        weights_per_ch = {1:[],2:[],3:[],4:[]}
        for i, inds in enumerate(ch_based_inds):
            current_bnd_type = bnd_type[i]
            weights = corresp_weights[current_elapsed_weights+current_mol_elapsed_bonds:
                                      current_elapsed_weights+current_mol_elapsed_bonds+inds]
            weights = weights.cpu().detach().numpy()
            if len(weights) == 1: remove_bnds = [False]
            else: remove_bnds = weights < pi_threshold * np.max(weights)
            current_remove_bnds.extend(remove_bnds)
            current_mol_elapsed_bonds+=inds
            weights_per_ch[current_bnd_type].append(weights)
        weights_per_molecule.append(weights_per_ch)
        current_elapsed_weights += len(bnd)
        removed_bnds_mask.append(current_remove_bnds)
        new_bnd = [b for b,rmv in zip(bnd, current_remove_bnds) if not rmv]
        all_new_bonds.append(new_bnd)
        all_rmvd_bonds.append([b for b, rmv in zip(bnd, current_remove_bnds) if rmv])
    return all_new_bonds, removed_bnds_mask, weights_per_molecule, all_rmvd_bonds



def gather_and_correct_bonds(atoms_and_bond_positions,molecules_atom_or_bond_symbols, field, normalize01, explicit_hydrogen,
                             check_inbetween_bonds_pdf,x,y,z,explicit_aromatic, threshold_bond, pi_threshold,unique_atm_symbols):
    
    # TODO I NEED TO ALSO SEPARATE TO EACH CORRESPONDING BOND TYPE 
    all_actual_bonds, all_actual_bnd_positions, indices, corresp_field = [], [], [], []
    
    start_bnd = len(unique_atm_symbols) - 1 if unique_atm_symbols is not None else 3+explicit_hydrogen
    unique_bnds = [1,2,3,4] if explicit_aromatic else [1,2,3]
    number_of_bonds = []
    all_candidate_bnds = []
    for ind,(atm_bnd_pos, mol_bnd_symb) in enumerate(zip(atoms_and_bond_positions,molecules_atom_or_bond_symbols)):
        candidaten_bnds,new2old_index = get_potential_atm_pair_positions(atm_bnd_pos, mol_bnd_symb,explicit_hydrogen, unique_atm_symbols)
        all_candidate_bnds.append(candidaten_bnds)
        actual_bnd = get_actual_bonds_through_pdf(candidaten_bnds, field[ind],x,y,z,var=0.05,normalize01=normalize01,explicit_hydrogen=explicit_hydrogen,threshold_bond=threshold_bond, unique_atm_symbols=unique_atm_symbols) if check_inbetween_bonds_pdf \
                            else get_actual_bnds(candidaten_bnds,atm_bnd_pos, mol_bnd_symb,threshold_bond=threshold_bond)
        number_of_bonds.append(len(actual_bnd))
        bnd2count, indices_ = reorder_inds_get_bnd_numbers(unique_bnds, actual_bnd)
        actual_bnd = np.array(actual_bnd)
        bond_positions = [(atm_bnd_pos[bnd[0]] +atm_bnd_pos[bnd[1]])/2 for bnd in actual_bnd[indices_]]
        all_actual_bonds.append(list(actual_bnd[indices_]))
        all_actual_bnd_positions.extend(bond_positions)
        indices.extend([v for k,v in bnd2count.items() if v != 0])
        corresp_field.extend([field[ind][start_bnd+k] for k,v in bnd2count.items() if v != 0])

    corresp_weights = rmv_gmm_low_bnd_weights(all_actual_bnd_positions, all_actual_bonds, indices, field, x_flat=x,y_flat=y,z_flat=z,corresp_field=corresp_field)
    # corresp_weights[len(all_actual_bonds[0]) + len(all_actual_bonds[1]) + len(all_actual_bonds[2]):len(all_actual_bonds[0]) + len(all_actual_bonds[1]) + len(all_actual_bonds[2])+len(all_actual_bonds[3])]
    
    # * filter only non empty bond lists
    non_empty_inds = [i for i in range(len(all_actual_bonds)) if len(all_actual_bonds[i]) != 0]
    non_empty_bond_list = [all_actual_bonds[i] for i in non_empty_inds]
    all_new_bonds_nonempty, removed_bnds_mask, weights_per_molecule_non_empty, rmvd_bonds_nonempty = remove_low_weight_bonds(corresp_weights, non_empty_bond_list, indices, pi_threshold)
    all_new_bonds = [[] for _ in range(len(all_actual_bonds))]
    all_rmvd_bonds = [[] for _ in range(len(all_actual_bonds))]
    weights_per_molecule = [{1:[], 2:[], 3:[], 4:[]} for _ in range(len(all_actual_bonds))]
    for i, ind in enumerate(non_empty_inds):
        all_new_bonds[ind] = all_new_bonds_nonempty[i]
        weights_per_molecule[ind] = weights_per_molecule_non_empty[i]
        all_rmvd_bonds[ind] = rmvd_bonds_nonempty[i]

    return all_new_bonds, removed_bnds_mask, all_candidate_bnds, weights_per_molecule, all_rmvd_bonds


def check_smiles_validity_wbond_chanls(atoms_and_bond_positions,molecules_atom_or_bond_symbols,true_smiles,
                                       check_inbetween_bonds_pdf=False, field=None,x=None,y=None,z=None,
                                       normalize01=False,sanity_checking=False, return_atm_pos_bdns=None,
                                       explicit_aromatic=False, explicit_hydrogen=False,return_smiles=False,
                                       optimize_bnd_gmm_weights=False,discard_fields=False,threshold_bond=0.75,
                                       pi_threshold=0.5, atm_symbs=None):
    
    if atm_symbs is not None:
        unique_atm_symbols =atm_symbs
    else:
        unique_atm_symbols = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]
    total_equal_or_valid=0
    total=0
    generated_smiles = []
    all_bonds_and_atoms = []
    fields_info = []
    removed_bonds_after_optimization=0
    corresp_weights = [None]
    # for atm_bnd in atoms_and_bond_positions:
    if optimize_bnd_gmm_weights:  # for parallelization purposes, gather all bonds and give them to a GMMbatchModel
        optimized_bnd, removed_bnds_mask, all_candidaten_bnds, corresp_weights, all_rmvd_bonds = gather_and_correct_bonds(atoms_and_bond_positions,molecules_atom_or_bond_symbols, 
                                    field, normalize01, explicit_hydrogen,check_inbetween_bonds_pdf,x,y,z, explicit_aromatic=explicit_aromatic, threshold_bond=threshold_bond, pi_threshold=pi_threshold, unique_atm_symbols=unique_atm_symbols)
        removed_bonds_after_optimization = sum([sum(rm) for rm in removed_bnds_mask])
    else:
        all_rmvd_bonds = [[]] * len(atoms_and_bond_positions)


    for ind,(atm_bnd_pos, mol_bnd_symb) in enumerate(zip(atoms_and_bond_positions,molecules_atom_or_bond_symbols)):
        
        if not optimize_bnd_gmm_weights:
            candidaten_bnds,new2old_index = get_potential_atm_pair_positions(atm_bnd_pos, mol_bnd_symb,explicit_hydrogen)
            actual_bnd = get_actual_bonds_through_pdf(candidaten_bnds, field[ind],x,y,z,var=0.05,normalize01=normalize01,explicit_hydrogen=explicit_hydrogen, threshold_bond=threshold_bond) #if check_inbetween_bonds_pdf else get_actual_bnds(candidaten_bnds,atm_bnd_pos, mol_bnd_symb)
        else:
            actual_bnd = optimized_bnd[ind]
            candidaten_bnds = all_candidaten_bnds[ind]
        

        mol = Chem.RWMol()
        if sanity_checking and "+" in true_smiles[ind]: continue # if we're sanity checking, drop ionized molecules
        for atom in mol_bnd_symb:
            if atom not in unique_atm_symbols: continue # if this is a bond symbol skip it
            a = Chem.Atom(atom)
            mol.AddAtom(a)
    
        for bnd in actual_bnd:
            try: mol.AddBond(int(bnd[0]),int(bnd[1]),bond_dict[int(bnd[2])])
            except: print("BOND ALREADY EXISTS??")
        smiles = mol2smiles(mol)
        all_bonds_and_atoms.append([mol_bnd_symb, actual_bnd, smiles])
        total += 1
        
        # if '3' not in clean_smiles(true_smiles[ind]): total+=1
        # elif '3' in clean_smiles(true_smiles[ind]): continue

        fields_info.append([field[ind] if not discard_fields else None, candidaten_bnds,actual_bnd,mol_bnd_symb,atm_bnd_pos])
        if exists(smiles): total_equal_or_valid+=1; generated_smiles.append(smiles)
        elif not exists(smiles) and return_smiles: generated_smiles.append(None) # return also faield smiles for later comparisons (e.g. compare to correct ones)
        if not sanity_checking: continue # if we're not sanity checking, we don't have gt to compare to, so just check validity

        if smiles is not None: Chem.MolToSmiles(mol,kekuleSmiles=True)
        elif smiles is None and true_smiles[ind] is not None: total_equal_or_valid-=1; continue #plot_one_mol(atm_bnd_pos, actual_bnd, plot_bnd=0, threshold=0.15, field=field[ind],x_grid=x,y_grid=y,z_grid=z,atm_symb=mol_bnd_symb, title=true_smiles[ind]); breakpoint(); continue


        if clean_smiles(Chem.CanonSmiles(true_smiles[ind])) != clean_smiles(Chem.CanonSmiles(smiles)):
            total_equal_or_valid-=1; plot_one_mol(atm_bnd_pos, actual_bnd, plot_bnd=0, threshold=0.15, field=field[ind],x_grid=x,y_grid=y,z_grid=z,atm_symb=mol_bnd_symb, title=true_smiles[ind], unique_atm_symbols=unique_atm_symbols); breakpoint()
        # plot_one_mol(atm_bnd_pos, actual_bnd, plot_bnd=4, threshold=0.75, field=field[ind],x_grid=x,y_grid=y,z_grid=z,atm_symb=mol_bnd_symb, title=true_smiles[ind])
        # else:
        #     plot_one_mol(atm_bnd_pos, actual_bnd, plot_bnd=5, threshold=0.95, field=field[ind],x_grid=x,y_grid=y,z_grid=z,atm_symb=mol_bnd_symb); breakpoint()
        
    return total_equal_or_valid,total,generated_smiles, all_bonds_and_atoms, fields_info, removed_bonds_after_optimization, corresp_weights, all_rmvd_bonds

# def check_smiles_validity(atom_position_list, atom_types,coords,all_atom_lists, explicit_hydrogen=False):
# def check_smiles_validity(atom_position_list, atom_types, explicit_hydrogen=False):

#     same_smiles = 0
#     total_smiles = 0
#     unique_atm_symbols = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]
#     atom_position_index = 0
#     all_mol_atms, all_mol_bonds, all_mol_smiles = [],[],[]
#     for index, atm_type in enumerate(atom_types):
#         all_atm_pos = []
#         all_atm_symbols = []
#         for ua_t in unique_atm_symbols:
#             no_atoms = sum(np.array(atm_type) == ua_t)
#             current_atm_inds = np.argwhere(np.array(atm_type) == ua_t).flatten()
#             current_positions = atom_position_list[index][current_atm_inds]
#             if no_atoms == 0: continue
#             atm_pos = current_positions.detach().cpu().numpy() if type(current_positions) == torch.Tensor  else current_positions
#             atom_position_index += no_atoms
#             all_atm_pos.extend(atm_pos)
#             all_atm_symbols.extend([ua_t] * no_atoms)
#         all_atm_pos = torch.tensor(all_atm_pos)

#         mol_generated = build_molecule(all_atm_pos, all_atm_symbols, unique_atm_symbols)
#         mol_generated, all_bonds = build_molecule_return_bonds(all_atm_pos, all_atm_symbols, unique_atm_symbols)
#         smiles_generated = mol2smiles(mol_generated)
#         all_mol_atms.append(all_atm_symbols)
#         all_mol_bonds.append(all_bonds)
#         all_mol_smiles.append(smiles_generated)

#         # * previously used just for sanity checking
#         # mol_true = build_molecule(torch.tensor(coords[index]),all_atom_lists[index],unique_atm_symbols)
#         # smiles_true = mol2smiles(mol_true)
#         # # sometimes even the QM9 dataset doesn't contain valid smiles :)

#         # print(smiles_true, smiles_generated)
#         # if smiles_true is not None and smiles_generated is None: total_smiles +=1; continue
#         # if smiles_true is None or smiles_generated is None: continue

#         # if clean_smiles(smiles_true) == clean_smiles(smiles_generated):
#         #     same_smiles += 1
#         # total_smiles += 1
#     # return same_smiles, total_smiles
#     return all_mol_atms,all_mol_bonds,all_mol_smiles

def optimize_atm_positions(atom_position_list, atom_types, explicit_hydrogen=False,field=None,x_grid=None,y_grid=None,z_grid=None,std=0.05,first_bnd_chn=None,atm_symbs=None):
    all_updated_positions = []
    if atm_symbs is not None: first_bnd_chn = len(atm_symbs) - 1
    if first_bnd_chn is not None:
        field = [field[i][:first_bnd_chn+1] for i in range(len(field))]
    no_atms = field[0].shape[0]

    N_list =  extract_N_inds_from_atms(atom_types ,explicit_hydrogen=explicit_hydrogen, unique_atm_symbols=atm_symbs)
    updated_positions = extract_positions_batch(field, x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), N_list, std=std, initial_guess=atom_position_list)

    # all mols are optimize together, so separate them based on no of atms
    total_atm_per_mol = [sum(N_list[no_atms*i:no_atms*(i+1)]) for i in range(len(N_list)//no_atms) ]
    atom_pos_inds = [sum(total_atm_per_mol[:i+1]) for i in range(len(total_atm_per_mol))]
    atom_pos_inds.insert(0,0)
    for i in range(len(atom_pos_inds)-1): all_updated_positions.append(updated_positions[atom_pos_inds[i]:atom_pos_inds[i+1]])

    for i in range(len(atom_position_list)): atom_position_list[i]=all_updated_positions[i].detach().cpu().numpy()
    return atom_position_list



def check_smiles_validity(atom_position_list, atom_types, explicit_hydrogen=False,field=None,x_grid=None,y_grid=None,z_grid=None,std=0.05, optimize_positions=True,true_smiles=[""]):

    same_smiles = 0
    total_smiles = 0
    unique_atm_symbols = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]
    atom_position_index = 0
    all_mol_atms, all_mol_bonds, all_mol_smiles = [],[],[]

    all_updated_positions = []
    if field is not None and field[0] is not None:
        no_atms = field[0].shape[0]
    else:
        no_atms = len(unique_atm_symbols)
    
    if optimize_positions:
        N_list =  extract_N_inds_from_atms(atom_types ,explicit_hydrogen=explicit_hydrogen)
        updated_positions = extract_positions_batch(field, x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), N_list, std=std, initial_guess=atom_position_list)

        # all mols are optimize together, so separate them based on no of atms
        total_atm_per_mol = [sum(N_list[no_atms*i:no_atms*(i+1)]) for i in range(len(N_list)//no_atms) ]
        atom_pos_inds = [sum(total_atm_per_mol[:i+1]) for i in range(len(total_atm_per_mol))]
        atom_pos_inds.insert(0,0)
        for i in range(len(atom_pos_inds)-1): all_updated_positions.append(updated_positions[atom_pos_inds[i]:atom_pos_inds[i+1]])
        for i in range(len(atom_position_list)): atom_position_list[i]=all_updated_positions[i]

    bnd_atm_smil, corresp_weights, fields_info = [], [{1:[], 2:[], 3:[], 4:[]} for _ in range(len(atom_position_list))], []

    for index, atm_type in enumerate(atom_types):



        all_atm_pos = []
        all_atm_symbols = []
        for ua_t in unique_atm_symbols:
            no_atoms = sum(np.array(atm_type) == ua_t)
            current_atm_inds = np.argwhere(np.array(atm_type) == ua_t).flatten()
            current_positions = atom_position_list[index][current_atm_inds]
            if no_atoms == 0: continue
            atm_pos = current_positions.detach().cpu().numpy() if type(current_positions) == torch.Tensor  else current_positions
            atom_position_index += no_atoms
            all_atm_pos.extend(atm_pos)
            all_atm_symbols.extend([ua_t] * no_atoms)
        all_atm_pos = torch.tensor(all_atm_pos)

        mol_generated = build_molecule(all_atm_pos, all_atm_symbols, unique_atm_symbols)
        mol_generated, all_bonds = build_molecule_return_bonds(all_atm_pos, all_atm_symbols, unique_atm_symbols)
        smiles_generated = mol2smiles(mol_generated)
        all_mol_atms.append(all_atm_symbols)
        all_mol_bonds.append(all_bonds)
        all_mol_smiles.append(smiles_generated)
        bnd_atm_smil.append([atm_type, all_bonds, smiles_generated])
        all_atm_pos = all_atm_pos.detach().cpu().numpy()
        
        # fields_info [field, [[bnd_a1, bnd_a2, pos],...],  [[bnd_a1, bnd_a2, bnd_t], ...], [atm_list], [atm_pos] ]
        bnd_pos = [ [b[0], b[1], (all_atm_pos[b[0]] + all_atm_pos[b[1]])/2] for b in all_bonds]
        fields_info.append([field[index], bnd_pos, all_bonds, atm_type, all_atm_pos])
        # * previously used just for sanity checking
        # mol_true = build_molecule(torch.tensor(coords[index]),all_atom_lists[index],unique_atm_symbols)
        # smiles_true = mol2smiles(mol_true)
        # # sometimes even the QM9 dataset doesn't contain valid smiles :)

        # print(smiles_true, smiles_generated)
        # if smiles_true is not None and smiles_generated is None: total_smiles +=1; continue
        # if smiles_true is None or smiles_generated is None: continue

        # if clean_smiles(smiles_true) == clean_smiles(smiles_generated):
        #     same_smiles += 1
        # total_smiles += 1
    # return same_smiles, total_smiles
    valid, tot = len([s for s in all_mol_smiles if s is not None]), len(all_mol_smiles)


    # when sanity checking graph extraction on true data
    not_same = 0
    if true_smiles[0]:
        for t_s, g_s in zip(true_smiles, all_mol_smiles):
    

            if t_s is None or g_s is None: continue # if t_s is None, it's already accounted as not valid

            if clean_smiles(Chem.CanonSmiles(t_s)) != clean_smiles(Chem.CanonSmiles(g_s)):
                not_same +=1
    valid -= not_same

    #! check if bnd_atm_smil coincides in all_bonds w other
    # return valid, tot, all_mol_smiles, bnd_atm_smil
    return valid, tot, all_mol_smiles, bnd_atm_smil,fields_info, 0,corresp_weights


def get_smiles_w_same_order(orig_smile):
    u_atms = ["C","O","N","F"]
    



    coords = []
    mol = Chem.MolFromSmiles(orig_smile)
    mol = Chem.AddHs(mol)
    check = AllChem.EmbedMolecule(mol,useRandomCoords=True)
    if check==-1: print("CHECK -1???")
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        positions = conf.GetAtomPosition(i)
        coords.append([positions.x,positions.y,positions.z])
    coords = np.array([coords]).reshape(-1,3)
    coords_aligned = align(coords)
    conf,new_coords,atom_symbols = update_mol_positions(mol,conf,coords_aligned)
    mol_ = read_smiles(orig_smile)
    edge_attr = nx.get_edge_attributes(mol_, name = "order")


    new_mol = Chem.RWMol()


    old2new_ind = {}
    new_ind=0
    for ua in u_atms:
        for ind,atom in enumerate(atom_symbols):
            if atom != ua: continue
            a = Chem.Atom(atom)
            new_mol.AddAtom(a)
            old2new_ind[ind]=new_ind
            new_ind+=1

    for bnd in edge_attr:
        new_mol.AddBond(old2new_ind[bnd[0]],old2new_ind[bnd[1]],bond_dict[edge_attr[bnd]])

    new_smiles = mol2smiles(mol)


def check_pipelin(sample):
    sample = sample * 0.2
    sample[0,1,1,1] = 1
    sample[1,1,1,2] = 1
    sample[2,1,3,1] = 1
    sample[3,1,3,1] = 1
    return sample


def fit_pred_field_sep_chn_batch(pred,cutoff,x,y,z,num_channels=-1,field_atoms=-1,smiles_list=[""],return_atoms=False, 
                                return_smiles=False, atom_and_bond_numbers=None, std=0.05,coords=None, new_resolutions=[0.15,0.15,0.15],
                                all_atom_lists=None, batch_size=10, true_smiles=None, noise_std=0.0,
                                refine=False,determine_bonds=False,normalize01=False, noise_type='normal', return_atm_pos_bdns=False,
                                explicit_aromatic=False,explicit_hydrogen=False,optimize_bnd_gmm_weights=False,
                                discard_fields=False, threshold_bond=0.75,threshold_atm=0.75, pi_threshold=0.5, 
                                atm_symb_bnd_by_channel=None, atm_dist_based=False, optimize_atom_positions=False):
    """

        batch size is specified in the number of atoms
        determine_bonds: if true, then the bonds are determined from the predicted field; else, there's only a check
                         for pdf values between each two atom positions (which are within certain ranges)

        # return_atm_pos_bdns returns atom positions and bonds even if they are not valid; 

        optimize_bnd_gmm_weights will fit the GMM weights (pi) to the field values
    """
    pred_field = pred.cpu().numpy()
    if atm_symb_bnd_by_channel is None:
        if explicit_hydrogen: atms = ["C","O","N","F","H",1,2,3] if not explicit_aromatic else ["C","O","N","F","H",1,2,3,1.5]
        else: atms = ["C","O","N","F",1,2,3] if not explicit_aromatic else ["C","O","N","F",1,2,3,1.5]
        first_bnd_chn = 4 if explicit_hydrogen else 3
        atm_symbs = ["C","O","N","F","H"] if explicit_hydrogen else ["C","O","N","F"]

    else:
        atms = list(atm_symb_bnd_by_channel)
        # below, ugly fix: atm_symb_bnd_by_channel may contain atom only atm symbols, or atm and bnds. Make it consistent
        if type(atms[-1]) != int: atms.extend([1,2,3,4] if explicit_aromatic else [1,2,3])
        atm_symbs = atms[:-4] if explicit_aromatic else atms[:-3]
        first_bnd_chn = len(atm_symbs)-1

    initial_guesses = []
    num_atoms_list = []
    samples = []

    molecules_atom_or_bond_symbols = []
    molecules_atom_positions = []
    true_smiles_ = []
    fields_ = []
    total_equal_or_valid,total = 0,0
    sanity_checking = exists(atom_and_bond_numbers)
    generated_smiles = []
    total_different_atm_no = 0
    all_generated_atm_bnd_smils = []
    all_fields = []
    total_rmvd_opt = 0
    all_cores_weights = []
    all_rmvd_bnds = []
    for idx,sample in enumerate(pred_field):
        molecule_atoms_or_bonds = []
        sample = torch.tensor(sample)
        sample = sample + torch.randn_like(sample) * noise_std if noise_type=='normal' else sample + torch.rand_like(sample) * noise_std
        different_atm_no = False

        for chan_idx, channel in enumerate(sample):
            if chan_idx > first_bnd_chn and determine_bonds == False: continue
            initial_guess = get_sphere_centers_deterministic(torch.tensor(channel), x,y,z,normalize01,set_threshold=threshold_atm)

            if sanity_checking and not explicit_hydrogen: # this part is for sanity checks with ground truth
                # if idx >= len(atom_and_bond_numbers) or chan_idx >= len(atom_and_bond_numbers[idx]): 
                if len(initial_guess) != atom_and_bond_numbers[idx][chan_idx]: 
                    print("different initial guess no of atms and true no of atms")
                    different_atm_no +=1
            num_atoms_or_bonds = len(initial_guess)
            if num_atoms_or_bonds == 0: continue
            samples.append(channel)
            initial_guesses.append(initial_guess)
            num_atoms_list.append(num_atoms_or_bonds)
            molecule_atoms_or_bonds.extend([atms[chan_idx]] * len(initial_guess))
        total_different_atm_no =total_different_atm_no+1 if different_atm_no else total_different_atm_no
        if sanity_checking: true_smiles_.append(true_smiles[idx])
        else: true_smiles_.append("")
        if len(initial_guesses) == 0:
            total+=1
        else:
            fields_.append(sample.detach().cpu().numpy())
            molecules_atom_positions.append(np.concatenate(initial_guesses))
            molecules_atom_or_bond_symbols.append(molecule_atoms_or_bonds)
            initial_guesses = []


        if ((idx % batch_size == 0 and idx != 0) or idx == len(pred_field)-1) and atm_dist_based:
            # ! right now, since atms are extracted in the same CONFH order, the below works, but be careful
            valid,tot,gen_smiles, \
                bnd_atm_smil,fields_info, removed_bonds_after_optimization, corresp_weights = check_smiles_validity(molecules_atom_positions,molecules_atom_or_bond_symbols,explicit_hydrogen,field=fields_,x_grid=x,y_grid=y,z_grid=z, std=std, true_smiles=true_smiles_)
            all_cores_weights.extend(corresp_weights)
            total_rmvd_opt += removed_bonds_after_optimization
            if return_atm_pos_bdns: all_fields.extend(fields_info)
            all_generated_atm_bnd_smils.extend(bnd_atm_smil)
            generated_smiles.extend(gen_smiles)
            total_equal_or_valid +=valid
            total += tot
            molecules_atom_positions = []
            molecules_atom_or_bond_symbols = []
            true_smiles_ = []
            fields_ = []

        elif ((idx % batch_size == 0 and idx != 0) or idx == len(pred_field)-1) and not atm_dist_based:
            # TODO need to refine the initial guesses first if refine=True
            if optimize_atom_positions: molecules_atom_positions = optimize_atm_positions(molecules_atom_positions,molecules_atom_or_bond_symbols,explicit_hydrogen,field=fields_,x_grid=x,y_grid=y,z_grid=z, std=std, first_bnd_chn=first_bnd_chn, atm_symbs=atm_symbs)
            # it's just the batch that's weird somehow
            # check_smiles_validity(molecules_atom_positions,molecules_atom_or_bond_symbols, explicit_hydrogen)
            valid,tot,gen_smiles, \
                bnd_atm_smil,fields_info, removed_bonds_after_optimization, \
                    corresp_weights, rmvd_bnds = check_smiles_validity_wbond_chanls(molecules_atom_positions,
                                                        molecules_atom_or_bond_symbols,
                                                        true_smiles_,check_inbetween_bonds_pdf=not determine_bonds,
                                                        field=fields_,x=x,y=y,z=z,normalize01=normalize01,sanity_checking=sanity_checking,
                                                        return_atm_pos_bdns=return_atm_pos_bdns, explicit_aromatic=explicit_aromatic,
                                                        explicit_hydrogen=explicit_hydrogen,return_smiles=return_smiles, 
                                                        optimize_bnd_gmm_weights=optimize_bnd_gmm_weights, discard_fields=discard_fields,
                                                        threshold_bond=threshold_bond, pi_threshold=pi_threshold, atm_symbs=atm_symbs)
            all_rmvd_bnds.extend(rmvd_bnds)
            all_cores_weights.extend(corresp_weights)
            total_rmvd_opt += removed_bonds_after_optimization
            if return_atm_pos_bdns: all_fields.extend(fields_info)
            all_generated_atm_bnd_smils.extend(bnd_atm_smil)
            generated_smiles.extend(gen_smiles)
            total_equal_or_valid +=valid
            total += tot
            molecules_atom_positions = []
            molecules_atom_or_bond_symbols = []
            true_smiles_ = []
            fields_ = []
    if total_equal_or_valid > 0:
        unq = compute_uniqueness(generated_smiles)
        nov = compute_novelty(generated_smiles,smiles_list)
    else:
        unq =0 
        nov = 0
    if optimize_bnd_gmm_weights: print("Total removed bonds after optimization: {}".format(total_rmvd_opt))
    # if total_different_atm_no != 0:
    #     logging.info("For the std coming below this line, same-atm no predictions had validity {}".format(total_equal_or_valid/(total-total_different_atm_no)))
    if return_smiles and return_atm_pos_bdns: return total_equal_or_valid/total,  unq, nov, all_generated_atm_bnd_smils, all_fields, generated_smiles, all_cores_weights, all_rmvd_bnds

    elif return_smiles: return total_equal_or_valid/total,  unq, nov, generated_smiles, all_cores_weights, all_rmvd_bnds
    
    elif return_atm_pos_bdns: return total_equal_or_valid/total,  unq, nov, all_generated_atm_bnd_smils, all_fields, all_cores_weights, all_rmvd_bnds
    
    return total_equal_or_valid/total,  unq, nov, all_cores_weights
        # if idx % batch_size == 0 and idx > 0:
            # for atm in 


    # the part below should probably be used only if I want to refine, which is probably not the case
    # if idx % batch_size == 0 and idx > 0:
    #     mean_positions = extract_positions_batch(samples, x,y,z, num_atoms_list, std, new_resolutions=new_resolutions, initial_guess=initial_guesses)


    #     molecules_atom_positions.extend(mean_positions)
    #     check_smiles_validity(molecules_atom_positions,molecules_atom_symbols,coords,all_atom_lists)
    #     breakpoint()


def fit_pred_field_curve_batch(pred,cutoff,x,y,z,num_channels,field_atoms,smiles_list,return_atoms=False, 
                                return_smiles=False, true_atm_numbers=None, std=0.05,coords=None, new_resolutions=[0.15,0.15,0.15],
                                all_atom_lists=None, batch_size=10, true_smiles=None, noise_std=0.0,noise_type='normal'):
    """

        batch size is specified in the number of atoms
    """
    pred_field = pred.cpu().numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    total_smiles, same_smiles = 0,0
    atms = ["C","O","N","F"]
    initial_guesses = []
    num_atoms_list = []
    samples = []

    molecules_atom_symbols = []
    molecules_atom_positions = []
    true_coords = []
    all_atm_lists = []

    for idx,sample in enumerate(pred_field):
        molecule_atoms = []
        current_samples = []
        current_initial_guesses = []
        current_num_atoms = []
        different_atm_numb = False
        for chan_idx, channel in enumerate(sample):
            # initial_guess = coords[idx][np.array(all_atom_lists[idx]) == atms[chan_idx]] if exists(all_atom_lists) else None
            # num_atoms =  true_atm_numbers[idx][chan_idx] if exists(true_atm_numbers) else extract_number_of_atoms_and_initial_estimate(pred_field,cutoff=0.8)

            channel = torch.tensor(channel)
            # channel = channel + torch.randn_like(channel) * noise_std
            channel = channel + torch.randn_like(channel) * noise_std if noise_type=='normal' else channel + torch.rand_like(channel) * noise_std

            initial_guess = get_sphere_centers_deterministic(torch.tensor(channel), x,y,z,normalize01=False)

            if len(initial_guess) != true_atm_numbers[idx][chan_idx]: different_atm_numb=True; continue

            num_atoms = len(initial_guess)
            if num_atoms == 0: continue

            current_samples.append(channel)
            current_initial_guesses.append(initial_guess)
            current_num_atoms.append(num_atoms)
            molecule_atoms.extend([atms[chan_idx]] * len(initial_guess))

            # samples.append(channel)
            # initial_guesses.append(initial_guess)
            # num_atoms_list.append(num_atoms)

        if different_atm_numb: continue
        initial_guesses.extend(current_initial_guesses)
        num_atoms_list.extend(current_num_atoms)
        samples.extend(current_samples)
        molecules_atom_symbols.append(molecule_atoms)

        all_atm_lists.append(all_atom_lists[idx])
        true_coords.append(coords[idx])



        if (idx % batch_size == 0 and idx > 0) or idx == len(pred_field)-1:
            mean_positions = extract_positions_batch(samples, x,y,z, num_atoms_list, std, new_resolutions=new_resolutions, initial_guess=initial_guesses)
            molecules_atom_positions.extend(mean_positions)
            same_sm, tot_sm = check_smiles_validity(molecules_atom_positions,molecules_atom_symbols,true_coords,all_atm_lists)
            molecules_atom_positions = []
            molecules_atom_symbols = []
            all_atm_lists = []
            true_coords = []
            initial_guesses = []
            samples = []
            num_atoms_list = []
            same_smiles += same_sm
            total_smiles += tot_sm

    if len(initial_guesses):
            mean_positions = extract_positions_batch(samples, x,y,z, num_atoms_list, std, new_resolutions=new_resolutions, initial_guess=initial_guesses)
            molecules_atom_positions.extend(mean_positions)
            same_sm, tot_sm = check_smiles_validity(molecules_atom_positions,molecules_atom_symbols,true_coords,all_atm_lists)
            molecules_atom_positions = []
            molecules_atom_symbols = []
            all_atm_lists = []
            true_coords = []
            initial_guesses = []
            samples = []
            num_atoms_list = []
            same_smiles += same_sm
            total_smiles += tot_sm
    print(same_smiles,total_smiles)
    return same_smiles/total_smiles


def fit_pred_field_curve(pred,cutoff,x,y,z,num_channels,field_atoms,smiles_list,return_atoms=False, 
                         return_smiles=False, true_atm_numbers=None, std=0.05,coords=None, new_resolutions=[0.15,0.15,0.15],
                         all_atom_lists=None):
    pred_field = pred.cpu().numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_samples = len(pred)
    Atom_position = []
    Atoms= []
    valid = 0
    valid_smiles = []
    atms = ["C","O","N","F"]
    for idx,sample in enumerate(pred_field):
        for chan_idx, channel in enumerate(sample):
            # initial_guess = coords[idx][np.array(all_atom_lists[idx]) == atms[chan_idx]] if exists(all_atom_lists) else None
            # num_atoms =  true_atm_numbers[idx][chan_idx] if exists(true_atm_numbers) else extract_number_of_atoms_and_initial_estimate(pred_field,cutoff=0.8)

            initial_guess = get_sphere_centers_deterministic(torch.tensor(channel), x,y,z)
            num_atoms = len(initial_guess)
            if num_atoms == 0: continue
            # breakpoint()


            mean_positions = extract_positions(channel, x,y,z, num_atoms, std, new_resolutions=new_resolutions, initial_guess=initial_guess)

            if coords is not None:
                if all_atom_lists is not None:
                    print(mean_positions)
                    print(coords[idx][np.array(all_atom_lists[idx]) == atms[chan_idx]])
                else:
                    distances = torch.cdist(torch.tensor(coords[idx],device=device,dtype=torch.float32), mean_positions)
                    print(torch.min(distances,dim=1))

            breakpoint()

    


def fit_pred_field(pred,cutoff,x,y,z,num_channels,field_atoms,smiles_list,return_atoms=False, return_smiles=False,
                   cutoff_steps=10):
  pred_field = pred.cpu().numpy()
  num_samples = len(pred)
  Atom_position = []
  Atoms= []
  valid = 0
  valid_smiles = []
  for idx,sample in enumerate(pred_field):
    temp = []
    temp_atoms = []
    for channel in range(num_channels):
        BIC= []
        means = []

        pred_dens = sample[channel].flatten()
        # if channel does not contain any atoms (speheres with high enough values), skip
        if sum(pred_dens > 0.8) == 0:
            continue
        pred_dens = pred_dens/max(pred_dens)
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        if cutoff_steps == 1:
            x_flat = x_flat[pred_dens >= cutoff]
            y_flat = y_flat[pred_dens >= cutoff]
            z_flat = z_flat[pred_dens >= cutoff]
        else:
            x_flat, y_flat, z_flat = extract_positions_mulitple_cutoffs(pred_dens, cutoff_steps, cutoff,x_flat,y_flat,z_flat)

        if len(x_flat) == 0: continue
        pos_arr = np.concatenate([x_flat.reshape(-1,1),y_flat.reshape(-1,1),z_flat.reshape(-1,1)],axis=1)
        # if channel does not contain any atoms (speheres with high enough values), skip
        for cl in range(1,10):
            gm = GaussianMixture(n_components=cl, random_state=0).fit(pos_arr)
            means.append(gm.means_)
            bic = gm.bic(pos_arr)
            BIC.append(bic)

        #print(BIC)
        #print(np.argmin(BIC)+1," Atoms found in the ",field_atoms[channel]," field")
        for cart in means[np.argmin(BIC)].tolist():
            temp.append(cart)
            temp_atoms.append(field_atoms[channel])
        
    Atom_position.append(temp)
    Atoms.append(temp_atoms)

    assert len(Atoms[idx]) == len(Atom_position[idx]), "Number of atoms and the positions are not consistent"
    valid_score,smiles = check_validity(torch.tensor([Atom_position[idx]]).view(-1,3),Atoms[idx],field_atoms)
    if valid_score>0:
       valid = valid + valid_score
       valid_smiles.append(smiles)
    #    print(smiles, smiles_list[idx], "\n")

  val = valid/num_samples
  if val>0:
    unq = compute_uniqueness(valid_smiles)
    nov = compute_novelty(valid_smiles,smiles_list)
  else:
      unq = 0
      nov = 0
  if return_atoms and return_smiles:
      return val,unq,nov,Atom_position, Atoms, valid_smiles
  elif return_atoms:
      return val, unq, nov, Atom_position, Atoms
  elif return_smiles:
      return val, unq, nov, valid_smiles
  else:
      return val,unq,nov
