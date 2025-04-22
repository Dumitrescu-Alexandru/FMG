from scipy.spatial.transform import Rotation
import pickle
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import copy
import torch
# from denoising_diffusion_pytorch.denoising_diffusion_pytorch import fit_pred_field, build_molecule, mol2smiles, fit_pred_field_curve, fit_pred_field_curve_batch,fit_pred_field_sep_chn_batch
from test_utils import fit_pred_field, build_molecule, mol2smiles, fit_pred_field_curve, fit_pred_field_curve_batch,fit_pred_field_sep_chn_batch
import os
from utils import get_mol_reps,get_stub_mol_reps, get_grid_limits,get_channel_mol_reps, get_mol_reps_w_bonds, custom_collate_agnostic_atm_types, setup_data_files, retrieve_smiles, collate_fn_compact_expl_ah_clsfreeguid_debugds,collate_fn_compact_expl_h_clsfreeguid_debugds,collate_fn_compact_expl_ah_debugds,collate_fn_compact_expl_h_debugds, create_gaussian_batch_pdf_values, AtomPositionSepBondsDatasetCompact, collate_fn_compact_expl_h_clsfreeguid_debugds_geom, collate_fn_general,collate_fn_general_debug
import numpy as np
final_data = []
import matplotlib.pyplot as plt
# def check_stub_data_molecule_retrieval(true_smiles, true_atoms, true_atoms_positions,  x_grid,y_grid,z_grid):
from utils import AtomPositionSepBondsDataset, AtomPositionDataset, get_data_folder

import argparse
parser = argparse.ArgumentParser('FieldGen')
parser.add_argument('--method', type=str, default="separate_channels")
parser.add_argument('--noise_std', type=float, default=0.01)
parser.add_argument('--noise_type', type=str, default='uniform')
parser.add_argument('--run_name', type=str, default='test_separate')
parser.add_argument('--resolution', type=float, default=0.33)
parser.add_argument("--ignore_aromatic", default=False, action="store_true")
parser.add_argument("--explicit_hydrogen", default=False, action="store_true")
parser.add_argument("--data_file", default="", type=str)
parser.add_argument("--agnostic_atm_types", default=False, action="store_true")
parser.add_argument("--cond_variable", default=[])
parser.add_argument("--discrete_conditioning", default=False, action="store_true")
parser.add_argument("--remove_bonds", default=False, action="store_true")
parser.add_argument("--arom_cycle_channel", default=False, action="store_true")
parser.add_argument("--optimize_atom_positions", default=False, action="store_true")

parser.add_argument("--data", default="QM9EDM", type=str)
parser.add_argument("--use_original_atm_pos", default=False, action="store_true")
parser.add_argument("--no_limits", default=False, action="store_true")
parser.add_argument("--force_create_data", default=False, action="store_true")
parser.add_argument('--std_atoms', type=float, default=0.05)
parser.add_argument('--not_use_pca', default=False, action='store_true')
parser.add_argument('--rescale', default=False, action="store_true")
parser.add_argument('--explicit_aromatic', default=False, action="store_true")
parser.add_argument('--no_sep_bnd_chn', default=False, action="store_true")
parser.add_argument('--debug_ds', default=True, action='store_true', help='return coordinates and bond type from dataset in order to visualize/debug')
parser.add_argument('--recheck_generated_flds', default=False, action='store_true')
parser.add_argument('--augment_rotations', default=False, action='store_true')
parser.add_argument('--subsample_points', default=-1, type=int)
parser.add_argument('--mixed_prec', default=False, action='store_true')
parser.add_argument('--compact_batch', default=False, action='store_true')
parser.add_argument("--backward_mdl_compat", default=False, action='store_true', help='old models had thresholding only on atoms; create the same for compact_batch ')
parser.add_argument("--remove_thresholding_pdf_vals", default=False, action='store_true')
parser.add_argument("--class_conditioning", default=False, action='store_true')
parser.add_argument("--optimize_bnd_gmm_weights", default=False, action='store_true')
parser.add_argument("--threshold_bond", default=0.75, type=float, help='threshold where bond is considered an actual bond')
parser.add_argument("--threshold_atm", default=0.75, type=float, help='threshold where atm is considered')
parser.add_argument("--rotate_coords", default=False, action='store_true')
parser.add_argument("--pi_threshold", default=0.5, type=float)

parser.add_argument("--center_atm_to_grids", default=False, action='store_true')

parser.add_argument("--subset_confs", default=30, type=int, help="subset number of conformers to use for GEOM dataset")

args = parser.parse_args()
args.ignore_aromatic = not args.explicit_aromatic


import logging
logging.getLogger('some_logger')
logging.basicConfig(filename=args.run_name+".logging", level=logging.INFO, force=True)


def rotate_coords(coords, inds, N_list, classes, bnds, explicit_aromatic, explicit_hydrogen, std_atoms, x_grid, y_grid, z_grid):
    new_coords, new_inds, new_N_list, new_classes, new_bnds = [], [], [], [], []
    removed = 0
    removed_coords = 0
    current_ind = 0
    no_channels = 7 + explicit_aromatic + explicit_hydrogen
    listed_bnds = []
    for _ in range(len(N_list)): listed_bnds.append([])
    limits = [sum([sum(n_list) for n_list in N_list[:i]]) for i in range(len(N_list) + 1)]
    limits.append(limits[-1]+1)
    limits = np.array(limits)
    for bnd in bnds:
        index = np.argwhere(bnd[0] < limits).reshape(-1)[0] -1
        listed_bnds[index].append(bnd)
    listed_bnds = [np.array(lb) for lb in listed_bnds]
    for i in range(len(N_list)):
        coords_ = coords[current_ind:current_ind+sum(N_list[i])]
        r = Rotation.random()
        coords_ = r.apply(coords_)

        lim1 = coords_.min() - std_atoms > x_grid.min() and coords_.min() - std_atoms > y_grid.min() and coords_.min() - std_atoms > z_grid.min()
        lim2 = coords_.max() + std_atoms < x_grid.max() and coords_.max() + std_atoms < y_grid.max() and coords_.max() + std_atoms < z_grid.max()
        if lim1 and lim2:
            new_coords.extend(coords_)
            new_inds.extend(inds[current_ind:current_ind+sum(N_list[i])] - no_channels * removed)
            new_N_list.append(N_list[i])
            new_classes.append(classes[i])


            new_bnds.extend(listed_bnds[i] - removed_coords)
        else: removed+=1; removed_coords += sum(N_list[i])
        current_ind += sum(N_list[i])
    return torch.tensor(np.stack(new_coords), dtype=torch.float32), \
        np.array(new_inds), new_N_list, torch.stack(new_classes), np.stack(new_bnds), removed


def plot_one_mol(atm_pos, actual_bnd, plot_bnd=None, threshold=0.7, field=None, x_grid=None,y_grid=None,z_grid=None, atm_symb=None):

    import matplotlib.pyplot as plt
    
    atm2color = {'C':'black', 'O':'red', 'N':'blue', 'F':'green', 'H':'white'}
    bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [atm2color[a] for a in atm_symb]
    ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300, edgecolor='black')
    print(actual_bnd,atm_pos)
    for bnd_ in actual_bnd:
        bnd_inds = [bnd_[0], bnd_[1]]
        line = atm_pos[bnd_inds]
        if bnd_[2] == 1.5: bnd_[2] = 4
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd_[2]], linewidth=bnd_[2])

        dist = np.linalg.norm(line[0]-line[1])
        bond_position = (line[0]+line[1])/2
        # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
    if plot_bnd is not None:
        breakpoint()
        x = x_grid[field[plot_bnd]>threshold]
        y = y_grid[field[plot_bnd]>threshold]
        z = z_grid[field[plot_bnd]>threshold]
        scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold], cmap='Greys')
        plt.colorbar(scatter)
    # for bnd in candidaten_bnds:
    plt.show()


def check_discretized_positions_generating_correct_smiles(true_smiles, true_atoms, true_atoms_positions,  x_grid,y_grid,z_grid):
    # check if discretized positions are generating correct smiles
    x, y, z = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    positions = np.vstack([x, y, z]).T
    # delete x,y,z variables
    del x, y, z
    for true_atm, true_atm_pos, smile in zip(true_atoms, true_atoms_positions, true_smiles):
        # discretize true_atm_pos
        true_atm_pos = np.array(true_atm_pos)


        closest_true_position_indices = np.argmin(np.linalg.norm(true_atm_pos[:, None, :] - positions, axis=2), axis=1)
        closest_true_positions = positions[closest_true_position_indices]

        threeD_inds = np.unravel_index(closest_true_position_indices, (x_grid.shape[0], x_grid.shape[1], x_grid.shape[2]))


        x_pred,y_pred,z_pred = threeD_inds[0], threeD_inds[1], threeD_inds[2]



        center_x_discretized, center_y_discretized, center_z_discretized = np.stack(x_grid[x_pred,y_pred,z_pred]), \
                                                                            np.stack(y_grid[x_pred,y_pred,z_pred]), \
                                                                            np.stack(z_grid[x_pred,y_pred,z_pred])
        discretized_positions = np.vstack([center_x_discretized, center_y_discretized, center_z_discretized]).T

        channel_atoms = ['C', 'O', 'N', 'F']
        mol,D_true = build_molecule(torch.tensor(true_atm_pos), true_atm, channel_atoms,return_dists=True)
        smiles_true = mol2smiles(mol)

        mol,D_discr = build_molecule(torch.tensor(discretized_positions), true_atm, channel_atoms,return_dists=True)
        smiles_true_discr = mol2smiles(mol)


        if smiles_true_discr is not None:
            no_stereochem_discret_smile = smiles_true_discr.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "")
            original_smile = smile.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "")
            true_pos_smile = smiles_true.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "")

            print_string = ""
            if true_pos_smile == no_stereochem_discret_smile:
                print_string += "true_discr = true_continuous; "
            else:
                print_string += smiles_true_discr+" =/= "+smiles_true+";"
                # get pairwise distances between true and discretized positions
                pairwise_distances = np.linalg.norm(true_atm_pos[:, None, :] - discretized_positions, axis=2)
                print(true_atm)
                print(np.min(pairwise_distances,axis=1))
                print(D_true, D_discr)
                # get closest discretized position for each true position
            if true_pos_smile == original_smile:
                print_string += "true_continuous = original; "
            else:
                print_string += "true_continuous =/= original; "

            print(print_string)
            # print(no_stereochem_discret_smile == original_smile and true_pos_smile == no_stereochem_discret_smile)
            # if not no_stereochem_discret_smile == original_smile and true_pos_smile == no_stereochem_discret_smile:
                # print(no_stereochem_discret_smile, original_smile, true_pos_smile)
        else:
            print("None found")
        print("\n")

def plot_denisty_GMMmean_Actualpositions(true_atoms, true_atoms_positions, predicted_atoms, predicted_atoms_positions, x_grid, y_grid, z_grid, denisty, smiles,
                                         dont_plot=False, select_atom='all'):

    for true_atm, true_atm_pos, pred_atm, pred_atm_pos, dens, smile in zip(true_atoms, true_atoms_positions, predicted_atoms, predicted_atoms_positions, denisty,smiles):
        fig = plt.figure()

        UNQ_elements = ["C", "O", "N", "F"]

        # get possible coordinate positions and stack them
        x, y, z = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
        positions = np.vstack([x, y, z]).T


        pred_atm_pos = np.array(pred_atm_pos)


        # get closest voxel indices for predicted/true atoms (GMM mean positions)
        closest_pred_position_indices = np.argmin(np.linalg.norm(pred_atm_pos[:, None,:] - positions, axis=2),axis=1)
        closest_true_position_indices = np.argmin(np.linalg.norm(true_atm_pos[:, None,:] - positions, axis=2),axis=1)

        # get back the non-flattened indices
        closest_pred_position_indices = np.unravel_index(closest_pred_position_indices, x_grid.shape)
        closest_true_position_indices = np.unravel_index(closest_true_position_indices, x_grid.shape)


        # using 4 atom types
        current_pred_atoms = np.array(pred_atm).flatten()
        current_true_atoms = np.array(true_atm).flatten()

        x_pred,y_pred,z_pred = closest_pred_position_indices[0], \
            closest_pred_position_indices[1], closest_pred_position_indices[2]
        x_true,y_true,z_true = closest_true_position_indices[0], \
            closest_true_position_indices[1], closest_true_position_indices[2]

        center_x_pred, center_y_pred, center_z_pred = np.stack(x_grid[x_pred,y_pred,z_pred]), \
                                                        np.stack(y_grid[x_pred,y_pred,z_pred]), \
                                                        np.stack(z_grid[x_pred,y_pred,z_pred])
        center_x_true, center_y_true, center_z_true = np.stack(x_grid[x_true,y_true,z_true]), \
                                                        np.stack(y_grid[x_true,y_true,z_true]), \
                                                        np.stack(z_grid[x_true,y_true,z_true])

        if select_atom!='all':
            center_x_pred, center_y_pred, center_z_pred = center_x_pred[current_pred_atoms == select_atom], center_y_pred[current_pred_atoms == select_atom], center_z_pred[current_pred_atoms == select_atom]
            center_x_true, center_y_true, center_z_true = center_x_true[current_true_atoms == select_atom], center_y_true[current_true_atoms == select_atom], center_z_true[current_true_atoms == select_atom]

        # create a 3D scatter plot with the true and predicted centers

        # TODO WILL NEED TO DO FOR ALL ATOMS

        true_discretized_pos = np.vstack([center_x_true, center_y_true, center_z_true]).T
        pred_pos = np.vstack([center_x_pred, center_y_pred, center_z_pred]).T

        channel_atoms = ['C', 'O', 'N', 'F']


        mol = build_molecule(torch.tensor(pred_pos), pred_atm, channel_atoms)
        smiles_pred = mol2smiles(mol)

        mol = build_molecule(torch.tensor(true_discretized_pos), true_atm, channel_atoms)
        smiles_true_discr = mol2smiles(mol)


        mol = build_molecule(torch.tensor(true_atm_pos), true_atm, channel_atoms)
        smiles_true_non_discr = mol2smiles(mol)


        # print("Predicted, discretized", smiles_pred)
        # print("True, discretized", smiles_true_discr)
        # print("True, non_discr", smiles_true_non_discr)
        # print("True", smile)


        if smiles_true_discr is not None:
            print(smiles_true_discr.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", ""))
            print(smile.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", ""))
            print(smile.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", "") == smiles_true_discr.replace("[", "").replace("]", "").replace("@", "").replace("H", "").replace("\\", "").replace("/", ""))
        else:
            print("None found")
        print("\n")


        if dens is None:


            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(center_x_pred, center_y_pred, center_z_pred, c="red", s=100, label='GMM identified atom', alpha=0.5)
            ax.scatter(center_x_true, center_y_true, center_z_true, c="blue", s=100, label='True atom', alpha=0.5, marker='x')
            ax.set_title("All atoms")
            fig.legend()
            ax.axes.set_xlim3d(left=x_grid.flatten().min(), right=x_grid.flatten().max())
            ax.axes.set_ylim3d(bottom=y_grid.flatten().min(), top=y_grid.flatten().max())
            ax.axes.set_zlim3d(bottom=z_grid.flatten().min(), top=z_grid.flatten().max())


        elif dens is not None:
            if select_atom != 'all':
                atom_ind = np.argwhere(np.array(channel_atoms) == select_atom).flatten()[0]
            else:
                atom_ind = 0


            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(center_x_pred, center_y_pred, center_z_pred, c="red", s=100, label='GMM identified atom', alpha=0.5)
            ax.scatter(center_x_true, center_y_true, center_z_true, c="blue", s=100, label='True atom', alpha=0.5, marker='x')
            ax.set_title("carbon")
            fig.legend()

            dens_atm = dens[atom_ind]
            ax = fig.add_subplot(122, projection='3d')
            dens_atm = dens_atm.flatten()
            x = x[dens_atm>0.2]
            y = y[dens_atm>0.2]
            z = z[dens_atm>0.2]
            dens_atm = dens_atm[dens_atm>0.2]


            ax.scatter(x,y,z, c=dens_atm, s=100, label='GMM identified atom',alpha=0.5)
            ax.axes.set_xlim3d(left=x_grid.flatten().min(), right=x_grid.flatten().max())
            ax.axes.set_ylim3d(bottom=y_grid.flatten().min(), top=y_grid.flatten().max())
            ax.axes.set_zlim3d(bottom=z_grid.flatten().min(), top=z_grid.flatten().max())

        fig.legend()
        plt.show()


def append_bond_numbers(atom_numbers, bond_lists,ignore_aromatic, data):
    unique_atoms = ["C","O","N","F"]

    shift = 0 if "GEOM" in data else -1
    atom_and_bond_numbers = []
    for atm_numbers, bnd_dict in zip(atom_numbers, bond_lists):
        bond_type_numbers = [0,0,0] if ignore_aromatic else [0,0,0,0]
        for _, v in bnd_dict.items():
            if ignore_aromatic and v == 1.5: print("!!!WARNING!!! append_bond_numbers ENCOUNTERED AROMATIC BOND BUT ignore_aromatic=TRUE"); continue
            elif not ignore_aromatic and v == 1.5: bond_type_numbers[3]+=1
            else: bond_type_numbers[v+shift]+=1
        atm_numbers.extend(bond_type_numbers)
        atom_and_bond_numbers.append(atm_numbers)
    return atom_and_bond_numbers


def get_atom_numbers(atom_lists, elements=None):
    atom_numbers = []
    elements = elements if elements is not None else UNQ_elements['QM9']
    for al in atom_lists:
        al = np.array(al)
        atom_numbers.append([np.sum(al == el) for el in elements])
    return atom_numbers

#loading molecular data
cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + "QM9.txt" if not args.data_file else str(cwd)+"/data/"+args.data_file
if "GEOM" in args.data:
    train_valid_test=True
    dataset_info = retrieve_smiles(args,train_valid_test=train_valid_test)
    Smiles,atoms,atom_pos = dataset_info 
    data_generator, unique_classes = setup_data_files(args, Smiles, train_valid_test,atoms,atom_pos, add_atom_num_cls=True, center_atm_to_grids=args.center_atm_to_grids)
    x_grid, y_grid, z_grid = data_generator.x_grid, data_generator.y_grid, data_generator.z_grid
    H, W, D = x_grid.shape
    C = len(data_generator.atms_considered) + 3 + args.explicit_aromatic
    train_data, val_data = data_generator.epoch_conformers_train.pop(), data_generator.epoch_conformers_val
    bonds_train, atm_symb_train, coords_train, smiles_train, no_atms_cls_train = [td[0] for td in train_data], [td[1] for td in train_data], [td[2] for td in train_data], [td[3] for td in train_data], [td[4] for td in train_data]
    bonds_val, atm_symb_val, coords_val, smiles_val, no_atms_cls_val = [vd[0] for vd in val_data], [vd[1] for vd in val_data], [vd[2] for vd in val_data], [vd[3] for vd in val_data], [vd[4] for vd in val_data]


    train_dataset_args = { 'x_grid':x_grid, 'y_grid':y_grid, 'z_grid':z_grid, 'std_atoms':args.std_atoms, 'ignore_aromatic':not args.explicit_aromatic, 'explicit_hydrogens':args.explicit_hydrogen, 'debug_ds':args.debug_ds, 'subsample_points':args.subsample_points, 'mixed_prec':args.mixed_prec, 'augment_rotations':args.augment_rotations,'explicit_aromatic':args.explicit_aromatic, 'use_subset':-1,'center_atm_to_grids':args.center_atm_to_grids, 'unique_atms':data_generator.atms_considered}

    data = AtomPositionSepBondsDatasetCompact(coords=coords_train, smiles=smiles_train, all_atom_symbols=atm_symb_train, all_bonds=bonds_train, atom_no_classes=no_atms_cls_train, **train_dataset_args)
    data_set = AtomPositionSepBondsDatasetCompact(coords=coords_train, smiles=smiles_train, all_atom_symbols=atm_symb_train, all_bonds=bonds_train, atom_no_classes=no_atms_cls_train, **train_dataset_args)
elif not args.data_file:
    with open(data_path) as f:
        Smiles = f.readlines()
else:

    args.debug_ds = True
    train_valid_test = args.data_file in ["qm9edm_small_explicit_aromatic_explicit_hydrogen_0315_2ndtry.bin", 
                                          "qm9edm_small_explicit_hydrogen_0315_2ndtry.bin",
                                            "qm9edm_almostAll_explicit_aromatic_explicit_hydrogen_033.bin",
                                              "qm9edm_almostAll_explicit_hydrogen_033.bin", 'qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin',
                                              'qm9edm_99data_explicit_hydrogen_033.bin']
    
    path_conf_yml_files = [f for f in os.listdir("./") if "yml" in f]
    args.data_paths = path_conf_yml_files[0]
    from omegaconf import OmegaConf
    data_path_conf = OmegaConf.load(args.data_paths)
    model_path, data_path = data_path_conf.model_path, data_path_conf.data_path
    if "QM9" in args.data: data_path = os.path.join(data_path, "qm9/data/")
    elif "GEOM" in args.data: data_path = os.path.join(data_path, "geom_data")
    dataset_info = retrieve_smiles(args,train_valid_test,data_path)
    if args.use_original_atm_pos :
        Smiles,atoms,atom_pos = dataset_info 
    else:
        Smiles,atoms,atom_pos = dataset_info[0], None, None
    Smiles = [list(Smiles[0]), list(Smiles[1]), list(Smiles[2])] if len(Smiles) == 3 else list(Smiles)
    
    if args.class_conditioning: data, data_val, C, H, W, D, x_grid, y_grid, z_grid, unique_classes = setup_data_files(args, Smiles, train_valid_test,atoms,atom_pos,add_atom_num_cls=args.class_conditioning, data_path=data_path)
    else: data, data_val, C, H, W, D, x_grid, y_grid, z_grid = setup_data_files(args, Smiles, train_valid_test,atoms,atom_pos,add_atom_num_cls=args.class_conditioning)

    x_grid,y_grid,z_grid, positions, smiles, atom_symbs, bonds = pickle.load(open(os.path.join(data_path, args.data_file), "rb"))
    smiles,atom_symbs,bonds,positions = smiles['train'],atom_symbs['train'],bonds['train'],positions['train']
resolution = 0.15


std = 0.05
# std = 0.7

# Set some limits for the molecules: try training on small molecules at first (and smaller resolution)



if std > 0.05:
    x = {"min": -6, "max": 6}
    y = {"min": -5, "max": 5}
    z = {"min": -4, "max": 5}
    limits = [x, y, z]

else:
    x = {"min": -4, "max": 4}
    y = {"min": -3, "max": 3}
    z = {"min": -2, "max": 3}
    limits = [x, y, z]

x = {"min":float('inf'),"max":float('-inf')}
y = {"min":float('inf'),"max":float('-inf')}
z = {"min":float('inf'),"max":float('-inf')}
limits = [x, y, z]

UNQ_elements = {'QM9':["C","O","N","F"], 'QM9EDM':["C","O","N","F"]}





def recheck_generated_fields():
    #! that one molecule containing a wrong bound
    # python sanity_check_resolution.py --resolution 0.33 --explicit_hydrogen --data_file qm9edm_almostAll_explicit_hydrogen_033.bin --recheck_generated_flds --optimize_bnd_gmm_weights
    fields = "misc/model-18qm9edm_almostAll_explicit_hydrogen_033_runema_results.pkl"
    fields = "misc/model-18qm9edm_almostAll_explicit_hydrogen_033_runema_results.pkl"
    fields = "misc/model-10qm9edm_all_clsfreeguid_explicit_aromatic_explicit_hydrogen_033_cleanGaussians_augment_rotema_results.pkl"

    # python sanity_check_resolution.py --resolution 0.33 --explicit_hydrogen --data_file qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin --recheck_generated_flds --optimize_bnd_gmm_weights --explicit_aromatic
    # fields = "misc/model-14qm9edm_all_clsfreeguid_explicit_aromatic_explicit_hydrogen_033_runema_results.pkl"

    # python sanity_check_resolution.py --resolution 0.33 --explicit_hydrogen --data_file qm9edm_99data_explicit_hydrogen_033.bin --recheck_generated_flds --optimize_bnd_gmm_weights
    # fields = "misc/model-12qm9edm_all_clsfreeguid_explicit_hydrogen_033_runema_results.pkl"

    generated_molecules = pickle.load(open(fields, "rb"))
    all_fields = []
    for f_batch in generated_molecules[4]:
        for f in f_batch:
            all_fields.append(f[0])
    all_fields = torch.tensor(np.array(all_fields))
    unq_elem = UNQ_elements['QM9']

    val, unq, nov, all_bnd_atm_smil,failed_fields, all_coresp_weights = fit_pred_field_sep_chn_batch(all_fields, 0.1, x_grid, y_grid, z_grid,len(UNQ_elements.get('QM9EDM')), 
                                        UNQ_elements.get('QM9EDM'), smiles,noise_std=0.0, normalize01=True,return_atm_pos_bdns=True,
                                        explicit_aromatic=args.explicit_aromatic, explicit_hydrogen=args.explicit_hydrogen, optimize_bnd_gmm_weights=True,
                                        threshold_atm=args.threshold_atm, threshold_bond=args.threshold_bond)
    print("Optimized bonds", val, unq, nov)
    breakpoint()

    val, unq, nov, all_bnd_atm_smil,failed_fields, all_coresp_weights = fit_pred_field_sep_chn_batch(all_fields, 0.1, x_grid, y_grid, z_grid,len(UNQ_elements.get('QM9EDM')), 
                                        UNQ_elements.get('QM9EDM'), smiles,noise_std=0.0, normalize01=True,return_atm_pos_bdns=True,
                                        explicit_aromatic=args.explicit_aromatic, explicit_hydrogen=args.explicit_hydrogen, optimize_bnd_gmm_weights=False,
                                        threshold_atm=args.threshold_atm, threshold_bond=args.threshold_bond)
    print("Non optimized bonds", val, unq, nov)
    breakpoint()

    # val, unq, nov, generated_smiles = fit_pred_field_sep_chn_batch(all_fields,0.1,x_grid, y_grid,z_grid, len(unq_elem),
    #                                             unq_elem, smiles_list=smiles,return_atoms=True, 
    #                         return_smiles=True, atom_and_bond_numbers=None, std=std,coords=None, 
    #                         new_resolutions=[0.15,0.15,0.15],all_atom_lists=None, true_smiles=None,
    #                             noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
    #                             explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)


if args.recheck_generated_flds:
    recheck_generated_fields()

if args.compact_batch:
    if "GEOM" in args.data:
        collate_fn = collate_fn_compact_expl_h_clsfreeguid_debugds_geom
    else:
        if args.class_conditioning:
            collate_fn = collate_fn_compact_expl_ah_clsfreeguid_debugds if not args.ignore_aromatic else collate_fn_compact_expl_h_clsfreeguid_debugds
        else:
            collate_fn = collate_fn_compact_expl_ah_debugds if not args.ignore_aromatic else collate_fn_compact_expl_h_debugds
# check current method


def plot_rmvd_weight(weights,pi_threshold):
    for w in weights:
        for bnd in [1,2,3,4]:
            if not len(w[bnd]): continue
            if np.max(w[bnd][0]) * pi_threshold > np.min(w[bnd][0]):
                plt.bar((np.arange(len(w[bnd][0]))), w[bnd][0])
                plt.show()


if args.method == "separate_channels" and args.data == "GEOM":
    # * test no thresholding, no 
    # python sanity_check_resolution.py --data GEOM --threshold_bond 0.4 --threshold_atm 0.4 --augment_rotations --remove_thresholding_pdf_vals --method separate_channels --class_conditioning --compact_batch --explicit_hydrogen --subset_confs 2 --optimize_bnd_gmm_weights
    atm_symb_bnd_by_index = [atm for atm in data.unique_atms]
    atm_symb_bnd_by_index.extend([1,2,3,4] if args.explicit_aromatic else [1,2,3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    data = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    items = next(iter(data))
    
    coords, inds, N_list, classes,sm, bnd_t, atm_t = items
    unq_elem = UNQ_elements['QM9']

    x_flat,y_flat,z_flat = torch.tensor(x_grid.flatten(),device=device, dtype=torch.float32), \
        torch.tensor(y_grid.flatten(),device=device,dtype=torch.float32), torch.tensor(z_grid.flatten(),device=device,dtype=torch.float32)
    grid_inputs = torch.stack([x_flat,y_flat,z_flat],dim=1)
    inp = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = C, grid_shapes=[H,W,D],
                                            threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)

    noise_lvls = [0, 0.001, 0.01, 0.05, 0.1, 0.2] if args.noise_type =='normal' else [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4,0.64]
    all_vals = []
    for std_noise in noise_lvls:

        for ind, d in enumerate(data):
            # if ind * batch_size < 50000: continue # start at some "serious" molecules
            coords, inds, N_list, classes,sm, bnd_t, atm_t = d
            sample_ = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = C, grid_shapes=[H,W,D],threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)
            atom_numbers = get_atom_numbers(atm_t, atm_symb_bnd_by_index[:-3-args.explicit_aromatic])
            atom_and_bond_numbers = append_bond_numbers(atom_numbers, bnd_t,args.ignore_aromatic, data=args.data)
            val, unq, nov, generated_smiles, weights = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, sm,return_atoms=True, 
                            return_smiles=True, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coords, 
                            new_resolutions=[0.15,0.15,0.15],all_atom_lists=atm_t, true_smiles=sm,noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                                explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen, threshold_bond=args.threshold_bond, 
                                optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold, atm_symb_bnd_by_channel=atm_symb_bnd_by_index)
            # plot_rmvd_weight(weights, args.pi_threshold)            
            all_vals.append(val)

            print("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))
            logging.info("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))

            
    breakpoint()


elif args.method == "ditsance_based" and args.data_file:
    # * test   no thresholding, no 
    # python sanity_check_resolution.py --data_file qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin --threshold_bond 0.4 --threshold_atm 0.4 --augment_rotations --remove_thresholding_pdf_vals --method separate_channels --explicit_aromatic --class_conditioning --compact_batch --explicit_hydrogen   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    data = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_general_debug)

    items = next(iter(data))
    
    coords, inds, N_list, classes,sm, bnd_t, atm_t = items
    unq_elem = UNQ_elements['QM9']

    x_flat,y_flat,z_flat = torch.tensor(x_grid.flatten(),device=device, dtype=torch.float32), \
        torch.tensor(y_grid.flatten(),device=device,dtype=torch.float32), torch.tensor(z_grid.flatten(),device=device,dtype=torch.float32)
    grid_inputs = torch.stack([x_flat,y_flat,z_flat],dim=1)
    inp = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = 8+(not args.ignore_aromatic), grid_shapes=[H,W,D],
                                            threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)

    noise_lvls = [0, 0.1, 0.15, 0.2, 0.3] if args.noise_type =='normal' else [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4,0.64]
    all_vals = []
    noise2_val = {}
    for std_noise in noise_lvls:

        for ind, d in enumerate(data):
            if ind * batch_size < 50000: continue # start at some "serious" molecules
            coords, inds, N_list, classes,sm, bnd_t, atm_t = d
            sample_ = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = 5, grid_shapes=[H,W,D],
                                            threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)
            
            atom_numbers = get_atom_numbers(atm_t)
            atom_and_bond_numbers = append_bond_numbers(atom_numbers, bnd_t,args.ignore_aromatic, data="QM9")
            val, unq, nov, smiles, weights, rmvd_bonds = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, sm,return_atoms=True, 
                            return_smiles=True, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coords, 
                            new_resolutions=[0.15,0.15,0.15],all_atom_lists=atm_t, true_smiles=sm,noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                                explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen, threshold_bond=args.threshold_bond, 
                                optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold,
                                atm_dist_based=True)
            print(val)
            all_vals.append(val)
            if ind * batch_size > 50300: break # test 300 samples

        print("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))
        logging.info("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))
        noise2_val[std_noise] = np.mean(all_vals)
    pickle.dump(noise2_val, open("all_vals_{}_dist_based.bin", "wb"))


    breakpoint()
elif args.method == "separate_channels" and args.data_file:
    # * test   no thresholding, no 
    # python sanity_check_resolution.py --data_file qm9edm_99data_explicit_aromatic_explicit_hydrogen_033.bin --threshold_bond 0.4 --threshold_atm 0.4 --augment_rotations --remove_thresholding_pdf_vals --method separate_channels --explicit_aromatic --class_conditioning --compact_batch --explicit_hydrogen   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    data = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    items = next(iter(data))
    
    coords, inds, N_list, classes,sm, bnd_t, atm_t = items
    unq_elem = UNQ_elements['QM9']

    x_flat,y_flat,z_flat = torch.tensor(x_grid.flatten(),device=device, dtype=torch.float32), \
        torch.tensor(y_grid.flatten(),device=device,dtype=torch.float32), torch.tensor(z_grid.flatten(),device=device,dtype=torch.float32)
    grid_inputs = torch.stack([x_flat,y_flat,z_flat],dim=1)
    inp = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = 8+(not args.ignore_aromatic), grid_shapes=[H,W,D],
                                            threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)

    noise_lvls = [0, 0.1, 0.15, 0.2, 0.3] if args.noise_type =='normal' else [0.1, 0.2, 0.4,0.64]
    all_vals = []
    noise2_val = {}

    for std_noise in noise_lvls:

        for ind, d in enumerate(data):
            if ind * batch_size < 50000: continue # start at some "serious" molecules
            coords, inds, N_list, classes,sm, bnd_t, atm_t = d
            sample_ = create_gaussian_batch_pdf_values(x=grid_inputs, coords=coords, N_list=N_list, std=args.std_atoms, device=device, gaussian_indices=inds,
                                            no_fields = 8+(not args.ignore_aromatic), grid_shapes=[H,W,D],
                                            threshold_vlals= not args.remove_thresholding_pdf_vals, backward_mdl_compat=args.backward_mdl_compat)
            atom_numbers = get_atom_numbers(atm_t)
            atom_and_bond_numbers = append_bond_numbers(atom_numbers, bnd_t,args.ignore_aromatic, data="QM9")

            val, unq, nov, all_bnd_atm_smil,failed_fields, bond_weights, removed_bonds = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, sm,return_atoms=True, 
                        return_smiles=False, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coords, 
                        new_resolutions=[0.15,0.15,0.15],all_atom_lists=atm_t, true_smiles=sm,noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                            explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen, threshold_bond=args.threshold_bond, 
                            optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold,
                            optimize_atom_positions=args.optimize_atom_positions, return_atm_pos_bdns=True)
            breakpoint()

            # val, unq, nov, smiles, weights, rmvd_bonds = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, sm,return_atoms=True, 
            #                 return_smiles=True, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coords, 
            #                 new_resolutions=[0.15,0.15,0.15],all_atom_lists=atm_t, true_smiles=sm,noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
            #                     explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen, threshold_bond=args.threshold_bond, 
            #                     optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold,
            #                     optimize_atom_positions=args.optimize_atom_positions)
            # breakpoint()
            
            # plot_rmvd_weight(weights, args.pi_threshold)            
            all_vals.append(val)
            if ind * batch_size > 50300: break # test 300 samples

        print("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))
        logging.info("Valid percentage for noise {} is {}".format(std_noise, np.mean(all_vals)))
        noise2_val[std_noise] = np.mean(all_vals)

    pickle.dump(noise2_val, open("all_vals_{}_bond_based.bin", "wb"))
    breakpoint()

elif args.method == "separate_channels" and args.data_file:
    # DataLoader(data, batch_size=10, shuffle=True, pin_memory=False, num_workers=1)
    random_inds = np.random.choice(len(smiles), 50, replace=False)

    positions, smiles, atom_symbs, bonds = [positions[i] for i in random_inds], [smiles[i] for i in random_inds],\
                                            [atom_symbs[i] for i in random_inds], [bonds[i] for i in random_inds]
    unq_elem = UNQ_elements['QM9']
    noise_lvls = [0, 0.001, 0.01, 0.05, 0.1, 0.2] if args.noise_type =='normal' else [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4,0.64]

    data = DataLoader(data, batch_size=300, shuffle=False, collate_fn=custom_collate_agnostic_atm_types)

    for std_noise in noise_lvls:

        for ind, d in enumerate(data):
            sample_, smile, all_bonds_dictionaries, coord, all_atom_lists = d
            print(sample_.shape)



            atom_numbers = get_atom_numbers(all_atom_lists)
            atom_and_bond_numbers = append_bond_numbers(atom_numbers, all_bonds_dictionaries,args.ignore_aromatic)


            val, unq, nov, generated_smiles = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, smile,return_atoms=True, 
                            return_smiles=True, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coord, 
                            new_resolutions=[0.15,0.15,0.15],all_atom_lists=all_atom_lists, true_smiles=smile,
                                noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                                explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)


            # atom_numbers = get_atom_numbers(atom_symbs)
            # atom_and_bond_numbers = append_bond_numbers(atom_numbers, bonds,args.ignore_aromatic)

            # val, unq, nov = fit_pred_field_sep_chn_batch(data,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, smiles,return_atoms=True, 
            #                         return_smiles=False, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=positions, 
            #                         new_resolutions=[0.15,0.15,0.15],all_atom_lists=atom_symbs, true_smiles=smiles,
            #                             noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
            #                             explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)
            print(std_noise, val,unq, nov)
            logging.info("Valid percentage for noise {} is {}".format(std_noise, val))
            break


elif args.method == "separate_channels_same_atm_chn":
    if not args.data_file: print("Please specify a data file for separate_channels_same_atm_chn method sanity checking; Exiting..." ); exit(1)
    args.data_file = args.data_file.replace(".bin", "") + ".bin"
    x_grid,y_grid,z_grid, atom_pos, smiles, atoms, bonds = pickle.load(open(os.path.join(get_data_folder() + args.data_file), "rb")) 
    dataset = AtomPositionSepBondsDataset(atom_pos['train'], smiles['train'], x_grid,y_grid,z_grid, 0.05, atoms['train'], bonds['train'], 
                                       ignore_aromatic=args.ignore_aromatic, explicit_hydrogens=args.explicit_hydrogen, debug_ds=False,
                                       agnostic_atm_types=args.agnostic_atm_types)
    data = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_agnostic_atm_types)
    UNQ_elements = {'QM9':["C","O","N","F", "H"]}
    unq_elem = UNQ_elements['QM9']

    # gather data

    # for d in data:
    #     sample_, smile, all_bonds_dictionaries, coord, all_atom_lists = d
    #     for s, sm, bnd, crd, atm in zip(sample_, smile, all_bonds_dictionaries, coord, all_atom_lists):

    #         actual_bnd = [[k[0],k[1],v] for k,v in bnd.items()]

    #         plot_one_mol(crd, actual_bnd, plot_bnd=5, threshold=0.9, field=s, x_grid=x_grid,y_grid=y_grid,z_grid=z_grid, atm_symb=atm)
    #         breakpoint()
    problem_smile = 0
    for ind, s in enumerate(dataset.smiles): 
        if s == "C[C@@]12C[C@@](C1)(O2)C(=O)N": problem_smile = ind; break

    
    for ind, d in enumerate(data):
        sample_, smile, all_bonds_dictionaries, coord, all_atom_lists = d
        
        dp = dataset.__getitem__(problem_smile)

        sample_, smile, all_bonds_dictionaries, coord, all_atom_lists = dp['field'].unsqueeze(0), [dp['smile']],\
              [dp['bond_type']], [dp['coord']], [dp['atoms']]
        

        atom_numbers = get_atom_numbers(all_atom_lists)
        atom_and_bond_numbers = append_bond_numbers(atom_numbers, all_bonds_dictionaries,args.ignore_aromatic)


        val, unq, nov, generated_smiles = fit_pred_field_sep_chn_batch(sample_,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, smile,return_atoms=True, 
                        return_smiles=True, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coord, 
                        new_resolutions=[0.15,0.15,0.15],all_atom_lists=all_atom_lists, true_smiles=smile,
                            noise_std=0, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                            explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)
        print(val, unq, nov, generated_smiles)
        breakpoint()
    AtomPositionSepBondsDataset
    final_smiles, data,unq_elem,x_grid,y_grid,z_grid, \
        coords,all_atom_lists,all_bonds_dictionaries,new_resolutions = get_mol_reps_w_bonds(Smiles[:50],"QM9",std=std,use_pca=False,rescale=False, 
                                                    limits=limits, resolution=args.resolution,return_true_mens=True,data_cutoff=0.0, discretized_center=False,
                                                    ignore_aromatic=args.ignore_aromatic, normalize01=True, explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)
    
    
elif args.method == 'separate_channels':
    final_smiles, data,unq_elem,x_grid,y_grid,z_grid, \
        coords,all_atom_lists,all_bonds_dictionaries,new_resolutions = get_mol_reps_w_bonds(Smiles[:50],"QM9",std=std,use_pca=False,rescale=False, 
                                                    limits=limits, resolution=args.resolution,return_true_mens=True,data_cutoff=0.0, discretized_center=False,
                                                    ignore_aromatic=args.ignore_aromatic, normalize01=True, explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)
    data = torch.tensor(np.stack(data))

    noise_lvls = [0, 0.001, 0.01, 0.05, 0.1, 0.2] if args.noise_type =='normal' else [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4,0.64]
    print(Smiles[:50])

    breakpoint()

    for std_noise in noise_lvls:
        atom_numbers = get_atom_numbers(all_atom_lists)
        atom_and_bond_numbers = append_bond_numbers(atom_numbers, all_bonds_dictionaries,args.ignore_aromatic)


        atom_numbers = get_atom_numbers(all_atom_lists)

        val, unq, nov = fit_pred_field_sep_chn_batch(data,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, final_smiles,return_atoms=True, 
                                return_smiles=False, atom_and_bond_numbers=atom_and_bond_numbers, std=std,coords=coords, 
                                new_resolutions=new_resolutions,all_atom_lists=all_atom_lists, true_smiles=final_smiles,
                                    noise_std=std_noise, determine_bonds=False,normalize01=True, noise_type=args.noise_type,
                                    explicit_aromatic=not args.ignore_aromatic, explicit_hydrogen=args.explicit_hydrogen)
        print(std_noise, val,unq, nov)
        logging.info("Valid percentage for noise {} is {}".format(std_noise, val))

else:
    resolution = 0.15
    final_smiles, data,unq_elem,x_grid,y_grid,z_grid, \
        coords,all_atom_lists,all_bonds_dictionaries,new_resolutions = get_mol_reps(Smiles[:10],"QM9",std=std,use_pca=True,rescale=False, 
                                                    limits=limits, resolution=resolution,return_true_mens=True,data_cutoff=0, discretized_center=False)
    data = torch.tensor(np.stack(data))
    atom_numbers = get_atom_numbers(all_atom_lists)

    noise_lvls = [0, 0.001, 0.01, 0.05, 0.1, 0.2] if args.noise_type =='normal' else [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4]


    for std_noise in noise_lvls:
        val = fit_pred_field_curve_batch(data,0.1,x_grid, y_grid,z_grid, len(unq_elem),unq_elem, final_smiles,return_atoms=True, 
                                return_smiles=True, true_atm_numbers=atom_numbers, std=std,coords=coords, 
                                new_resolutions=new_resolutions,all_atom_lists=all_atom_lists, true_smiles=final_smiles, noise_std=std_noise, noise_type=args.noise_type)
        print(val)
        logging.info("Valid percentage for noise {} is {}".format(std_noise, val))

breakpoint()


