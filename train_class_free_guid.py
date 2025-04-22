from omegaconf import OmegaConf

# normalize01? separate channels?
from utils import AtomPositionDataset,AtomPositionSepBondsDataset
from visualize_utils import visualize_failed_mols_w_hydrogens
import pickle
import warnings
import os
#from utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
from utils import *
import pandas as pd
import numpy as np
import argparse
import sys
import time
import torch
import torch.optim as optim
import random
from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion,GaussianDiffusionDiffSchds, Unet, CNN, Trainer, Unet3D
import logging
from dit_model import DiT_S_2_3D
UNQ_elements = {'QM9':["C","O","N","F"], 'QM9EDM':["C","O","N","F"], "QM9EDM_small":["C","O","N","F"], "GEOM":['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'H']}

parser = argparse.ArgumentParser('FieldGen')
parser.add_argument('--ntimesteps', type=int, default=1000)
parser.add_argument('--use_original_atm_pos', default=False, action="store_true", help='do not extract atom positions with rdkit, from smiles; '
                    'by having atm pos from rdkit-smiles-processing, stereochemical information can be lost')
parser.add_argument('--explicit_hydrogen', default=False, action='store_true')
parser.add_argument('--explicit_aromatic', default=False, action="store_true")
parser.add_argument('--resolution', type=float, default=0.33)
parser.add_argument('--no_limits', default=False, action="store_true", help="no restrictions on molecule size")
parser.add_argument('--gradient_accumulate_every', type=int, default=1)
parser.add_argument('--large_mdl', default=False, action="store_true")
parser.add_argument('--run_name', default="", type=str)
parser.add_argument('--std_atoms', type=float, default=0.05)
parser.add_argument('--not_use_pca', default=False, action='store_true')
parser.add_argument('--model', default='cnn', type=str)
parser.add_argument('--input_type', default='separate_bonds', type=str, choices=['separate_bonds', 'only_atoms'],
                    help="only atoms is the original formulation of bonds being determined by atom distances"
                    "separate bonds is the new formulation of bonds being determined by pdfs on separate corresponding channels")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--val_batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=8e-5)
parser.add_argument('--rescale', default=False, action="store_true")
parser.add_argument('--no_sep_bnd_chn', default=False, action="store_true")
parser.add_argument('--nsamples', type=int, default=1000)
parser.add_argument("--data", type=str, default="QM9", choices=["ZINC","QM9", "QM9EDM", "QM9EDM_small", "GEOM"])
parser.add_argument("--load_name", type=str, default=None)
parser.add_argument("--force_create_data", default=False, action='store_true')
parser.add_argument("--test_every", default=10000, type=int)
parser.add_argument("--data_file", type=str, default="none", help="specify file containing grid, coordinates and smiles"
                                                                  "of already-preprocessed-wPCA mols file")
parser.add_argument('--kernels', nargs='+', help='Kernel dimensions for CNN architecture')
parser.add_argument("--subsample_points", default=-1, type=float, help="Instead of traianing on the whole field, predict a subset")
parser.add_argument('--beta_schedule', default='sigmoid', choices=['linear', 'sigmoid', 'cosine'], help='Noise beta schedule')
parser.add_argument('--debug_ds', default=False, action='store_true', help='return coordinates and bond type from dataset in order to visualize/debug')
parser.add_argument("--mixed_prec", default=False, action='store_true', help="use mixed precision training")
parser.add_argument("--augment_rotations", default=False, action="store_true", help="randomly rotate the coordinates before creating the fields")
parser.add_argument("--compact_batch", default=False, action="store_true", help="return only the atom types, bond types and their coordinates from the dataloader")
parser.add_argument("--remove_thresholding_pdf_vals", default=False, action='store_true')
parser.add_argument("--backward_mdl_compat", default=False, action='store_true', help='old models had thresholding only on atoms; create the same for compact_batch ')
parser.add_argument("--optimize_bnd_gmm_weights", default=False, action='store_true')
parser.add_argument("--threshold_bond", default=0.75, type=float, help='threshold where bond is considered an actual bond')
parser.add_argument("--threshold_atm", default=0.75, type=float, help='threshold where atm is considered')
parser.add_argument("--blur", default=False, action='store_true', help='use blur component to diffusion')
parser.add_argument("--center_atm_to_grids", default=False, action='store_true', help="center the mean atm positions to some grid values; bonds are placed inbetween the new atms")
parser.add_argument("--noise_scheduler_conf", default=None, type=str, help="load a noise scheduler configuration from a yaml file")
parser.add_argument("--arom_cycle_channel", default=False, action='store_true', help="add an extra channel for aromatic cycles of 5/6 atoms")
parser.add_argument("--subset_confs", default=30, type=int, help="Number of conf files to be used. Use lower for faster loading")
parser.add_argument("--multi_gpu", default=False, action='store_true', help="use multiple gpus if available")
parser.add_argument("--data_paths", default=None, type=str, help="specify paths to data files for each resolution")
parser.add_argument("--consider_arom_chns_as_atms", default=False, action="store_true", help='legacy compatibility w a problem where no of atms for the class conditioning considers the aromatic channles')
parser.add_argument("--fix_chiral_conf", default=None, type=str, help="fix chiral conf of molecules containing specific conformation", choices=[None, "SP2CSP3COH"])
parser.add_argument("--fix_pi_values", default=False, action='store_true', help="fix pi of the GMM to have same constant across dataset")
parser.add_argument("--remove_bonds", default=False, action='store_true', help="train only on atoms and extract bonds based on their positions")
parser.add_argument("--inv_rot_loss_angles", default=-1, type=int, help="rotate ground truth by some no of angles backprop through softmin")
parser.add_argument("--only_min_activ", default=False, action="store_true", help="for invariant rotation loss, take only min wrt loss")
parser.add_argument("--objective", default="pred_noise", type=str, choices=["pred_noise", "pred_x0"], help="objective to use for training")
parser.add_argument("--create_simple_chiral_ds", default=False, action="store_true", help="create a dataset with pre-specified chiral config")
# * DiT architecture hyperparameters

parser.add_argument("--depth", default=4, type=int, help="depth of DiT architecture")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden dimension")
parser.add_argument("--small_attention", default=False, action='store_true', help="use restrcited attention into 5x5 grids")
parser.add_argument("--patch_size", default=2, type=int, help="patch size for attention")
parser.add_argument("--att_size", default=2, type=int, help="patch size for attention")
parser.add_argument("--cond_variable", default=[], nargs="+", help="specify a conditioning parameter for conditional generation")
parser.add_argument("--discrete_conditioning", default=False, action="store_true", help="use discrete conditioning variables as classes, representing bind indices")
parser.add_argument("--remove_null_cond_emb", default=False, action="store_true")

args = parser.parse_args()

logging.getLogger('some_logger')
logging.basicConfig(filename=args.run_name+".logging", level=logging.INFO, force=True)



if args.inv_rot_loss_angles != -1 and (args.remove_thresholding_pdf_vals or args.backward_mdl_compat or not args.fix_pi_values):
    print("!!!WARNING!!! inv_rot_loss_angles is only available for remove_thresholding_pdf_vals=False, backward_mdl_compat=False and fix_pi_values=True. Setting them to False, False, True")
    logging.info("!!!WARNING!!! inv_rot_loss_angles is only available for remove_thresholding_pdf_vals=False, backward_mdl_compat=False and fix_pi_values=True. Setting them to False, False, True")
    args.inv_rot_loss_angles = -1

if args.center_atm_to_grids and not args.compact_batch:
    print("!!!WARNING!!! center_atm_to_grids is only available for compact_batch=True. Setting compact_batch=True")
    logging.info("!!!WARNING!!! center_atm_to_grids is only available for compact_batch=True. Setting compact_batch=True")
    args.compact_batch = True

if args.augment_rotations and not args.compact_batch:
    print("!!!WARNING!!! augment_rotations is only available for compact_batch=True. Setting compact_batch=True")
    logging.info("!!!WARNING!!! augment_rotations is only available for compact_batch=True. Setting compact_batch=True")
    args.compact_batch = True

if args.compact_batch:
    print("MAKE SURE BACKWARD COMPATIBILITY IS APPROPRIATELY SET. RIGHT NOW U REQUIRED {}".format(args.backward_mdl_compat))
    logging.info("MAKE SURE BACKWARD COMPATIBILITY IS APPROPRIATELY SET. RIGHT NOW U REQUIRED {}".format(args.backward_mdl_compat))


if args.data == 'QM9EDM' and not args.no_limits: 
    print("QM9EDM data SHOULD NOT have any limits for molecule sizes; setting no_limits=True"); args.no_limits=True


if "GEOM" in args.data and args.explicit_aromatic:
    confs = min(5, args.subset_confs)
    print(f"SETTING args.subset_confs={confs} for explicit aromatic (train on lowest 5 energy conformations)")
    args.subset_confs = confs

# if "GEOM" in args.data and args.explicit_aromatic:
#     print("GEOM DATA EXPLICIT AROMATIC IS NOT IMPLEMENTED. SETTING explicit_aromatic TO FALSE"); args.explicit_aromatic=False

if not args.compact_batch and args.arom_cycle_channel:
    print("!!!WARNING!!! AROMATIC CYCLE CHANNEL IS ONLY AVAILABLE FOR COMPACT_BATCH."
          " SETTING arom_cycle_channel TO FALSE; MAKE USE THE DATA SPECIFIED IN data_file is correct"); args.arom_cycle_channel=False

if "GEOM" in args.data and args.fix_pi_values:
    print("!!!WARNING!!! FIXING PI VALUE IS NOT IMPLEMENTED FOR GEOM DATA. SETTING fix_pi_values TO FALSE")
    args.fix_pi_values = False

if args.data_paths is None:
    path_conf_yml_files = [f for f in os.listdir("./") if "yml" in f]
    args.data_paths = path_conf_yml_files[0]
    print("!!!WARNING!!! NO DATA PATHS CONFIG SPECIFIED. USING {}".format(args.data_paths))

if args.fix_chiral_conf is not None and args.data == "GEOM":
    print("!!!WARNING!!! FIXING CHIRAL CONFS IS NOT IMPLEMENTED FOR GEOM DATA. SETTING fix_chiral_conf TO NONE")
    args.fix_chiral_conf = None


data_path_conf = OmegaConf.load(args.data_paths)
model_path, data_path = data_path_conf.model_path, data_path_conf.data_path
if "QM9" in args.data: data_path = os.path.join(data_path, "qm9/data/")
elif "GEOM" in args.data: data_path = os.path.join(data_path, "geom_data")
# train_valid_test is a pseudoargument for now for coda readability; QM9EDM has such splits so use them as such
train_valid_test = "QM9EDM" in args.data or "GEOM" in args.data

dataset_info = retrieve_smiles(args,train_valid_test,data_path)
if (args.use_original_atm_pos and args.data!= "GEOM"):
    Smiles,atoms,atom_pos,cond_variable = dataset_info
elif args.use_original_atm_pos:
    Smiles,atoms,atom_pos,cond_variable = dataset_info, None, None, None
else:
    Smiles,atoms,atom_pos,cond_variable = dataset_info[0], None, None, None
Smiles = [list(Smiles[0]), list(Smiles[1]), list(Smiles[2])] if len(Smiles) == 3 else list(Smiles)

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = args.resolution
# Set some limits for the molecules: try training on small molecules at first (and smaller resolution)




if args.no_limits:
    x = {"min":float('inf'),"max":float('-inf')}
    y = {"min":float('inf'),"max":float('-inf')}
    z = {"min":float('inf'),"max":float('-inf')}

else:
    # limits for "small data"
    # x = {"min": -4, "max": 4}
    # y = {"min": -3, "max": 3}
    # z = {"min": -2, "max": 3}
    # [[,],[], [-4.870370240210199,4.951091612332089]]
    # limits chosen s.t. 99.4% of the moelcules fit, with 32^3 voxels on a 0.33A resolution
    x = {"min":-5.26791692653561, "max":5.10421340654044}
    y = {"min":-4.777702113781722,"max":5.13692682839705}
    z = {"min":-4.8703702402102, "max":4.95109161233209}


limits = [x, y, z]

if "GEOM" in args.data:
    # self.results_folder self.run_name + '.pt'
    # if exists(self.data_generator):
    data_generator, unique_classes, bins = setup_data_files(args, Smiles, train_valid_test,atoms,atom_pos, 
                                                    add_atom_num_cls=True, center_atm_to_grids=args.center_atm_to_grids,
                                                    data_path=data_path, consider_arom_chns_as_atms=args.consider_arom_chns_as_atms,
                                                      fix_chiral_conf=args.fix_chiral_conf)
    x_grid, y_grid, z_grid = data_generator.x_grid, data_generator.y_grid, data_generator.z_grid
    H, W, D = x_grid.shape
    C = len(data_generator.atms_considered) + 3 + args.explicit_aromatic
    train_data, val_data = data_generator.epoch_conformers_train.pop(), data_generator.epoch_conformers_val


    bonds_train, atm_symb_train, coords_train, smiles_train, no_atms_cls_train = [td[0] for td in train_data], [td[1] for td in train_data], [td[2] for td in train_data], [td[3] for td in train_data], [td[4] for td in train_data]
    bonds_val, atm_symb_val, coords_val, smiles_val, no_atms_cls_val = [vd[0] for vd in val_data], [vd[1] for vd in val_data], [vd[2] for vd in val_data], [vd[3] for vd in val_data], [vd[4] for vd in val_data]

    train_dataset_args = { 'x_grid':x_grid, 'y_grid':y_grid, 'z_grid':z_grid, 'std_atoms':args.std_atoms, 'ignore_aromatic':not args.explicit_aromatic, 'explicit_hydrogens':args.explicit_hydrogen, 'debug_ds':args.debug_ds, 'subsample_points':args.subsample_points, 'mixed_prec':args.mixed_prec, 'augment_rotations':args.augment_rotations,'explicit_aromatic':args.explicit_aromatic, 'use_subset':-1,'center_atm_to_grids':args.center_atm_to_grids, 'unique_atms':data_generator.atms_considered}
    val_dataset_args = {'x_grid':x_grid, 'y_grid':y_grid, 'z_grid':z_grid, 'std_atoms':args.std_atoms, 'ignore_aromatic':not args.explicit_aromatic, 'explicit_hydrogens':args.explicit_hydrogen, 'debug_ds':args.debug_ds, 'subsample_points':args.subsample_points, 'mixed_prec':args.mixed_prec, 'augment_rotations':args.augment_rotations, 'explicit_aromatic':args.explicit_aromatic, 'use_subset':-1,'center_atm_to_grids':args.center_atm_to_grids, 'unique_atms':data_generator.atms_considered}

    # data = AtomPositionSepBondsDatasetCompact(coords=coords_train, smiles=smiles_train, all_atom_symbols=atm_symb_train, all_bonds=bonds_train, atom_no_classes=no_atms_cls_train, **train_dataset_args)
    # data_val = AtomPositionSepBondsDatasetCompact(coords=coords_val, smiles=smiles_val, all_atom_symbols=atm_symb_val,  all_bonds=bonds_val,atom_no_classes=no_atms_cls_val, **val_dataset_args)

    data, data_val = data_generator.get_next_epoch_dl(train_dataset_args, val_dataset_args)
    # data.__getitem__(5000)
    # breakpoint()
    # print(data_generator.atms_considered, data.unique_atms, data_val.unique_atms)
    atoms_considered=data_generator.atms_considered
    norm_factors = None
    atomno_conditioned_variable_sampler = None


else:
    create_simple_chiral_ds(args,data_path=data_path) # do I use this for anything? Probably not
    data, data_val, C, H, W, D, x_grid, y_grid, z_grid, unique_classes = setup_data_files(args, Smiles, train_valid_test,atoms,atom_pos, add_atom_num_cls=True, center_atm_to_grids=args.center_atm_to_grids, data_path=data_path, consider_arom_chns_as_atms=args.consider_arom_chns_as_atms, fix_chiral_conf=args.fix_chiral_conf, cond_variable=cond_variable)
    if args.remove_bonds: C = C - 3 - args.explicit_aromatic
    data_generator = None

    train_dataset_args, val_dataset_args= None, None
    atoms_considered = None # defaults are already created for QM9, leave them alone if not using GEOM data
    # set all axis to the largest one (x), so that everything fits when randomly rotating
    norm_factors = get_cond_var_normalizing_factors(data, args)
    if norm_factors is not None: atomno_conditioned_variable_sampler = ConditioningVarSampler(data, norm_factors) 
    else: atomno_conditioned_variable_sampler = None


expl_H = check_containing_H(data)
if expl_H ^ args.explicit_hydrogen: print("WARNING: explicit hydrogen is set to {} but data contains {} hydrogen. Setting args.explicit_hydtogen to {}".format(args.explicit_hydrogen, "explicit" if expl_H else "implicit", expl_H)); args.explicit_hydrogen = expl_H

data_type = "GEOM" if "GEOM" in args.data else "QM9"

if args.augment_rotations:
    max_, min_ = np.max(x_grid), np.min(x_grid)
    no_points = x_grid.shape[0]
    y = np.linspace(min_, max_, no_points)
    z = np.linspace(min_, max_, no_points)
    x = np.linspace(min_, max_, no_points)

    x_grid,y_grid,z_grid = np.meshgrid(x,y,z, indexing='ij')


# * debugging the data
if args.debug_ds:
    for i in range(len(data)):
        print(i)
        item = data.__getitem__(i)
        if 'c' not in item[1]: continue
        breakpoint()
        mol = Chem.MolFromSmiles(item[1])
        if args.explicit_hydrogen: mol = Chem.AddHs(mol)
        elements = []
        for i in range(mol.GetNumAtoms()): elements.append(mol.GetAtomWithIdx(i).GetSymbol())

        visualize_failed_mols_w_hydrogens([[item[0], item[2], item[2], elements, item[3]]], title=item[1])
    breakpoint()

if args.model == 'cnn':
    model = CNN(channels=4 + args.explicit_hydrogen if args.no_sep_bnd_chn else 7 + args.explicit_aromatic + args.explicit_hydrogen).to(device)
elif args.model == 'DiT':
    model = DiT_S_2_3D(in_channels=4 + args.explicit_hydrogen if args.no_sep_bnd_chn else 7 + args.explicit_aromatic + args.explicit_hydrogen, depth=args.depth, hidden_size=args.hidden_size, num_classes=unique_classes, small_attention=args.small_attention, patch_size=args.patch_size, att_size=args.att_size).to(device)
else:
    print("\n\n\!!! HAVE CHANGED INPUT CHANNELS TO BE = C !!! go to line 200ish in train_class_fre_guid if it doesn't work")
    nom_factors = [torch.tensor(nf) for nf in norm_factors] if norm_factors is not None else None
    if args.large_mdl:
        # model = Unet3D(dim=128,num_classes=unique_classes, dim_mults = (1, 1, 2, 2), channels=4 + args.explicit_hydrogen if args.no_sep_bnd_chn else 7 + args.explicit_aromatic + args.explicit_hydrogen)
        model = Unet3D(dim=16,num_classes=unique_classes, dim_mults = (1, 1, 1, 1), channels=C, norm_factors=norm_factors, cond_var=args.cond_variable if not args.discrete_conditioning else [], 
                       remove_null_cond_emb=args.remove_null_cond_emb)

    else:
        # model = Unet3D(dim=128,num_classes=unique_classes, dim_mults = (1, 2, 3), channels=4 + args.explicit_hydrogen if args.no_sep_bnd_chn else 7 + args.explicit_aromatic + args.explicit_hydrogen)
        model = Unet3D(dim=128,num_classes=unique_classes, dim_mults = (1, 2, 3), channels=C, norm_factors=norm_factors, cond_var=args.cond_variable if not args.discrete_conditioning else [],
                       remove_null_cond_emb=args.remove_null_cond_emb)

noise_scheduler_conf = OmegaConf.load(args.noise_scheduler_conf) if args.noise_scheduler_conf is not None else None
if args.noise_scheduler_conf is not None:
    diffusion = GaussianDiffusionDiffSchds(
        model,
        image_size = (C,H,W,D),
        timesteps = args.ntimesteps,           # number of steps
        sampling_timesteps = args.ntimesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        beta_schedule=args.beta_schedule,
        blur=args.blur,
        noise_scheduler_conf=noise_scheduler_conf,
        objective=args.objective
    )
else:    
    diffusion = GaussianDiffusion(
        model,
        image_size = (C,H,W,D),
        timesteps = args.ntimesteps,           # number of steps
        sampling_timesteps = args.ntimesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        beta_schedule=args.beta_schedule,
        blur=args.blur,
        objective=args.objective
    )

trainer = Trainer(
    diffusion,
    data,
    # results_folder = str(cwd) + '/Models/',
    unique_elements = UNQ_elements.get(args.data),
    x=x_grid,
    y=y_grid,
    z=z_grid,
    smiles_list=Smiles[0] if train_valid_test else Smiles, # this is list of lists of smiles if I do train/val/test
    train_batch_size = args.batch_size,
    val_batch_size=args.val_batch_size,
    train_lr = args.lr,
    train_num_steps = 2_000_000,         # total training steps
    save_and_sample_every = args.test_every,
    gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,             # whether to calculate fid during training
    # grids = [x_grid, y_grid, z_grid]
    results_folder=model_path,
    run_name=args.run_name,
    sep_bond_chn= not args.no_sep_bnd_chn,
    valid_data=data_val,
    load_name=args.load_name,
    explicit_aromatic=args.explicit_aromatic,
    explicit_hydrogen=args.explicit_hydrogen,
    subsample_points=args.subsample_points,
    mixed_prec=args.mixed_prec,
    compact_batch=args.compact_batch,
    remove_thresholding_pdf_vals=args.remove_thresholding_pdf_vals,
    std_atoms=args.std_atoms,
    backward_mdl_compat=args.backward_mdl_compat,
    optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights,
    threshold_bond=args.threshold_bond,
    threshold_atm=args.threshold_atm,
    augment_rotations=args.augment_rotations,
    center_atm_to_grids=args.center_atm_to_grids,
    model_type = args.model,
    data_generator=data_generator,
    data_type=data_type,
    train_dataset_args=train_dataset_args,
    val_dataset_args=val_dataset_args,
    no_fields=C,
    unique_atms_considered=atoms_considered,
    arom_cycle_channel=args.arom_cycle_channel,
    multi_gpu=args.multi_gpu,
    data_path=data_path,
    fix_pi_values=args.fix_pi_values,
    data_file=args.data_file,
    remove_bonds=args.remove_bonds,
    inv_rot_loss_angles=args.inv_rot_loss_angles,
    only_min_activ=args.only_min_activ,
    cond_variables=args.cond_variable,
    atomno_conditioned_variable_sampler=atomno_conditioned_variable_sampler,
    norm_factors=norm_factors,
    discrete_conditioning=args.discrete_conditioning
)

trainer.train()
#loss = diffusion(torch.tensor([data[0]]).view(-1,C,H,W,D).float())
#print("Loss ", loss.item())
#loss.float().backward()




