from omegaconf import OmegaConf
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
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, CNN, Trainer, Unet3D, num_to_groups
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusionDiffSchds as GaussianDiffusionDiffSchdsNoGuid
from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion as GaussianDiffusionFreeGuidance, GaussianDiffusionDiffSchds
from denoising_diffusion_pytorch.classifier_free_guidance import Unet3D as Unet3DClsFreeGuid
import logging
from utils import get_bin_atm_upper_lims
from test_utils import fit_pred_field_sep_chn_batch, check_smiles_validity, extract_N_inds_from_atms,extract_positions_batch, check_mol_atm_stability, check_same_bond
from dit_model import DiT_S_2_3D




# helper funuctions to parse the result in a nice format
def parse_generated_mols(all_bnd_atm_smil,fields):
    results_ = []
    for atm_bnd_sm, field_bondPos_bonds_atms_atmCrds in zip(all_bnd_atm_smil,fields):
        mol_dict = dict()
        mol_dict['atoms'] = atm_bnd_sm[0]
        mol_dict['bonds'] = atm_bnd_sm[1]
        mol_dict['smiles'] = atm_bnd_sm[2]
        mol_dict['field'] = field_bondPos_bonds_atms_atmCrds[0]
        mol_dict['coords'] = field_bondPos_bonds_atms_atmCrds[4]
        results_.append(mol_dict)
    return results_

def parse_batches(res_dict):
    all_results = []
    for all_bnd_atm_smil_batch,fields_batch in zip(res_dict[3],res_dict[4]):
        all_results.extend(parse_generated_mols(all_bnd_atm_smil_batch,fields_batch))
    return all_results


parser = argparse.ArgumentParser('FieldGen')
parser.add_argument('--model', type=str, default="model-8res_0315_qm9edmrun.pt")
parser.add_argument('--num_imgs', type=int, default=300)
parser.add_argument('--resolution', type=float, default=0.33)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument("--explicit_aromatic", default=False, action='store_true')
parser.add_argument("--explicit_hydrogen", default=False, action='store_true')
parser.add_argument("--data", default="qm9edm",type=str)
parser.add_argument("--timesteps", default=1000, type=int)
parser.add_argument("--cls_free_guid", default=False, action="store_true")
parser.add_argument('--beta_schedule', default='linear', type=str)
parser.add_argument('--discard_fields', default=False, action="store_true")
parser.add_argument('--run_name', default="", type=str)
parser.add_argument("--cond_scale", default=3., type=float)
parser.add_argument("--optimize_bnd_gmm_weights", default=False, action="store_true")
parser.add_argument("--threshold_atm", default=0.75, type=float)
parser.add_argument("--threshold_bond", default=0.75, type=float)
parser.add_argument("--blur", default=False, action="store_true")
parser.add_argument("--pi_threshold", default=0.5, type=float)
parser.add_argument("--model_type", default="unet", type=str)
parser.add_argument("--small_mdl", default=False, action="store_true")
parser.add_argument("--add_pe" ,default=False, action="store_true")
parser.add_argument("--save_all_imgs" ,default=False, action="store_true")
parser.add_argument("--legacy_attention", default=False, action="store_true", help="Create mid attention using dim heads 32 and 4 heads")
parser.add_argument("--augment_rotations", default=False, action="store_true")
parser.add_argument("--noise_scheduler_conf", default=None, type=str, help="load a noise scheduler configuration from a yaml file")
parser.add_argument("--data_paths", default=None, type=str, help="specify paths to data files for each resolution")
parser.add_argument("--data_type" ,default= "QM9", choices=["QM9", "GEOM"], type=str, help="specify paths to data files for each resolution")
parser.add_argument("--retest_w_atm_dist", default=False, action="store_true", help="retest with atom distance")
parser.add_argument("--test_w_atm_dist", default=False, action="store_true", help="retest with atom distance")
parser.add_argument("--pos_optimize_bs", default=10,type=int)
parser.add_argument("--objective", default='pred_noise', type=str, help="specify objective for training")
parser.add_argument("--cond_variable", default=[], nargs="+", help="specify a conditioning parameter for conditional generation")
parser.add_argument("--use_original_atm_pos", default=True, type=bool)
parser.add_argument("--optimize_atom_positions", default=False, action="store_true")
parser.add_argument("--test_cond_gen_on_atom_no", default=None, type=str)
parser.add_argument("--discrete_conditioning", default=False, action="store_true")
parser.add_argument("--remove_seed", default=False, action="store_true")
parser.add_argument("--extra_strct_chn", default=False, action="store_true", help="Add extra `agnostic` atom channel")


# * Dit model arguments
parser.add_argument("--depth", default=4, type=int, help="depth of DiT architecture")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden dimension")
parser.add_argument("--small_attention", default=False, action='store_true', help="use restrcited attention into 5x5 grids")
parser.add_argument("--patch_size", default=2, type=int, help="patch size for attention")

args = parser.parse_args()


logging.getLogger('some_logger')
logging.basicConfig(filename=args.model.replace(".pt", ".logging"), level=logging.INFO, force=True)


if args.data_paths is None:
    path_conf_yml_files = [f for f in os.listdir("./") if "yml" in f]
    args.data_paths = path_conf_yml_files[0]
    print("!!!WARNING!!! NO DATA PATHS CONFIG SPECIFIED. USING {}".format(args.data_paths))


data_path_conf = OmegaConf.load(args.data_paths)
model_path, data_path = data_path_conf.model_path, data_path_conf.data_path
if "QM9" in args.data_type.upper(): data_path = os.path.join(data_path, "qm9/data/")
elif "GEOM" in args.data_type.upper(): data_path = os.path.join(data_path, "geom_data")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_ema=True
num_images=args.num_imgs
batch_size=args.batch_size
mdl = args.model
data = args.data.replace(".bin", "")+'.bin'

if args.run_name:
    save_results_path = os.path.join(model_path, args.run_name + "_ema_results.pkl" if load_ema else args.run_name + "_results.pkl")
else:
    save_results_path = os.path.join(model_path, mdl.replace(".pt","ema_results.pkl" if load_ema else "results.pkl") )
if os.path.exists(mdl):
    model_file = mdl
else:    
    model_file = os.path.join(os.path.join(model_path),mdl)
# model_file = "Models/small_qm9edm_0315_mdls/model-12res_0315_qm9edm_small_run_explicit_aromatic.pt"
# model_file = "Models/small_qm9edm_0315_mdls/model-12res_0315_qm9edm_small_run.pt"

if "geom_data" in data:
    all_atom_symbols,all_bonds = [], []
    grid_lims_ = {'min_min_x': 0, 'min_min_y': 0, 'min_min_z': 0, 'max_max_x': 0, 'max_max_y': 0, 'max_max_z': 0}
    for i in range(1,31):
        if i == 1: del_inds, grid_lims= pickle.load(open(os.path.join(data_path,"conf_{}_limits.bin".format(i)), "rb"))
        else: _, grid_lims= pickle.load(open(os.path.join(data_path,"conf_{}_limits.bin".format(i)), "rb"))
        for k,v in grid_lims.items():
            if "min" in k: grid_lims_[k] = min(grid_lims_[k], v)
            else: grid_lims_[k] = max(grid_lims_[k], v)

    x_lim,y_lim,z_lim = {"min":grid_lims_['min_min_x'], "max":grid_lims_['max_max_x']}, {"min":grid_lims_['min_min_y'], "max":grid_lims_['max_max_y']}, {"min":grid_lims_['min_min_z'], "max":grid_lims_['max_max_z']}
    resolution = args.resolution
    bin_numbers = [*map(lambda m: int(np.ceil((m.get("max")-m.get("min"))/(resolution*8))*8), [x_lim,y_lim,z_lim])]
    x_grid,y_grid,z_grid = np.meshgrid( *map(lambda bin_n,lims: np.linspace(lims.get("min"),lims.get("max"),bin_n),bin_numbers, [x_lim, y_lim, z_lim]), indexing='ij')
        
    splits = pickle.load(open(os.path.join(data_path, "splits.pkl"), "rb"))
    del_inds.extend(splits['val'])
    del_inds.extend(splits['test'])

    all_smiles,all_atom_symbols,all_bonds = [],[],[]
    # all_mol_graphs = pickle.load(open(os.path.join(data_path, "geom_conf1_aligned.bin"), "rb")) # conf_1 contains all molecules

    all_atms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'H'] if args.explicit_hydrogen else ['C', 'N', 'O', 'F', 'P', 'S', 'Cl']
    C = len(all_atms) + args.explicit_aromatic + 3
    mol_info = pd.read_csv(os.path.join(data_path, "mol_summary.csv"))
    smiles = mol_info['smiles']
    kept_inds = set(list(range(len(smiles)))) - set(del_inds)
    all_smiles = [smiles[i] for i in kept_inds]
else:
    data_file = os.path.join(data_path,data) if "/" not in data else data
    data = pickle.load(open(data_file, 'rb'))


    if args.cond_variable: x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds,cond_vars = pickle.load(open(data_file, "rb"))
    else: x_grid, y_grid, z_grid, all_coords, all_smiles,all_atom_symbols,all_bonds,_ = pickle.load(open(data_file, "rb"))

    
    
    all_atms = set(all_atom_symbols['train'][0])
    for atm_s in all_atom_symbols['train']: all_atms.update(atm_s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state_dict = torch.load(model_file, map_location=device)

C = len(all_atms) + args.explicit_aromatic + 3 if not args.test_w_atm_dist else len(all_atms)
if 'data_file' in globals() and "simple_chiral" in data_file: C+=1
if args.augment_rotations: x_grid,y_grid,z_grid = max_axis(x_grid,y_grid,z_grid)

# C = 7 + args.explicit_aromatic + args.explicit_hydrogen


rmv = ('X'  in all_atms) + ('Y' in all_atms)

H,W,D = x_grid.shape

noise_scheduler_conf = OmegaConf.load(args.noise_scheduler_conf) if args.noise_scheduler_conf is not None else None
if args.cond_variable and args.discrete_conditioning:
    #! NOT IMPLEMENTED 
    #! SAVING THE COND VARS
    #! YET!!!!

    data_file = "geom_conf1.bin" if "geom_data" in args.data else args.data+".bin"
    data =  pickle.load(open(os.path.join(data_path,data_file), "rb"))
    cond_variable = data[7]['train']
    bin_limits_2_classes = discretize_cond_var(cond_var=cond_variable)
    sampled_cond_var = pickle.load(open(f"test_cond_props/sampled_properties_{args.num_imgs}_alpha.pkl", "rb"))[1]

    cond_classes = get_cond_var_bin_cls(sampled_cond_var, bin_limits_2_classes)

    classes = torch.tensor(cond_classes).to(device)
    score_model = Unet3DClsFreeGuid(dim=128,num_classes=len(bin_limits_2_classes), dim_mults = (1, 2, 3), channels=C + args.extra_strct_chn, add_pe=args.add_pe)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)
    else:
        model = GaussianDiffusionFreeGuidance(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                              timesteps=args.timesteps, loss_type='l2', blur=args.blur, objective=args.objective)
    cond_var = None   
elif args.cond_variable:
    data_file = args.data
    args.center_atm_to_grids, args.consider_arom_chns_as_atms, args.fix_chiral_conf,args.no_sep_bnd_chn,args.data_file, args.no_limits,args.force_create_data, args.compact_batch, args.remove_bonds,args.std_atoms,args.debug_ds, args.subsample_points, args.mixed_prec, args.arom_cycle_channel = False, False, None, False, data_file, False, False, True, False, 0.05, False, False, False, False
    args.data = "QM9"
    Smiles,atoms,atom_pos,cond_variable = retrieve_smiles(args,True,data_path)
    args.data = data_file
    data, data_val, C, H, W, D, x_grid, y_grid, z_grid, unique_classes = setup_data_files(args, Smiles, True,atoms,atom_pos, add_atom_num_cls=True, center_atm_to_grids=args.center_atm_to_grids, data_path=data_path, consider_arom_chns_as_atms=args.consider_arom_chns_as_atms, fix_chiral_conf=args.fix_chiral_conf, cond_variable=cond_variable)
    norm_factors = get_cond_var_normalizing_factors(data, args)
    atomno_conditioned_variable_sampler = ConditioningVarSampler(data, norm_factors) 

    data_file = "geom_conf1.bin" if "geom_data" in args.data else args.data+".bin"
    if not args.remove_seed: np.random.seed(42)

    classes, cond_var = atomno_conditioned_variable_sampler.sample(args.num_imgs)
    classes, cond_var = torch.tensor(classes).to(device), torch.tensor(cond_var)

    pickle.dump([classes, cond_var], open(save_results_path.replace(".pkl", "") + "_cond_vars_and_cls.pkl", "wb"))

    score_model = Unet3DClsFreeGuid(dim=128,num_classes=unique_classes, dim_mults = (1, 2, 3), channels=C + args.extra_strct_chn, add_pe=args.add_pe, cond_var=args.cond_variable)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)
    else:
        model = GaussianDiffusionFreeGuidance(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                              timesteps=args.timesteps, loss_type='l2', blur=args.blur, objective=args.objective)
elif args.model_type == "DiT":
    bins_upper_lim, bin_counts = get_bin_atm_upper_lims(file=args.data + ".bin", return_counts=True, data_path=data_path)
    bins_upper_lim.insert(0,1)
    bins = [[bins_upper_lim[i], bins_upper_lim[i+1]] for i in range(len(bins_upper_lim)-1)]
    unique_classes = len(bins)
    all_atms_no = sum(bin_counts.values())
    cls_probs = {k:(v/all_atms_no) for k,v in bin_counts.items()}
    if not args.remove_seed: np.random.seed(42)
    classes = torch.tensor(np.random.choice(list(cls_probs.keys()), args.num_imgs, p=list(cls_probs.values()))).to(device)
    score_model = Unet3DClsFreeGuid(dim=128, num_classes=unique_classes, dim_mults = (1, 1, 2, 2), 
                         channels=C+ args.extra_strct_chn)
    score_model =  DiT_S_2_3D(in_channels=C, depth=args.depth, hidden_size=args.hidden_size, 
                              num_classes=unique_classes, small_attention=args.small_attention, 
                              patch_size=args.patch_size).to(device)

    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                            timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)        
    else:
        model = GaussianDiffusionFreeGuidance(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                            timesteps=args.timesteps, loss_type='l2', blur=args.blur, objective=args.objective)
    cond_var = None        
elif args.cls_free_guid and not args.small_mdl:
    bins_upper_lim, bin_counts = get_bin_atm_upper_lims(file=args.data + ".bin", return_counts=True, data_path=data_path)
    bins_upper_lim.insert(0,1)
    bins = [[bins_upper_lim[i], bins_upper_lim[i+1]] for i in range(len(bins_upper_lim)-1)]
    unique_classes = len(bins)
    all_atms_no = sum(bin_counts.values())
    cls_probs = {k:(v/all_atms_no) for k,v in bin_counts.items()}
    if not args.remove_seed: np.random.seed(42)
    classes = torch.tensor(np.random.choice(list(cls_probs.keys()), args.num_imgs, p=list(cls_probs.values()))).to(device)
    score_model = Unet3DClsFreeGuid(dim=128, num_classes=unique_classes, dim_mults = (1, 1, 2, 2), 
                         channels=C+ args.extra_strct_chn, legacy_attention=args.legacy_attention)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)
    else:
        model = GaussianDiffusionFreeGuidance(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                            timesteps=args.timesteps, loss_type='l2', blur=args.blur, objective=args.objective)
    cond_var = None
elif args.cls_free_guid and args.small_mdl:
    data_file = "geom_conf1.bin" if "geom_data" in args.data else args.data.replace(".bin", "")+".bin"
    bins_upper_lim, bin_counts = get_bin_atm_upper_lims(file=data_file, return_counts=True, data_path=data_path, geom="geom_data" in args.data, explicit_h=args.explicit_hydrogen)
    bins_upper_lim.insert(0,1)
    bins = [[bins_upper_lim[i], bins_upper_lim[i+1]] for i in range(len(bins_upper_lim)-1)]
    unique_classes = len(bins)
    all_atms_no = sum(bin_counts.values())
    cls_probs = {k:(v/all_atms_no) for k,v in bin_counts.items()}
    if not args.remove_seed: np.random.seed(42)
    # test non-conditional generation on model of N_a ~ p(N_a | c)
    if args.test_cond_gen_on_atom_no is not None:
        classes, cond_var  = pickle.load(open(f"test_cond_props/sampled_properties_{args.num_imgs}_{args.test_cond_gen_on_atom_no}.pkl", "rb"))
        bins_conditional = pickle.load(open(f"test_cond_props/atom_no_cls_bins_{args.test_cond_gen_on_atom_no}.pkl", "rb"))
        classes = [np.argwhere(bins_conditional[c][1]>= np.array(bins_upper_lim))[-1,0]-1 for c in classes]
        classes = torch.tensor(classes).to(device)
    else:
        classes = torch.tensor(np.random.choice(list(cls_probs.keys()), args.num_imgs, p=list(cls_probs.values()))).to(device)
    score_model = Unet3DClsFreeGuid(dim=128,num_classes=unique_classes, dim_mults = (1, 2, 3), channels=C+ args.extra_strct_chn, add_pe=args.add_pe)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)
    else:
        model = GaussianDiffusionFreeGuidance(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                              timesteps=args.timesteps, loss_type='l2', blur=args.blur, objective=args.objective)
    cond_var = None        
elif not args.cls_free_guid:
    score_model = Unet3D(dim=128, dim_mults = (1, 2, 3), channels=C+ args.extra_strct_chn, add_pe=args.add_pe)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchdsNoGuid(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf)
    else:
        model = GaussianDiffusion(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                              timesteps=args.timesteps, loss_type='l2', blur=args.blur)
    cond_var = None
else:
    score_model = Unet3D(dim=128, dim_mults=(1, 1, 2, 2), channels=C+ args.extra_strct_chn)
    if noise_scheduler_conf is not None:
        model = GaussianDiffusionDiffSchds(score_model, image_size=(C+args.extra_strct_chn,H,W,D), beta_schedule=args.beta_schedule,
                                          timesteps=args.timesteps, loss_type='l2', blur=args.blur, noise_scheduler_conf=noise_scheduler_conf, objective=args.objective)
    else:
        model = GaussianDiffusion(score_model, image_size=[C+args.extra_strct_chn,*x_grid.shape], timesteps=args.timesteps,
                                beta_schedule=args.beta_schedule, objective=args.objective)
    cond_var = None

model.sampling_timesteps = args.timesteps
model.is_ddim_sampling = False
state_dict = torch.load(model_file)
if load_ema:
    mdl_state = {param_n.replace('ema_model.',''):param for param_n,param in state_dict['ema'].items() if 'online' not in param_n and 'initted' not in param_n and 'step' not in param_n}
else:
    mdl_state = state_dict['model']


def remove_noise_params(mdl_state, model):
    noise_params = ["all_betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod","log_one_minus_alphas_cumprod","sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod","posterior_variance","posterior_log_variance_clipped",
    "posterior_mean_coef1","posterior_mean_coef2", "p2_loss_weight"]
    train_inf_tmstp_fact = None
    if model.all_betas.shape[1] != mdl_state['all_betas'].shape[1]:
        train_inf_tmstp_fact = model.all_betas.shape[1]/mdl_state['all_betas'].shape[1]
        print("Removing noise parameters from model state dict. You are generating"
              "using a different number of timesteps than training ")
        for n_p in noise_params:
            mdl_state.pop(n_p)
    return mdl_state, train_inf_tmstp_fact
    
mdl_state, train_inf_tmstp_fact = remove_noise_params(mdl_state, model)
# model.all_betas.shape

model.load_state_dict(mdl_state, strict=False)

model = model.to(device)

if train_inf_tmstp_fact is not None:
    model.model.train_inf_tmstp_fact = train_inf_tmstp_fact

model.eval()


if args.test_w_atm_dist:
    batches = num_to_groups(num_images, batch_size)
    tot_val, tot_unq, tot_nov = 0,0,0
    for ind, b in enumerate(batches):
        with torch.no_grad():
            if args.save_all_imgs and ind == 0: 
                all_images, all_images_all_timesteps = model.sample(classes=classes[ind*b:(ind+1)*b], cond_scale=args.cond_scale, save_all_imgs=args.save_all_imgs)
                pickle.dump(all_images_all_timesteps, open(save_results_path.replace(".pkl", "_all_images_all_timesteps.pkl"), 'wb'))
            elif args.cls_free_guid: 
                all_images = model.sample(classes=classes[ind*b:(ind+1)*b], cond_scale=args.cond_scale, save_all_imgs=False)
            else: all_images = model.sample(batch_size=b,return_specific_timesteps=None)
                # if return_smiles and return_atm_pos_bdns: return total_equal_or_valid/total,  unq, nov, all_generated_atm_bnd_smils, all_fields, generated_smiles
        
        # below is for the extra structure channel
        if args.extra_strct_chn: all_images = [tr_imgs[:,:-1] for tr_imgs in all_images]
        
        # below is for extra aromatic channels
        if rmv: all_images_original = copy.deepcopy(all_images); all_images = all_images[:,:-rmv]
        data_smiles = all_smiles['train'] if "QM9" in args.data_type else all_smiles
        atm_symb_bnd_by_channel=all_atms if "GEOM" in args.data_type else None
        print(all_images[0].shape)
        val, unq, nov, all_bnd_atm_smil,failed_fields, bond_weights = fit_pred_field_sep_chn_batch(all_images, 0.1, x_grid, y_grid, z_grid, None, 
                                            None, data_smiles,noise_std=0.0, normalize01=True,return_atm_pos_bdns=True,
                                            explicit_aromatic=args.explicit_aromatic, explicit_hydrogen=args.explicit_hydrogen,
                                            discard_fields=args.discard_fields, optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights,
                                            threshold_bond=args.threshold_bond, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold,
                                            atm_symb_bnd_by_channel=atm_symb_bnd_by_channel,atm_dist_based=args.test_w_atm_dist)
        tot_val,tot_unq,tot_nov = tot_val+val,tot_unq+unq,tot_nov+nov
        print(all_bnd_atm_smil)
        no_fields = ind*b+b
        print("After {} samples, val/unq/nov are {}/{}/{}".format(no_fields,tot_val/(ind+1), tot_unq/(ind+1), tot_nov/(ind+1)))
        logging.info("After {} samples, val/unq/nov are {}/{}/{}".format(no_fields,tot_val/(ind+1), tot_unq/(ind+1), tot_nov/(ind+1)))
        results = pickle.load(open(save_results_path, 'rb')) if os.path.exists(save_results_path) else [[],[],[],[], []]

        results[0].append(val)
        results[1].append(unq)
        results[2].append(nov)
        results[3].append(all_bnd_atm_smil)
        if rmv:
            failed_fields_original_imgs = []
            for f, orig_img in zip(failed_fields, all_images_original):
                f[0] = orig_img.detach().cpu().numpy() if not args.discard_fields else None
                failed_fields_original_imgs.append(f)
            results[4].append(failed_fields_original_imgs)
        else:
            results[4].append(failed_fields)

        resulted_weights = pickle.load(open(save_results_path.replace(".pkl", "_weights.pkl"), 'rb')) if os.path.exists(save_results_path.replace(".pkl", "_weights.pkl")) else []
        resulted_weights = list(resulted_weights)
        resulted_weights.extend(bond_weights)
        resulted_weights = np.array(resulted_weights)
        pickle.dump(resulted_weights, open(save_results_path.replace(".pkl", "_weights.pkl"), 'wb'))


        pickle.dump(results,open(save_results_path, 'wb'))
            
    print(val,unq,nov)
elif args.retest_w_atm_dist:
    data = pickle.load(open("data/qm9/data/"+ args.data + ".bin", "rb"))
    x_grid, y_grid, z_grid = data[0], data[1], data[2]
    if args.run_name is not None:
        result_file = "misc/"+args.run_name + "_ema_results.pkl"
    else:
        result_file = "misc/"+args.model.replace("data/", "").replace(".pt", "ema_results.pkl")

    if "{}" in result_file: 
        n_files = 0 
        while os.path.exists(result_file.format(n_files)): n_files+=1
        all_results = [[],[],[],[],[]]
        for i in range(n_files):
            result_file = result_file.format(i)
            current_results = pickle.load(open(result_file, 'rb'))
            for j in range(5): all_results[j].extend(current_results[j])
        results = all_results
    else:
            
        results = pickle.load(open(result_file, 'rb'))
    all_atms, all_bonds, all_smiles, atom_positions, all_fields = [], [], [], [], []
    no_atms = 5 if args.explicit_hydrogen else 4

    for batch in results[3]:
        for mol_grph_info in batch:
            all_atms.append(mol_grph_info[0])
            all_bonds.append(mol_grph_info[1])
            all_smiles.append(mol_grph_info[2])
    for batch in results[4]:
        for mol_grph_info in batch:
            if mol_grph_info[0] is not None: all_fields.append(mol_grph_info[0][:no_atms])
            else: all_fields.append(None)
            atom_positions.append(mol_grph_info[4])
    N_lists = extract_N_inds_from_atms(all_atms ,explicit_hydrogen=args.explicit_hydrogen)

    batch_inds = [[i*args.pos_optimize_bs, (i+1)*args.pos_optimize_bs] for i in range(np.ceil(len(all_atms)/args.pos_optimize_bs).astype(int))]
    all_updated_positions = []
    # for ind, (st_ind, end_ind) in enumerate(batch_inds):
    #     print("Optimizing positions for batch {}/{}".format(ind+1, len(batch_inds)))
    #     end_ind = min(end_ind, len(all_atms))
    #     n_list = np.array(N_lists[st_ind*5: end_ind*5]).tolist()
    #     flds = [all_fields[i] for i in range(st_ind, end_ind)]
    #     atm_pstns = [atom_positions[i] for i in range(st_ind, end_ind)]
    #     updated_positions = extract_positions_batch(flds, x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), n_list, std=0.05, initial_guess=atm_pstns)

    #     # * all updated atom positions from all samples are put together - need to separate them
    #     total_atm_per_mol = [sum(n_list[5*i:5*(i+1)]) for i in range(len(n_list)//5) ]
    #     atom_pos_inds = [sum(total_atm_per_mol[:i+1]) for i in range(len(total_atm_per_mol))]
    #     atom_pos_inds.insert(0,0)

    #     for i in range(len(atom_pos_inds)-1): all_updated_positions.append(updated_positions[atom_pos_inds[i]:atom_pos_inds[i+1]].detach().cpu().numpy())

    valid, tot, all_mol_smiles, atm_bnd_smil,fields_info, _,corresp_weights = check_smiles_validity(atom_positions, all_atms, explicit_hydrogen=args.explicit_hydrogen, 
                                                                                     x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, optimize_positions=False,
                                                                                     field=all_fields)
    all_field_info = []
    for atm_bnd_smil_, atm_p in zip(atm_bnd_smil, atom_positions):
        bnd = atm_bnd_smil_[1]
        atm = atm_bnd_smil_[0]
        all_field_info.append([None, [(bnd[i][0], bnd[i][1], atm_p[bnd[i][0]], atm_p[bnd[i][1]]) for i in range(len(bnd))], 
                               [np.array(bnd[i]) for i in range(len(bnd))],  atm,  atm_p ])
    pickle.dump([[valid/tot],[1],[1], [atm_bnd_smil], [all_field_info]], open(result_file.replace("ema_results.pkl", "bond_dist_results.pkl"), 'wb'))
    breakpoint()
    dist_based_f_info = [atm_bnd_smil]

    breakpoint()

    all_mol_atms,all_bonds_dist_based, all_smiles_dist_based = check_smiles_validity(atom_positions, all_atms, explicit_hydrogen=args.explicit_hydrogen, 
                                                                                     x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, optimize_positions=False,
                                                                                     field=all_fields)
    
    all_mol_atms,all_bonds_dist_based, all_smiles_dist_based = check_smiles_validity(atom_positions, all_atms, explicit_hydrogen=args.explicit_hydrogen)
    check_same_bond(all_atms, all_bonds, all_bonds_dist_based)
    breakpoint()

    check_mol_atm_stability()
elif args.retest_w_atm_dist:
    result_file = "misc/"+args.model.replace("data/", "").replace(".pt", "ema_results.pkl")
    results = pickle.load(open(result_file, 'rb'))
    all_atms, all_bonds, all_smiles, atom_positions, all_fields = [], [], [], [], []
    no_atms = 5 if args.explicit_hydrogen else 4

    for batch in results[3]:
        for mol_grph_info in batch:
            all_atms.append(mol_grph_info[0])
            all_bonds.append(mol_grph_info[1])
            all_smiles.append(mol_grph_info[2])
    for batch in results[4]:
        for mol_grph_info in batch:
            all_fields.append(mol_grph_info[0][:no_atms])
            atom_positions.append(mol_grph_info[4])
    N_lists = extract_N_inds_from_atms(all_atms ,explicit_hydrogen=args.explicit_hydrogen)

    batch_inds = [[i*args.pos_optimize_bs, (i+1)*args.pos_optimize_bs] for i in range(np.ceil(len(all_atms)/args.pos_optimize_bs).astype(int))]
    all_updated_positions = []
    for ind, (st_ind, end_ind) in enumerate(batch_inds):
        print("Optimizing positions for batch {}/{}".format(ind+1, len(batch_inds)))
        end_ind = min(end_ind, len(all_atms))
        n_list = np.array(N_lists[st_ind*5: end_ind*5]).tolist()
        flds = [all_fields[i] for i in range(st_ind, end_ind)]
        atm_pstns = [atom_positions[i] for i in range(st_ind, end_ind)]
        updated_positions = extract_positions_batch(flds, x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), n_list, std=0.05, initial_guess=atm_pstns)

        # * all updated atom positions from all samples are put together - need to separate them
        total_atm_per_mol = [sum(n_list[5*i:5*(i+1)]) for i in range(len(n_list)//5) ]
        atom_pos_inds = [sum(total_atm_per_mol[:i+1]) for i in range(len(total_atm_per_mol))]
        atom_pos_inds.insert(0,0)

        for i in range(len(atom_pos_inds)-1): all_updated_positions.append(updated_positions[atom_pos_inds[i]:atom_pos_inds[i+1]])
    all_mol_atms,all_bonds_dist_based, all_smiles_dist_based = check_smiles_validity(all_updated_positions, all_atms, explicit_hydrogen=args.explicit_hydrogen)
    
    all_mol_atms,all_bonds_dist_based, all_smiles_dist_based = check_smiles_validity(atom_positions, all_atms, explicit_hydrogen=args.explicit_hydrogen)
    check_same_bond(all_atms, all_bonds, all_bonds_dist_based)
    breakpoint()

    check_mol_atm_stability()
else:
    print("Note. If you generate multiple times with the same save_results_path, results will be appended to the same file.")
    batches = num_to_groups(num_images, batch_size)
    tot_val, tot_unq, tot_nov = 0,0,0
    all_removed_bonds = []
    for ind, b in enumerate(batches):
        with torch.no_grad():
            if args.save_all_imgs and ind == 0: 
                all_images, all_images_all_timesteps = model.sample(classes=classes[ind*b:(ind+1)*b], cond_scale=args.cond_scale, save_all_imgs=args.save_all_imgs)
                pickle.dump(all_images_all_timesteps, open(save_results_path.replace(".pkl", "_all_images_all_timesteps.pkl"), 'wb'))
            elif args.cls_free_guid: 
                classes_= classes[ind*b:(ind+1)*b]
                if cond_var is not None:
                    cond_var_ = (torch.tensor(cond_var[ind*b:(ind+1)*b]) - norm_factors[0])/(norm_factors[1]-norm_factors[0])
                    cond_var_ = cond_var_.to(dtype=torch.float32)
                else: cond_var_ = None
                all_images = model.sample(classes=classes_, 
                            cond_scale=args.cond_scale, 
                            save_all_imgs=False, 
                            cond_var = cond_var_,)
            else: all_images = model.sample(batch_size=b,return_specific_timesteps=None)
                # if return_smiles and return_atm_pos_bdns: return total_equal_or_valid/total,  unq, nov, all_generated_atm_bnd_smils, all_fields, generated_smiles
        if rmv: all_images_original =copy.deepcopy(all_images); all_images = all_images[:,:-rmv]
        data_smiles = all_smiles['train'] if "QM9" in args.data_type else all_smiles
        atm_symb_bnd_by_channel=all_atms if "GEOM" in args.data_type else None
        print(all_images[0].shape)
        
        val, unq, nov, all_bnd_atm_smil,failed_fields, bond_weights, removed_bonds = fit_pred_field_sep_chn_batch(all_images, 0.1, x_grid, y_grid, z_grid, None, 
                                            None, data_smiles,noise_std=0.0, normalize01=True,return_atm_pos_bdns=True,
                                            explicit_aromatic=args.explicit_aromatic, explicit_hydrogen=args.explicit_hydrogen,
                                            discard_fields=args.discard_fields, optimize_bnd_gmm_weights=args.optimize_bnd_gmm_weights,
                                            threshold_bond=args.threshold_bond, threshold_atm=args.threshold_atm, pi_threshold=args.pi_threshold,
                                            atm_symb_bnd_by_channel=atm_symb_bnd_by_channel, optimize_atom_positions=args.optimize_atom_positions)
        tot_val,tot_unq,tot_nov = tot_val+val,tot_unq+unq,tot_nov+nov
        no_fields = ind*b+b
        print("After {} samples, val/unq/nov are {}/{}/{}".format(no_fields,tot_val/(ind+1), tot_unq/(ind+1), tot_nov/(ind+1)))
        logging.info("After {} samples, val/unq/nov are {}/{}/{}".format(no_fields,tot_val/(ind+1), tot_unq/(ind+1), tot_nov/(ind+1)))
        results = pickle.load(open(save_results_path, 'rb')) if os.path.exists(save_results_path) else [[],[],[],[], []]

        results[0].append(val)
        results[1].append(unq)
        results[2].append(nov)
        results[3].append(all_bnd_atm_smil)
        if rmv:
            failed_fields_original_imgs = []
            for f, orig_img in zip(failed_fields, all_images_original):
                f[0] = orig_img.detach().cpu().numpy() if not args.discard_fields else None
                failed_fields_original_imgs.append(f)
            results[4].append(failed_fields_original_imgs)
        else:
            results[4].append(failed_fields)
        all_removed_bonds.extend(removed_bonds)

        resulted_weights = pickle.load(open(save_results_path.replace(".pkl", "_weights.pkl"), 'rb')) if os.path.exists(save_results_path.replace(".pkl", "_weights.pkl")) else []
        resulted_weights = list(resulted_weights)
        resulted_weights.extend(bond_weights)
        resulted_weights = np.array(resulted_weights)
        pickle.dump(resulted_weights, open(save_results_path.replace(".pkl", "_weights.pkl"), 'wb'))
        pickle.dump(all_removed_bonds, open(save_results_path.replace(".pkl", "_rmvd_bnds.pkl"), 'wb'))


        pickle.dump(results,open(save_results_path, 'wb'))

        clean_results = parse_batches(results)
        pickle.dump(clean_results,open(save_results_path.replace(".pkl", "_clean_results.bin"), 'wb'))
        if ind == len(batches)-1:
            print(f"\033[92mResults saved to {save_results_path.replace('.pkl', '_clean_results.bin')}\033[0m")
            print("format is a list of dictionaries (one per molecule) with keys:\n" \
                "'atoms': list of #atoms characters,\n"\
                "'bonds':list of np arrays containing [atm_index1, atm_index2, bond_type]. Atom indices correspond to 'atoms' list\n"\
                "'smiles': RDKit parsed smiles string (when the mol is valid),\n"\
                "'field': numpy array of shape [#bonds, x, y, z],\n"\
                "'coords': numpy array of shape [#atoms, 3] containing atom coordinates")
            
    