import argparse

from torch.utils.data import Dataset
import torch
import numpy as np
import urllib.request
import urllib
from torch.nn.utils.rnn import pad_sequence
import tarfile
from math import inf

import os
charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)



def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass

def is_int(str):
    try:
        int(str)
        return True
    except:
        return False
    

def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    print('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = os.path.join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


    
def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    print('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = os.path.join(gdb9dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits




def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    print('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        for key, val in molecules.items():
            # now I also ahve the smiles (which are strings which should not be padded)
            if type(val[0]) == str:
                molecules[key] = val
            elif val[0].dim() > 0:
                molecules[key] = pad_sequence(val, batch_first=True)
            else:
                molecules[key] = torch.stack(val)
        # molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}
    return molecules

def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts

def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data

def download_dataset_qm9(datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = os.path.join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)

    print('Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(gdb9dir))

    print('Beginning download of GDB9 dataset!')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = os.path.join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_file = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_data =
    # tardata = tarfile.open(gdb9_tar_file, 'r')
    # files = tardata.getmembers()
    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    print('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(
            gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True)

    # Subtract thermochemical energy if desired.
    if calculate_thermo:
        # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
        therm_energy = get_thermo_dict(gdb9dir, cleanup)

        # For each of train/validation/test split, add the thermochemical energy
        for split_idx, split_data in gdb9_data.items():
            gdb9_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed GDB9 data into train/validation/test splits
    print('Saving processed data:')
    for split, data in gdb9_data.items():
        savedir = os.path.join(gdb9dir, split+'.npz')
        np.savez_compressed(savedir, **data)

    print('Processing/saving complete!')


def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]


    mol_smile_nostereo = xyz_lines[num_atoms+3].split()[0]
    mol_smile_wstereo = xyz_lines[num_atoms+3].split()[1]

    mol_smile = {'mol_wo_stereo':mol_smile_nostereo, 'mol_w_stereo':mol_smile_wstereo}

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule.update(mol_smile)
    molecule = {key: torch.tensor(val) if type(val)!= str else val for key, val in molecule.items()}

    return molecule


def prepare_dataset(datadir, dataset, subset=None, splits=None, cleanup=True, force_download=False):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces a fresh download of the dataset.

    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data. 

    Notes
    -----
    TODO: Delete the splits argument?
    """

    # If datasets have subsets,
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    # Assume one data file for each split
    datafiles = {split: os.path.join(
        *(dataset_dir + [split + '.npz'])) for split in split_names}

    # Check datafiles exist
    datafiles_checks = [os.path.exists(datafile)
                        for datafile in datafiles.values()]

    # Check if prepared dataset exists, and if not set flag to download below.
    # Probably should add more consistency checks, such as number of datapoints, etc...
    new_download = False
    if all(datafiles_checks):
        print('Dataset exists and is processed.')
    elif all([not x for x in datafiles_checks]):
        # If checks are failed.
        new_download = True
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    # If need to download dataset, pass to appropriate downloader
    if new_download or force_download:
        print('Dataset does not exist. Downloading!')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, splits, cleanup=cleanup)
        else:
            raise ValueError(
                'Incorrect choice of dataset! Must chose qm9/md17!')

    return datafiles


def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, 'qm9', subset, splits, force_download=force_download)


    return 
    

def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            print('The number of species is not the same in all datasets!')
        else:
            print('Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                print('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                print('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                print('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}


def setup_shared_args(parser):
    """
    Sets up the argparse object for the qm9 dataset
    
    Parameters 
    ----------
    parser : :class:`argparse.ArgumentParser`
        Argument Parser with arguments.
    
    Parameters 
    ----------
    parser : :class:`argparse.ArgumentParser`
        The same Argument Parser, now with more arguments.
    """
    # Optimizer options
    parser.add_argument('--num-epoch', type=int, default=255, metavar='N',
                        help='number of epochs to train (default: 511)')
    parser.add_argument('--batch-size', '-bs', type=int, default=25, metavar='N',
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='N',
                        help='Value of alpha to use for exponential moving average of training loss. (default: 0.9)')

    parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer (default: 0)')
    parser.add_argument('--cutoff-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer for learnable radial cutoffs (default: 0)')
    parser.add_argument('--lr-init', type=float, default=1e-3, metavar='N',
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--lr-final', type=float, default=1e-5, metavar='N',
                        help='Final (held) learning rate (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=inf, metavar='N',
                        help='Timescale over which to decay the learning rate (default: inf)')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. (cos | linear | exponential | pow | restart) (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=BoolArg, default=True,
                        help='Decay learning rate every minibatch instead of epoch.')
    parser.add_argument('--sgd-restart', type=int, default=-1, metavar='int',
                        help='Restart SGD optimizer every (lr_decay)^p epochs, where p=sgd_restart. (-1 to disable) (default: -1)')

    parser.add_argument('--optim', type=str, default='amsgrad', metavar='str',
                        help='Set optimizer. (SGD, AMSgrad, Adam, RMSprop)')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', action=BoolArg, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    # Saving and logging options
    parser.add_argument('--save', action=BoolArg, default=True,
                        help='Save checkpoint after each epoch. (default: True)')
    parser.add_argument('--load', action=BoolArg, default=False,
                        help='Load from previous checkpoint. (default: False)')

    parser.add_argument('--test', action=BoolArg, default=True,
                        help='Perform automated network testing. (Default: True)')

    parser.add_argument('--log-level', type=str, default='info',
                        help='Logging level to output')

    parser.add_argument('--textlog', action=BoolArg, default=True,
                        help='Log a summary of each mini-batch to a text file.')

    parser.add_argument('--predict', action=BoolArg, default=True,
                        help='Save predictions. (default)')

    ### Arguments for files to save things to
    # Job prefix is used to name checkpoint/best file
    parser.add_argument('--prefix', '--jobname', type=str, default='nosave',
                        help='Prefix to set load, save, and logfile. (default: nosave)')

    # Allow to manually specify file to load
    parser.add_argument('--loadfile', type=str, default='',
                        help='Set checkpoint file to load. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save model checkpoint to
    parser.add_argument('--checkfile', type=str, default='',
                        help='Set checkpoint file to save checkpoints to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to best model checkpoint to
    parser.add_argument('--bestfile', type=str, default='',
                        help='Set checkpoint file to best model to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save logging information to
    parser.add_argument('--logfile', type=str, default='',
                        help='Duplicate logging.info output to logfile. Set to empty string to generate from prefix. (default: (empty))')
    # Filename to save predictions to
    parser.add_argument('--predictfile', type=str, default='',
                        help='Save predictions to file. Set to empty string to generate from prefix. (default: (empty))')

    # Working directory to place all files
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory as a default location for all files. (default: ./)')
    # Directory to place logging information
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Directory to place log and savefiles. (default: log/)')
    # Directory to place saved models
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Directory to place log and savefiles. (default: model/)')
    # Directory to place model predictions
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Directory to place log and savefiles. (default: predict/)')
    # Directory to read and save data from
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                        help='Directory to look up data from. (default: data/)')

    # Dataset options
    parser.add_argument('--num-train', type=int, default=-1, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-valid', type=int, default=-1, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')

    parser.add_argument('--force-download', action=BoolArg, default=False,
                        help='Force download and processing of dataset.')

    # Computation options
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)

    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')

    parser.add_argument('--num-workers', type=int, default=8,
                        help='Set number of workers in dataloader. (Default: 1)')

    # Model options
    parser.add_argument('--num-cg-levels', type=int, default=4, metavar='N',
                        help='Number of CG levels (default: 4)')

    parser.add_argument('--maxl', nargs='*', type=int, default=[3], metavar='N',
                        help='Cutoff in CG operations (default: [3])')
    parser.add_argument('--max-sh', nargs='*', type=int, default=[3], metavar='N',
                        help='Number of spherical harmonic powers to use (default: [3])')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[10], metavar='N',
                        help='Number of channels to allow after mixing (default: [10])')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[10.], metavar='N',
                        help='Gain at each level (default: [10.])')

    parser.add_argument('--charge-power', type=int, default=2, metavar='N',
                        help='Maximum power to take in one-hot (default: 2)')

    parser.add_argument('--hard-cutoff', dest='hard_cut_rad',
                        type=float, default=1.73, nargs='*', metavar='N',
                        help='Radius of HARD cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-cutoff', dest='soft_cut_rad', type=float,
                        default=1.73, nargs='*', metavar='N',
                        help='Radius of SOFT cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-width', dest='soft_cut_width',
                        type=float, default=0.2, nargs='*', metavar='N',
                        help='Width of SOFT cutoff in Angstroms (default: 0.2)')
    parser.add_argument('--cutoff-type', '--cutoff', type=str, default=['learn'], nargs='*', metavar='str',
                        help='Types of cutoffs to include')

    parser.add_argument('--basis-set', '--krange', type=int, default=[3, 3], nargs=2, metavar='N',
                        help='Radial function basis set (m, n) size (default: [3, 3])')

    # TODO: Update(?)
    parser.add_argument('--weight-init', type=str, default='rand', metavar='str',
                        help='Weight initialization function to use (default: rand)')

    parser.add_argument('--input', type=str, default='linear',
                        help='Function to apply to process l0 input (linear | MPNN) default: linear')
    parser.add_argument('--num-mpnn-levels', type=int, default=1,
                        help='Number levels to use in input featurization MPNN. (default: 1)')
    parser.add_argument('--top', '--output', type=str, default='linear',
                        help='Top function to use (linear | PMLP) default: linear')

    parser.add_argument('--gaussian-mask', action='store_true',
                        help='Use gaussian mask instead of sigmoid mask.')

    parser.add_argument('--edge-cat', action='store_true',
                        help='Concatenate the scalars from different \ell in the dot-product-matrix part of the edge network.')
    parser.add_argument('--target', type=str, default='',
                        help='Learning target for a dataset (such as qm9) with multiple options.')

    return parser

def setup_argparse(dataset):
    """
    Sets up the argparse object for a specific dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.  Currently MD17 and QM9 are supported.

    Returns
    -------
    parser : :class:`argparse.ArgumentParser`
        Argument Parser with arguments.
    """
    parser = argparse.ArgumentParser(description='Cormorant network options for the md17 dataset.')
    parser = setup_shared_args(parser)
    if dataset == "md17":
        parser.add_argument('--subset', '--molecule', type=str, default='',
                            help='Subset/molecule on data with subsets (such as md17).')
    elif dataset == "qm9":
        parser.add_argument('--subtract-thermo', action=BoolArg, default=False,
                            help='Subtract thermochemical energy from relvant learning targets in QM9 dataset.')
    else:
        raise ValueError("Dataset is not recognized")
    return parser
def init_argparse(dataset):
    """
    Reads in the arguments for the script for a given dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.  Currently 'md17' and 'qm9' are supported.

    Returns
    -------
    args : :class:`Namespace`
        Namespace with a dictionary of arguments where the key is the name of
        the argument and the item is the input value.
    """

    parser = setup_argparse(dataset)
    args = parser.parse_args([])
    d = vars(args)
    d['dataset'] = dataset

    return args

if __name__=="__main__":
    batch_size = 10
    num_workers = -1
    filter_n_atoms = None
    # Initialize dataloader
    args = init_argparse('qm9')
    # data_dir = cfg.data_root_dir
    datadir = './'
    dataset='qm9'
    remove_h = False
    initialize_datasets(args, datadir, dataset,
                        subtract_thermo=args.subtract_thermo,
                        force_download=args.force_download,
                        remove_h=remove_h)
