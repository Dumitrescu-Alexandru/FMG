# E(3)-equivariant models cannot learn chirality: Field-based molecular generation

Official implementation of [E(3)-equivariant models cannot learn chirality: Field-based molecular generation](https://openreview.net/forum?id=mXHTifc1Fn&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))

by [Alexandru Dumitrescu](https://dumitrescu-alexandru.github.io/site/), [Dani Korpela](https://www.linkedin.com/in/dani-korpela-81557319a/?originalSubdomain=fi), [Markus Heinonen](https://www.linkedin.com/in/dani-korpela-81557319a/?originalSubdomain=fi), [Yogesh Verma](https://yoverma.github.io/yoerma.github.io/), [Valerii Iakovlev](https://www.linkedin.com/in/valerii-iakovlev-a12596190/), [Vikas Garg](https://www.mit.edu/~vgarg/), and [Harri Lähdesmäki](https://www.aalto.fi/fi/ihmiset/harri-lahdesmaki)


## Install environment

Tested on python 3.9.18

run 
```
	pip install -r requirements.txt
```

## Paths

By default, the model and data paths from `config_paths_local.yml` are used. Change this file accordingly if needed.


## Generated samples

The results of this paper are based on the generated molecules found under `generated_molecules/geom_drugs` and `generated_molecules/qm9`.

## QM9

### Data (required for both generation and training)

- go to `data/qm9/data/qm9_edm`.
- run `python create_edm_data_splits.py`. This will create `train.npz`, `test.npz`, `valid.npz` in `data/qm9/data/qm9_edm/qm9`.
- run training (from Section _QM9 model training_). This automatically creates correct format training file and saves it with the name specified in `--data_file` argument. Saved in `data/qm9/data`.
- if you want to train the model, you can rerun the same training command

### Generating

Pre-trained model weights: download from https://drive.google.com/drive/folders/1XpOfCPRvPu22dSgbWgfGRF0Lul7ygdC7?usp=sharing

Before generating, please follow the steps from the previous subsection (_Data_).

Run:
```
	python -u test_rdkit_metrics.py --model <model_path> --data <training_data_file> --explicit_hydrogen --timesteps 100 --cls_free_guid --small_mdl --threshold_atm 0.4 --threshold_bond 0.4 --batch_size 50 --optimize_bnd_gmm_weights --beta_schedule cosine --noise_scheduler_conf sched_configs/qm9_explh_sepSched_destrAtmsFirst.yaml --run_name <save_file_name> --discard_fields --num_imgs 300
```

<model_path>: path to pre-trained model

<training_data_file>: training data file (ensuring resolution, grids, and channel counts are are the same as the training). Using the instructions from _QM9 data (required for both generation and training)_ and having run _QM9 model training_ command, the name should be `qm9_data_explicit_hydrogen_033_res.bin`.

<save_file_name>: filename for the generated molecules. The path under which it is saved is the same as the "model_path" in config_paths_local.yml.


If many molecules are required, the argument `--discard_fields` may be handy, since the fields take a large amount of storage space.

### Training
For QM9, the best configuration run is:

```
	python -u train_class_free_guid.py --data QM9EDM_small --batch_size 64 --beta_schedule cosine --data_file qm9_data_explicit_hydrogen_033_res --model unet --run_name test_run_name --resolution 0.33 --lr 3e-5 --explicit_hydrogen --ntimesteps 100 --compact_batch --noise_scheduler_conf sched_configs/qm9_explh_sepSched_destrAtmsFirst.yaml --use_original_atm_pos --test_every 1

```

## GEOM-Drugs


### Data (generation only, full training data available soon!)

- download required data files for generation from https://drive.google.com/file/d/1g_8A-7R7UAjemOHKZdEBcQ93kSnmvIdN/view?usp=sharing into `data/geom_data`.
- Go to `data/geom_data`. Extract files `tar xvf geom_generation_data.tar`.

### Generating

Pre-trained model weights: download from https://drive.google.com/file/d/16S2p_ioHxnrt2KSCK40mMoTGphQTv9oX/view?usp=sharing

Before generating, please follow the steps from the previous subsection (_Data_).

```
	python generate.py --model <model_path> --batch_size 2 --threshold_atm 0.35 --threshold_bond 0.35 --optimize_bnd_gmm_weights --noise_scheduler_conf  sched_configs/geom_explh_sepSched_destrAtmsFirst.yaml --timesteps 100 --data geom_data --cls_free_guid --small_mdl  --data_type GEOM --explicit_hydrogen --optimize_atom_positions --run_name <training_data_file> --num_imgs 4 --discard_fields
```

<model_path>: path to pre-trained model

<save_file_name>: filename for the generated molecules. The path under which it is saved is the same as the "model_path" in config_paths_local.yml.


### Training a model

Available soon!



## Testing generated molecules

Available soon!