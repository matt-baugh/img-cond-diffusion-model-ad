# Image-conditioned Diffusion Models for Medical Anomaly Detection

We integrate our method into the benchmark ["Unsupervised Pathology Detection: A Deep Dive Into the State of the Art"](https://ieeexplore.ieee.org/document/10197302) to enable future development and comparisons. 

Code for our method is inside `UPD_study/models/ours/`.

# Usage

Download this repository by running

```bash
git clone https://github.com/img-cond-diffusion-model-ad
```

in your terminal.

## Environment

Create and activate the Anaconda environment:

```bash
conda env create -f environment.yml
conda activate anomaly_restoration
```

Additionally, you need to install the repository as a package:

```bash
python3 -m pip install --editable .
```

To be able to use [Weights & Biases](https://wandb.ai) for logging follow the instructions at https://docs.wandb.ai/quickstart.
<!-- 
A quick guide on the folder and code structure can be found [here](structure.md). -->

## Data

### DDR 

To download and prepare the DDR dataset, run:

```bash
bash UPD_study/data/data_preprocessing/prepare_DDR.sh
```

### MRI: CamCAN, ATLAS, BraTS 

To download and preprocess ATLAS and BraTS, first download ROBEX from https://www.nitrc.org/projects/robex  and extract it in data/data_preprocessing/ROBEX. Then run:

```bash
bash UPD_study/data/data_preprocessing/prepare_ATLAS.sh
bash UPD_study/data/data_preprocessing/prepare_BraTS.sh
```
For ATLAS you need to apply for the data at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html and receive the encryption password. During the run of prepare_ATLAS.sh you will be prompted to input the password.

For BraTS, Kaggle's API will be used to download the data. To be able to interact with the API, follow the instructions at https://www.kaggle.com/docs/api.

To download the CamCAN data, you need to apply for it at https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/index.php. After you download them, put them in data/datasets/MRI/CamCAN and run:

```bash
python UPD_study/data/data_preprocessing/prepare_data.py --dataset CamCAN
```

## Our experiments

We recommend using [accellerate](https://huggingface.co/docs/accelerate/index) to train/evaluate models over multiple GPUs.

For any experiment, to select which image modality you use:
```bash
--modality [MRI|RF]
```
Where MRI is for brain MRI and RF is for DDR.
To select which sequence of MRI you use:
```bash
--sequence [t1|t2]
```
For evaluating the T1 model, on BraTS-T1 use `--brats_t1=f` while for ATLAS use `--brats_t1=t`.
In the following script examples I will denote the dataset choice as `<DATASET_OPTIONS>`
### Training

To train a model on fold `f` \in [0,9], run:

```bash
accelerate launch \
            --num_processes=$num_processes --mixed_precision=fp16 \
            ./UPD_study/models/ours/ours_trainer.py \
            --fold=$f <DATASET_OPTIONS>
```

### Fold Evaluation
```bash
accelerate launch \
            --num_processes=$num_processes --mixed_precision=fp16 \
            ./UPD_study/models/ours/ours_trainer.py \
            --fold=$f -ev=t  <DATASET_OPTIONS>
```

### Ensemble Evaluation
```bash
accelerate launch \
            --num_processes=$num_processes --mixed_precision=fp16 \
            ./UPD_study/models/ours/ours_ensemble.py \
            --fold=$f -ev=t  <DATASET_OPTIONS>
```

## Original Benchmark Experiments

To generate the "Main Results" from Tables 1 and 3 over all three seeds run:
```bash
bash UPD_study/experiments/main.sh 
```
Alternatively, for a single seed run:

```bash
bash UPD_study/experiments/main_seed10.sh 
```


To generate the "Self-Supervised Pre-training" results from Tables 2 and 4 over all three seeds run:
```bash
bash UPD_study/experiments/pretrained.sh
```
Alternatively, for a single seed run:

```bash
bash UPD_study/experiments/pretrained_seed10.sh      
```

To generate the "Complexity Analysis" results from Table 5 run:
```bash
bash UPD_study/experiments/benchmarks.sh
```

To generate "The Effects of Limited Training Data" results from Fig. 3 run:
```bash
bash UPD_study/experiments/percentage.sh
```
##

The repository contains PyTorch implementations for [VAE](https://arxiv.org/abs/1907.02796), [r-VAE](https://arxiv.org/abs/2005.00031), [f-AnoGAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640), [H-TAE-S](https://arxiv.org/abs/2207.02059), [FAE](https://arxiv.org/abs/2208.10992), [PaDiM](https://arxiv.org/abs/2011.08785), [CFLOW-AD](https://arxiv.org/abs/2107.12571), [RD](https://arxiv.org/abs/2201.10703), [ExpVAE](https://arxiv.org/abs/1911.07389), [AMCons](https://arxiv.org/abs/2203.01671), [PII](https://arxiv.org/abs/2107.02622), [DAE](https://openreview.net/forum?id=Bm8-t_ggzPD) and [CutPaste](https://arxiv.org/abs/2104.04015).

## Cite this work:

```
@InProceedings{baugh2024imageconditioned,
    author="Baugh, Matthew
    and Reynaud, Hadrien
    and Marimont, Sergio Naval
    and Cechnicka, Sarah
    and M{\"u}ller, Johanna P.
    and Tarroni, Giacomo
    and Kainz, Bernhard",
    title="Image-Conditioned Diffusion Models for Medical Anomaly Detection",
    booktitle="Uncertainty for Safe Utilization of Machine Learning in Medical Imaging",
    editor="Sudre, Carole H.
    and Mehta, Raghav
    and Ouyang, Cheng
    and Qin, Chen
    and Rakic, Marianne
    and Wells, William M.",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="117--127",
    isbn="978-3-031-73158-7"
}
```

## Acknowledgements

(Some) HPC resources were provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR projects b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.
