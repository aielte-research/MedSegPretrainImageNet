# Enhancing pretraining efficiency for medical image segmentation via transferability metrics

This repository contains the code and all the results related to the paper “Enhancing pretraining efficiency for medical image segmentation via transferability metrics” (arXiv:[2410.18677](https://arxiv.org/abs/2410.18677)).

## Reproducibility

### Requirements

For the list of required packages see `scr/requirements.yml`.

### Data


Directory structures for the different datasets:

#### ACDC:

```bash
.
+--training
|  +patient001
|  +...
+--testing
   +patient101
   +...

```
Root directory: `../data/ACDC/`

Patient folders contain data as it is in the [original dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/).


#### COVID-QU:

```bash
.
+--Inf_segm
|  +--data_arrays
|     +--COVID-19
|        +--train_and_val
|        |  +--imgs.npy
|        |  +--inf_masks.npy
|        +--test
|           +--imgs.npy
|           +--inf_masks.npy
+--Lung_segm
   +...
```

Root directory: `../data/COVID_QU/`

The imgs.npy and inf_masks.npy files contain Nx256x256 shaped arrays with the images and the segmentation masks respectively.
The train_and_val folder contains the data from the [original](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) train and validation set. The test folder contains the data from the original test set.


#### IDRiD:

```bash
.
+--Segmentation
   +--A. Segmentation
      +--1. Original Images
      |  +--a. Training Set     # Folder with train image .jpg files
      |  +--b. Testing Set      # Folder with test image .jpg files
      +--2. All Segmentation Groundtruths
         +--a. Training Set     # Folder with train segmentation masks
         |  +--1. Microaneurysms
         |  +--2. Haemarrhages
         |  +--3. Hard Exudates
         |  +--4. Soft Exudates
         |  +--5. Optic Disc
         +--b. Testing Set      # Folder with test segmentation masks
            +--1. Microaneurysms    
            +--2. Haemarrhages
            +--3. Hard Exudates
            +--4. Soft Exudates
            +--5. Optic Disc
```

Root directory: `../data/idrid/`

The `../Original Images/Training Set` and `../Original Images/Test Set` contains the 2848x4288 images from the [original dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid).
The  folders in `../All Segmentation Groundtruths/Training Set` and `../All Segmentation Groundtruths/Training Set` contains the .tif files with the segmentation masks (as in the original dataset).


#### ImageNet:

```bash
.
+--data
   +n01440764_10026.npy
   +n01440764_10027.npy
   +...
   +ILSVRC2012_val_00000001.npy
   +ILSVRC2012_val_00000002.npy
   +...
+labels.json
```

Root directory: `../data/imagenet/`

`data` contains RGB images as .npy files, with channels-first encoding. Pixel intensities are represented as ints between 0 and 255. Filenames are taken from the [original dataset](https://image-net.org/challenges/LSVRC/index.php).

labels.json contains a dictionary, where the keys are the file paths for the arrays in `data`, and the values are the corresponding class indices (represented as an int between 1 and 1000).


### Running experiments

Experiments can be run with the following script:

```
python src/experiment.py [path_to_config.yaml]
```

Config files for training models on each dataset can be found in the `config` directory.

### Encoder weights

Weights of the ImageNet-pretrained models should be in an `../encoder_weights` directory. Our pretrained model weights can be downloaded from [here](https://zenodo.org/records/13971513).

## Results

The `results` directory contains detailed results for two sets of experiments:
* `downstream_scores.csv` lists the Dice index, Jaccard index, MCC, mAP, HD95, balanced accuracy, sensitivity, AUC, accuracy, and specificity scores of each model on the downstream datasets (though only a subset of these were calculated for the multiclass case of ACDC);
* `robustness_scores.csv` lists the calculated robustness of each encoder, through a combination of
    * distance metrics: cosine distance, L2 distance, and the inverse of the Pearson correlation,
    * margins,
    * representation taken from different encoder levels,
    * pooled and not pooled.

<!--
## (Acknowledgement)

## Citation
-->
