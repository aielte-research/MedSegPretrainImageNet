# ImageNet pretraining for medical image segmentation
## Enhancing efficiency via transferability metrics

This repository contains the code and all the results related to the paper “ImageNet pretraining for medical image segmentation: Enhancing efficiency via transferability metrics”.

## Reproducibility

### Requirements


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

Patient folders contain data as it is in the original dataset.


#### COVID-QU:

```bash
.
+--Inf_segm
|  +--data_arrays
      +--COVID-19
         +--train_and_val
            +--imgs.npy
            +--inf_masks.npy
         +--test
            +--imgs.npy
            +--inf_masks.npy
+--Lung_segm
   +...
```

The imgs.npy and inf_masks.npy files contain Nx256x256 shaped arrays with the images and the segmentation masks respectively.
The train_and_val folder contains the data from the original train and validation set. The test folder contains the data from the original test set.


#### IDRiD:

```bash
.
+--Segmentation
   +--A. Segmentation
      +--1. Original Images
         +--a. Training Set     # Folder with train image .jpg files
         +--b. Testing Set      # Folder with test image .jpg files
      +--2. All Segmentation Groundtruths
         +--a. Training Set     # Folder with train segmentation masks
            +--1. Microaneurysms
            +--2. Haemarrhages
            +--3. Hard Exudates
            +--4. Soft Exudates
            +--5. Optic Disc
         +--b. Testing Set      # Folder with test segmentation masks
            +--1. Microaneurysms    
            +--2. Haemarrhages
            +--3. Hard Exudates
            +--4. Soft Exudates
            +--5. Optic Disc
```

The `../Original Images/Training Set` and `../Original Images/Test Set` contains the ?x? images from the original dataset.
The  folders in `../All Segmentation Groundtruths/Training Set` and `../All Segmentation Groundtruths/Training Set` contains the .tif files with the segmentation masks (as in the original dataset).


#### ImageNet:

```bash
.
+--...
```




### Running experiments

Experiments can be run with the following script:

```
python experiment.py [path_to_config.yaml]
```

Config files for training models on each dataset can be found in the `config` directory.

### Encoder weights

Weights of the ImageNet-pretrained encoders can be found ???

## Results

The `results` directory contains detailed results for two sets of experiments:
* `downstream_scores.csv` lists the Dice index, Jaccard index, MCC, mAP, HD95, balanced accuracy, sensitivity, AUC, accuracy, and specificity scores of each model on the downstream datasets (though only a subset of these were calculated for the multiclass case of ACDC);
* `robustness_scores.csv` lists the calculated robustness of each encoder, through a combination of
    * distance metrics: cosine distance, *L*^2^ distance, and the inverse of the Pearson correlation,
    * margins,
    * representation taken from different encoder levels,
    * pooled and not pooled.

<!--
## (Acknowledgement)

## Citation
-->
