# ImageNet pretraining for medical image segmentation
## Enhancing efficiency via transferability metrics

This repository contains the code and all the results related to the paper “ImageNet pretraining for medical image segmentation: Enhancing efficiency via transferability metrics”.

## Reproducibility

### Requirements


### Data


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