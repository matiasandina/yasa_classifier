# yasa_classifier

Repository to train a sleep staging classifier. We will use:

* An open source sleep staging dataset with ground truth labels (see [dataset](#dataset))
* `yasa` to extract features
* `lightgbm` to train a classifier


## Dataset

The dataset was collected from [OSF](https://osf.io/py5eb/). It contains mouse EEG/EMG recordings (sampling rate: 512 Hz) and sleep stage labels (epoch length: 2.5 sec).

Training was performed using extracted features from 24h recordings.

## Procedure to reproduce

Dataset can be downloaded using this link 

```
https://files.osf.io/v1/resources/py5eb/providers/osfstorage/?zip=
```


or downloaded manually from [OSF](https://osf.io/py5eb/). See [Dataset Structure](#dataset-structure) for details of expected file structure.

### Dataset structure

Datasets have the following structure

    .
    ├── Mouse01
    │   ├── Day1_dark_cycle
    │   │   ├── EEG.mat
    │   │   ├── EMG.mat
    │   │   └── labels.mat
    │   ├── Day1_light_cycle
    │   │   ├── EEG.mat
    │   │   ├── EMG.mat
    │   │   └── labels.mat
    │   ├── Day2_dark_cycle
    │   │   ├── EEG.mat
    │   │   ├── EMG.mat
    │   │   └── labels.mat
    │   └── Day2_light_cycle
    │       ├── EEG.mat
    │       ├── EMG.mat
    │       └── labels.mat

### Extract Features

`01-extract_features.qmd` was run to extract features. An important note is that it uses a local version of `SleepStaging()` (`from staging import SleepStaging`) that differs from the implementation in `yasa`. This was included for reproducibility, though we have plans to include this version in `yasa` itself and will be no longer needed.

### Train the model

`02-train.qmd` was run to train on the 24 hour recordings. The outputs of this notebook are saved into `/output`. 

### Output structure

Model output will be stored as

```
output/
├── classifiers/
│   ├── eeg_full/
│   │   ├── model.joblib
│   │   └── feature_importances.csv
│   ├── eeg_no_kurt/
│   │   ├── model.joblib
│   │   └── feature_importances.csv
│   ├── eeg_no_std/
│   |   ├── model.joblib
│   |   └── feature_importances.csv
|   |
|   | ...
```


### Evaluate the model
`03-evaluate.qmd` was run to evaluate and produce accuracy and cohen's kappa metrics. The outputs of this notebook are saved into `/output`.

## Contribute

This is a preliminary release, file issues to enhance functionality.

