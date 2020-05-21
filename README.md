# HearSay - Rumour Veracity Detection Architecture

A GitHub repository for the Social and Cultural Dynamics Exam Paper.

A collaboration between [@saxogrammaticus](https://github.com/saxogrammaticus) and [@maltehb](https://github.com/maltehb)

Contains code to reproduce results.

To download the data follow this link: https://competitions.codalab.org/competitions/19938

To hydrate the twitter data we suggest using the hydrator by DocNow: https://github.com/DocNow/hydrator

NB. Some of the code is originally sourced from [@Kochkinaelena's RumourEval2019 baseline model](https://github.com/kochkinaelena/RumourEval2019)
## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/saxogrammaticus/HearSay.git
cd HearSay
pip install -r ./requirements.txt
```

## Data Preparation

The data required to reproduce this demo is avaiable at the link above. Please follow the instructions below for preparing the correct directory structure:
1. Unzip folder and subfolders (Windows: 7-zip is recommended)
2. Move all the twittter subfolders inside the test folder, into the twitter english folder of the training directory. All train and test folders should now be in the same folder.
3. Move final-eval-key.json to the same folders as train-key.json and dev-key.json
4. Delete so you are left with only your dataset folder “twitter-english”, and your json files
5. Extract the ids from the three json files, save them in separate newline separated txt files and hydrate them. Don't forget to remove the reddit ids.
6. Save the hydrated tweets as csv and run the csv preprocessing script ‘csv_wrangler.py’
7. Remove the csv files not ending in 'clean.csv'
8. All files should be in Data folder in the main HearSay directory

The final directory  structure should look something like so:

```
Data/
├── twitter-english/
│   ├── afghanistan
│   ├── african-american
│   ├── ...
│   ├── putinmissing
│   └── wildfires-deduction
│
├── dev-key.json
├── train-key.json  
├── final-eval-key.json
│
├── final-dev-key-clean.csv
├── final-train-key-clean.csv  
└── final-eval-key-clean.csv
```

## Data preprocessing

Preprocess the data with the Preprocess.py processing script.
The processing script can be called as follows

```bash
python ./Preprocess.py 
```
The preprocessing script accepts several arguments:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `"Data/"` | Directory path for all data. Folder should be in structure presented above
dataset_cache | `str` | `"twitter-english"` | Name of the dataset
metacontext | `str` | `"Data/"` | Folder contains hydrated csv files
output | `str` | `"outdata"` | Output directory for preprocessed data



## Training

Once the data has been preprocessed, the training can be run with the Train.py file.
The training script can be called as follows

```bash
python ./Train.py 
```
The training script accepts several arguments:
Argument | Type | Default value | Description
---------|------|---------------|------------
input | `str` | `""` | Folder containing both train/dev/test data
params | `str` | `''` | Folder for hyperparameter files
HPsearch | `bool` | `False` | Flag for whether or not hyperparameter search should be activated
log_path | `str` | `"logs"` | Path for logging data
log_name | `str` | `"log"` | Name for logging folder

To replicate our results run script with HPsearch set to true.
NB. Hyperparameter search alone took 14 hours on an NVIDIA Tesla K80 GPU with 12 GiB. 
    Training once best Hparams were found, took an additional 4 hours.
```bash
python ./Train.py --input "outdata" --HPsearch
```
