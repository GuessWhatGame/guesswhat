# Guesswhat?! models

This repo aim at reproducing the results from the series of GuessWhat?! papers, namely:
- GuessWhat?! Visual object discovery through multi-modal dialogue - https://arxiv.org/abs/1611.08481
- End-to-end optimization of goal-driven and visually grounded dialogue systems - https://arxiv.org/abs/1703.05423

The code was equally developed bu Florian Strub (University of Lille) and Harm de Vries (University of Montreal)

The project is part of the CHISTERA - IGLU Project.

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
    * [Download](#Download)
    * [Requirements](#requirements)
    * [File architecture](#file-architecture)
    * [Data](#data)
    * [Pretrained models](#pretrained-models)
* [Reproducing results](#reproducing-results)
    * [Process Data](#data)
    * [Train Oracle](#oracle)
    * [Train Guesser](#guesser)
    * [Train Qgen](#qgen)
* [FAQ](#faq)    
* [Citation](#citation)

## Introduction

We introduce \GW, a two-player guessing game as a testbed for research on the interplay of computer vision and dialogue systems. The goal of the game is to locate an unknown object in a rich image scene by asking a sequence of questions.
Higher-level image understanding, like spatial reasoning and language grounding, is required to solve the proposed task.

## Installation


### Download

Our code has internal dependences called submodules. To properly clone the repository, please use the following git command:\

```
git clone --recursive git@github.com:GuessWhatGame/guesswhat.git
```

### Requirements

The code works on both python 2 and 3. It relies on the tensorflow python API.
It requires the following python packages:

```
pip install \
    tensorflow-gpu \
    nltk \
    tqdm
```


### File architecture
In the following, we assume that the following file/folder architecture is respected:

```
guesswhat
├── config         # store the configuration file to create/train models
|   ├── oracle
|   ├── guesser
|   ├── qgen
|   └── looper
|
├── out            # store the output experiments (checkpoint, logs etc.)
|   ├── oracle
|   ├── guesser
|   ├── qgen
|   └── looper
|
├── data          # contains the Guesshat data
|   └── img       # contains the coco img
|        └── raw
|
├── vqa            # vqa package dir
|   ├── datasets   # datasets classes & functions dir (vqa, coco, images, features, etc.)
|   ├── external   # submodules dir (VQA, skip-thoughts.torch)
|   ├── lib        # misc classes & func dir (engine, logger, dataloader, etc.)
|   └── models     # models classes & func dir (att, fusion, notatt, seq2vec)
|
└── src            # source files
```

To complete the git-clone file arhictecture, you can do:

```
cd guesswhat
mkdir data; mkdir data/img ; mkdir data/img/raw
mkdir out; mkdir out/oracle ; mkdir out/guesser; mkdir out/qgen; mkdir out/looper ; 
```

Of course, one is free to change this file architecture!

### Data
GuessWhat?! relies on two datasets:
 - the [GuessWhat?!](https://guesswhat.ai/) dataset that contains the dialogue inputs
 - The [MS Coco](http://mscoco.org/) dataset that contains the image inputs

To download the GuessWhat?! dataset please follow the following instruction:
```
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz -P data/
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz -P data/
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz -P data/
```

To download the MS Coco dataset, please follow the following instruction:
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/img/
unzip data/img/train2014.zip -d data/img/raw

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/img/
unzip data/img/val2014.zip -d data/img/raw

# creates a folder `raw` with filenames as expected by preprocessing script below
python ./src/guesswhat/preprocess_data/rewire_coco_image_id.py \ 
   -image_dir `pwd`/data/img/raw \
   -data_out `pwd`/data/img/raw
```

NB: Please check that md5sum are correct after downloading the files to check whether they have been corrupted.
To do so, you can use the following command:
```
md5sum $file
```


### Pretrained networks

Pretrained networks can be downloaded [here](http://florian-strub.com/pretrained_models.zip).
(Warning Tensorflow 1.1). Those networks can then be loaded to reproduce the results.


## Reproducing results

To launch the experiments in the local directory, you first have to set the pyhton path:
```
export PYTHONPATH=src:${PYTHONPATH} 
```
Note that you can also directly execute the experiments in the source folder.

### Process Data

Before starting the training, one needs to compute the image features and the word dictionnary

#### Extract image features
Following the original papers, we are going to extract fc8 features from the coco images by using a VGG-16 network. 

First, one need to download the vgg pretrained network provided by [slim-tensorflow](https://github.com/tensorflow/models/tree/master/slim):

```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -P data/
tar zxvf data/vgg_16_2016_08_28.tar.gz -C data/
```

GuessWhat?! requires to both computes the image features from the full picture 
To do so, you need to use the pythn script guesswhat/src/guesswhat/preprocess_data/extract_img_features.py .
```
array=( img crop )
for mode in "${array[@]}"; do
   python src/guesswhat/preprocess_data/extract_img_features.py \
     -image_dir data/img/raw \
     -data_dir data \
     -data_out data \
     -network vgg \
     -ckpt data/vgg_16.ckpt \
     -feature_name fc8 \
     -mode $mode
done
```

Noticeably, one can also extract VGG-fc7 or Resnet150-block4 features. Please follow the script documentation for more advanced setting. 

#### Create dictionnary

To create the GuessWhat?! dictionary, you need to use the pythn script guesswhat/src/guesswhat/preprocess_data/create_dico.py .

```
python src/guesswhat/preprocess_data/create_dictionary.py -dataset_path data  
```


### Train Oracle
To train the oracle, you need to select/configure the input you want to use.
To do so, you have update the file config/oracle/config.json
By default, the oracle is trained with  spatial+category but one may add/remove inputs.
More information are available in the config folder.

Once the config file is set, you can launch the training step:
```
python src/guesswhat/train/train_oracle.py \
   -data_dir data \
   -image_dir data/gw_img_vgg_fc8_224 \
   -crop_dir data/gw_crop_vgg_fc8_224 \
   -config config/oracle/config.json \
   -exp_dir out/oracle \
   -no_thread 2 
```

After training, we obtained the following results:

| Set       | Loss   | Error |
| --------  |:-----:| -----:|
| Train     | 0.130 | 17.5% |
| Valid     | 0.155 | 20.6% |
| Test      | 0.157 | 21.1% |


### Train Guesser
Identically, you first have to update the config/guesser/config.json

```
python src/guesswhat/train/train_guesser.py \
   -data_dir data \
   -image_dir data/vgg_img \
   -config config/guesser/config.json \
   -exp_dir out/guesser \
   -no_thread 2 
```

After training, we obtained the following results:

| Set       | Loss   | Error |
| --------  |:-----:| -----:|
| Train     | 0.681 | 27.6% |
| Valid     | 0.906 | 34.7% |
| Test      | 0.947 | 35.8% |


### Train QGen
Identically, you first have to update the config/guesser/config.json
```
python src/guesswhat/train/train_qgen_supervised.py \
   -data_dir data \
   -image_dir data/vgg_img \
   -config config/qgen/config.json \
   -exp_dir out/qgen \
   -no_thread 2 
```

After training, we obtained the following results:

| Set       | Loss  |
| --------  |:-----:|
| Train     | 1.31 |
| Valid     | 1.75 |
| Test      | 1.76 |


### Train Looper

The looper use three pretrained models to play the GuessWhat?! game.
Therefore, it provides a user-simulation scheme to perform RL training methods.

In this codebase, the QGen is fine-tuned by using REINFORCE.
The QGen keep playing GuessWhat?! with the Oracle and it is rewarded when the Guesser find the correct object at the end of the dialogue.

To do so, one need to first pretrain the three models.
Each model has a configuration hash and checkpoint. These configuration hash will be used as an entry point for the Looper.

```
python src/guesswhat/train/train_qgen_reinforce.py
    -data_dir data/ \
    -exp_dir out/loop/ \
    -config config/looper/config.json \
    -image_dir data/vgg_img \
    -networks_dir out \
    -no_thread 2
```

We obtain the following scores (with +/- 0.3%)

| Set       | Cross-entropy   | Reinforce |
| --------  |:-----:| -----:|
| Train (new object)    | 39.0% | 58.5% |
| Valid (new images)    | 41.8% | 57.6% |
| Test  (new images)    | 39.8% | 56.4% |

## FAQ

 - When I start a python script, I have the following message: ImportError: No module named generic.data_provider.iterator (or equivalent module). It is likely that your python path is not correctly set. Add the "src" folder to your python path (PYTHONPATH=src)
 

## Citation

GuessWhat?! framework - https://arxiv.org/abs/1611.08481
```
@inproceedings{guesswhat_game,
author = {Harm de Vries and Florian Strub and Sarath Chandar and Olivier Pietquin and Hugo Larochelle and Aaron C. Courville},
title = {GuessWhat?! Visual object discovery through multi-modal dialogue},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017}
}
```

Reinforcement Learning applied to GuessWhat?! - https://arxiv.org/abs/1703.05423
```
@inproceedings{end_to_end_gw,
author = {Florian Strub and Harm de Vries and J\'er\'emie Mary and Bilal Piot and Aaron C. Courville and Olivier Pietquin},
title = {End-to-end optimization of goal-driven and visually grounded dialogue systems},
booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
year = {2017}
}
```

## Acknowledgement
 - SequeL Team
 - Mila Team

We would also like people that help improving the code base namely: Rui Zhao, Hannes Schulz.





