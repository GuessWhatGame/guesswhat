# Guesswhat?! models

This repo aims at reproducing the results from the series of GuessWhat?! papers, namely:
- GuessWhat?! Visual object discovery through multi-modal dialogue [1](https://arxiv.org/abs/1611.08481)
- End-to-end optimization of goal-driven and visually grounded dialogue systems [2](https://arxiv.org/abs/1703.05423)

The code was equally developed by Florian Strub (University of Lille) and Harm de Vries (University of Montreal) and it is part of the CHISTERA - IGLU Project.

[This branch](https://github.com/GuessWhatGame/guesswhat/tree/refacto_v2) contains a more advanced codebase and multiple baselines.

**WARNING: After refactoring the code of the original paper, we fixed a bug in the codebase (the last generated question was ignored in some cases). New scores are greatly above the scores reported in [1] but some results analysis are now obsolete (qgen stop learning to stop, greedy has the highest accuracy). We apologize for the inconvenience.**

## Introduction

We introduce GuessWhat?!, a two-player guessing game as a testbed for research on the interplay of computer vision and dialogue systems. The goal of the game is to locate an unknown object in a rich image scene by asking a sequence of questions.
Higher-level image understanding, like spatial reasoning and language grounding, is required to solve the proposed task.

## Installation


### Download

Our code has internal dependences called submodules. In order to retrieve the submodules that reside outside of the main repository, the ```--recursive``` flag is necessary. Please use the command below:

```
git clone --recursive https://github.com/GuessWhatGame/guesswhat.git
```

### Requirements

To support the right version of tensorflow needed to run the code, a Python version up to 3.6 is needed. 

The rest of the requirements you can install by running  ```pip install -r setup/requirements.txt``` or have a complete setup together with the right Python version by running ```conda env create --name guesswhat-env --file setup/conda.yml python=3.6``` if you are using anaconda for your virtual enviroment management. 


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
|        ├── ft_vgg_img
|        ├── ft_vgg_crop
|        └── raw
|
└── src            # source files
```

To complete the git-clone file arhictecture, you can do:

```
cd guesswhat
mkdir data; mkdir data/img ; mkdir data/img/raw ; mkdir data/img/ft_vgg_img ; mkdir data/img/ft_vgg_crop
mkdir out; mkdir out/oracle ; mkdir out/guesser; mkdir out/qgen; mkdir out/looper ; 
```

Of course, one is free to change this file architecture!

### Data
GuessWhat?! relies on two datasets: GuessWhat?! which contains the dialogue inputs and [MS Coco](https://cocodataset.org/#download) which contains the image inputs.

To download the GuessWhat?! dataset please follow the following instruction:
```
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```

To download the MS Coco dataset, please follow the following instruction:
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/img/
unzip data/img/train2014.zip -d data/img/raw

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/img/
unzip data/img/val2014.zip -d data/img/raw

# creates a folder `raw` with filenames as expected by preprocessing script below
python src/guesswhat/preprocess_data/rewire_coco_image_id.py \ 
   -image_dir data/img/raw \
   -data_out data/img/raw
```

NB: Please check that md5sum are correct after downloading the files to check whether they have been corrupted.
To do so, you can use the following command:
```
md5sum $file
```


### Pretrained networks

Pretrained networks can be downloaded here:

V1 of the code:
 - Tensorflow (1.0<->1.2): [network](http://florian-strub.com/github/pretrained_models.tf1-2.zip).
 - Tensorflow (1.3): [network](http://florian-strub.com/github/pretrained_models.tf1-3.zip).

You need to use the following tag to checkout the corresponding code:
```
git checkout tags/v1
```

V2 of the code:
 - coming sooon! It would include GW?! with advanced FiLM models, new RL algorithms, new tools!

Note that the reported results comes from the first version (v1) of pre-trained networks.

## Reproducing results

To launch the experiments in the local directory, you first have to set the pyhton path:
```
export PYTHONPATH=src:${PYTHONPATH} 
```
Note that you can also directly execute the experiments in the source folder.

### Process Data

Before starting the training, one needs to compute the image features and the word dictionary

#### Extract image features
Following the original papers, we are going to extract fc8 features from the coco images by using a VGG-16 network. There are two solutions available to obtain the vgg-img features: 

- Solution 1: You can directly download the vgg features:  
```
wget www.florian-strub.com/github/ft_vgg_img.zip -P data/img
unzip data/img/ft_vgg_img.zip -d data/img/
```

- Solution 2: You can download vgg-16 pretrained network provided by [slim-tensorflow](https://github.com/tensorflow/models/tree/master/research/slim):

```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -P data/
tar zxvf data/vgg_16_2016_08_28.tar.gz -C data/
```

However, in order to obtain the crop features, you must use the download available as Solution 2 and run the script at the end of this section to generate the crop features. This is a computationally costly step. In order to run just the guesser, just downloading the VGG features from solution 1 is enough. To train the Oracle, the crop features are required. 

GuessWhat?! requires both to compute the image features from the full image. In order to compute the crop image features(and image features if you would like to compute them from scratch), you need to use the [python script](guesswhat/src/guesswhat/preprocess_data/extract_img_features.py), as shown below.

Replace ```$mode``` with ```img``` or ```crop```.

```
python src/guesswhat/preprocess_data/extract_img_features.py \
   -img_dir data/img/raw \
   -data_dir data \
   -out_dir data/img/ft_vgg_$mode \
   -network vgg \
   -ckpt data/vgg_16.ckpt \
   -feature_name fc8 \
   -mode $mode
```

Using this script, one can extract both VGG-fc7 and Resnet features. Please follow the script documentation for more advanced setting.


#### Create dictionary
Creating a dictionary is a necessary step before running the model. To create the GuessWhat?! dictionary, you need to use the [designated script](guesswhat/src/guesswhat/preprocess_data/create_dico.py) .

```
python src/guesswhat/preprocess_data/create_dictionary.py -data_dir data -dict_file dict.json -min_occ 3
```

### Updating the config file before training
The [config file](config/oracle/config.json) contains the default values for training. If you would like to provide different values, you will need to edit the file and it's default values. This applies to training every part of the game. Please look at the config folder for more information about this.


### Train Oracle
By default, the oracle is trained with spatial+category input. You can change this as mentioned above.
Once the config file is set, you can launch the training step:
```
python src/guesswhat/train/train_oracle.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
   -crop_dir data/img/ft_vgg_crop \
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
```
python src/guesswhat/train/train_guesser.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
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
```
python src/guesswhat/train/train_qgen_supervised.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
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
    -img_dir data/img/ft_vgg_img \
    -crop_dir data/img/ft_vgg_crop \
    -networks_dir out/ \
    -oracle_identifier <oracle_identifier> \
    -qgen_identifier <qgen_identifier> \
    -guesser_identifier <guesser_identifier> \
    -evaluate_all false \
    -store_games true ´\
    -no_thread 2
```

Activate the flag evaluate_all to also compute the accuracy with BeamSearch and Sampling (Time-consuming).

Detailled accuracies:

| New Images| Cross-entropy  | Reinforce |
| --------  |:-----:|:-----:|
| Sampling   | 39.2% | 56.5 % |
| Greedy     | 40.8% | 58.4 % |
| BeamSearch | 44.6%| 58.4 % |


| New Objects| Cross-entropy  | Reinforce |
| --------  |:-----:|:-----:|
| Sampling   | 41.6%  | 58.5% |
| Greedy     | 43.5%  | 60.3% |
| BeamSearch | 47.1% | 60.2% |

Note that those scores are *exact* accuracies. 

### Plot dataset

It is possible to plot figures analysing guesswhat raw dataset or QGen generated games (if you set the flag, store_games to True). To do so, 

```
python src/guesswhat/statistics/statistics/plot_them_all.py
   -data_dir data 
   -out_dir out 
   -name train
   -ignore_incomplete false
```

- The "ignore_incomplete flag" will take into account or ignore incomplete games in the dataset.
- The "name" flag correspond to the following variable guesswhat.${name}.jsonl.gz

Note: If you want to compute you personal plot, you can user the AbstractPloter interface and append your implementation into the plot_them_all.py script.

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
@inproceedings{strub2017end,
  title={End-to-end optimization of goal-driven and visually grounded dialogue systems},
  author={Strub, Florian and De Vries, Harm and Mary, Jeremie and Piot, Bilal and Courville, Aaron and Pietquin, Olivier},
  booktitle={Proceedings of international joint conference on artificial intelligenc (IJCAI)},
  year={2017}
}
```

## Acknowledgement
 - SequeL Team
 - Mila Team

We would also like people that help improving the code base namely: Rui Zhao, Hannes Schulz.





