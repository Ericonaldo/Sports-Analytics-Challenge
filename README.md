# Sports-Analytics-Challenge

Sports Analytics Challenge sponsored by PSG. Link: https://www.agorize.com/zh/challenges/xpsg/pages/regles-du-jeu-futur?lang=en

==============================

## Project Organization
------------
    ├── README.md                             <- The top-level README for developers using this project.
    ├── data
    │   ├── train                             <- Training data used in experiments.
    │   ├── processed                         <- Data used for indication and further processing during experiments.
    │   ├── valid                             <- Valid data used in experiments.
    │   └── XPSG - available resources        <- The original, immutable data dump (after unzipping).
    │
    │
    ├── models                                <- Trained and serialized models, contained lightGBM models and RNN models
    │
    │
    ├── fig                                   <- Some useful figures for indication during experiments.
    │
    │
    ├── code                                  <- Python script source code and Jupyter notebook source code for use in this project.
    │   │
    │   ├── main_psgx.py                      <- The required python code file which contains Result(xml1) function.
    │   │
    │   ├── install_psgx.py                   <- The required python code file which is used for installing requirements.
    │   │
    │   ├── utils.py                          <- Useful tool codes used in experiments, containing some data preprocessing, feature extraction and other tools.
    │   │
    │   ├── train_gbdt.py                     <- Training a gbdt model with lightGBM which give the answer of the last 3 questions. See details below.
    │   │
    │   ├── evaluation_gbdt.py                <- Testing the above gbdt model on valid sets and show the evaluation results. See details below.
    │   │
    │   ├── evaluation_rules.py               <- Testing naive rules on valid sets and show the evaluation results. This result gives a baseline.
    │   │
    │   ├── dataset.py                        <- This file define some class of pytorch Dataset, used for training of deep learning models with pytorch.
    │   │
    │   ├── train_team_event_rnn.py           <- Training rnn models with pytorch which give the answer of the last 3 questions. See details below.
    │   │ 
    │   ├── train_player_rnn.py               <- Training rnn models with pytorch which give the answer of the first question. See details below.
    │   │ 
    │   ├── evaluation_rnn.py                 <- Testing the above rnn models on valid sets and show the evaluation results. See details below.
    │   │ 
    │   ├── train_xyt_gbdt.ipynb              <- Some draft codes during experiments.
    │   │ 
    │   ├── construct_train.ipynb             <- Some draft codes during experiments.
    │   │ 
    └   └── test.ipynb                        <- Some draft codes during experiments.

## Instructions

### Before training

#### Get validation sets

1. Enter your terminal and cd into "code" dir

2. `python`

3. `from utils import *`

4. `construct_val_sets(valid_num=500)`('valid_num' means the valid set num)



#### Train and evaluate the gbdt model (only for question 2-4):

Train the gbdt model:

1. Enter your terminal and cd into "code" dir

2. `python train_gbdt.py  --corr --xymodel` ('corr' means saving corr img, '\*model' means which model do you want to train)

Evaluate the gbdt model:

1. Enter your terminal and cd into "code" dir

2. `python evaluation_gbdt.py --valid 100 --tmodel -- xymodel` ('valid' means the valid set num (if is not constructed before), '\*model' means which model do you want to test)


#### Train and evaluate the deep model (for question 1-4):

Train the deep model:

1. Enter your terminal and cd into "code" dir

2. `python train_team_event_rnn.py  --load` ('load' means if load a pre trained model (not need if train fron sketch), this model is for question 2-4)

3. `python train_player_rnn.py  --load` ('load' means if load a pre trained model (not need if train fron sketch), this model is for question 1))

Evaluate the deep model:

1. Enter your terminal and cd into "code" dir

2. `python evaluation_gbdt.py --valid 100 --test_p --test_xyt --xyt_epoch 360 --p_epoch` ('valid' means the valid set num (if is not constructed before), 'test_\*' means which model do you want to test, 'xyt_epoch' means which epoch of model do you want to load for 'team_event_rnn' model, default as 360, 'p_epoch' means which epoch of model do you want to load for 'player_rnn' model, default as 500)


## Solution details

### Introduction

In this competition, I take the 4 (or 3) questions dividely, and propose 2 solutions for those problems.

For the No.1 question which is to predict the player id, I take it as a Fine-grained classification problem, for which the high level class is the position of a player and it gives information when predicting the low level class -- the true players. (I also want to take it as a metric learning problem but without enough time on that.)

For the No.2 question which is to predict the team (home 1 or away 0), I take it as a 2-class classification problem.

For the No.3 question which is to predict the ball position, I take it as a regression problem.

### Solutions

#### GBDT(Lightgbm) Solution


#### Deep Learning Solution


### Details

### Conclusion and Acknowledgement

Note that the raw data is difficult to handle, which spend the most of time of a solo player like me (can only work during spare time) on organizing data/input and features. With more time, I think I can try more ideas because the fundamental codes had been establised and more "algorithmic" works can be easily done.

At last, thanks the organizors and sponsors for providing a amazing detailed Opta data which I have never seen before. I always want to make some things with my knowledge of AI. This can be a great start for a soccer-fan researcher as me.



