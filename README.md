# Sports-Analytics-Challenge

Sports Analytics Challenge sponsored by PSG. Link: https://www.agorize.com/zh/challenges/xpsg/pages/regles-du-jeu-futur?lang=en

==============================

# Project Organization
------------
    ├── README.md          					      <- The top-level README for developers using this project.
    ├── data
    │   ├── train          						  <- Training data used in experiments.
    │   ├── processed      						  <- Data used for indication and further processing during experiments.
    │   ├── valid      						      <- Valid data used in experiments.
    │   └── XPSG - available resources            <- The original, immutable data dump (after unzipping).
    │
    │
    ├── models                                    <- Trained and serialized models, contained lightGBM models and RNN models
    │
    │
    ├── fig                                       <- Some useful figures for indication during experiments.
    │
    │
    ├── code             		                  <- Python script source code and Jupyter notebook source code for use in this project.
    │   │
    │   ├── main_psgx.py                          <- The required python code file which contains Result(xml1) function.
    │   │
    │   ├── install_psgx.py  					  <- The required python code file which is used for installing requirements.
    │   │
    │   ├── utils.py                              <- Useful tool codes used in experiments, containing some data preprocessing, feature extraction and other tools.
    │   │
    │   ├── train_gbdt.py                         <- Training a gbdt model with lightGBM which give the answer of the last 3 questions. See details below.
    │   │
    │   ├── evaluation_gbdt.py  				  <- Testing the above gbdt model on valid sets and show the evaluation results. See details below.
    │   │
    │   ├── evaluation_rules.py  				  <- Testing naive rules on valid sets and show the evaluation results. This result gives a baseline.
    │   │
    │   ├── dataset.py                            <- This file define some class of pytorch Dataset, used for training of deep learning models with pytorch.
    │   │
    │   ├── train_team_event_rnn.py               <- Training rnn models with pytorch which give the answer of the last 3 questions. See details below.
    │   │ 
    │   ├── train_player_rnn.py                   <- Training rnn models with pytorch which give the answer of the first question. See details below.
    │   │ 
    │   ├── evaluation_rnn.py                     <- Testing the above rnn models on valid sets and show the evaluation results. See details below.
    │   │ 
    │   ├── train_xyt_gbdt.ipynb                  <- Some draft codes during experiments.
    │   │ 
    │   ├── construct_train.ipynb                 <- Some draft codes during experiments.
    │   │ 
    └	└── test.ipynb                            <- Some draft codes during experiments.

# Instructions

To Train the gbdt model:

1. Enter your terminal

2. python train_gbdt.py  --corr --xymodel ('corr' means saving corr img, '\*model' means which model do you want to train)

To Evaluate the gbdt model:

1. Enter your terminal

2. python evaluation_gbdt.py --valid 100 --tmodel -- xymodel --pmodel ('valid' means the valid set num, '*model' means which model do you want to test)

To Evaluate your own gbdt model:

1. Enter your terminal

2. python

3. >>from utils import *

4. >>construc a evaluation function following the evaluate_gbdt(models=[p_model, x_model, y_model, t_model]) in evaluation_gbdt

5. >>construct_val_sets(val_num=100) (set the validation number that you need)

6. >>your_own_eval_func(models=[p_model, x_model, y_model, t_model]) (remember to load your models first, args could be None)

# Design details

In this competition, I take the 4 (or 3) questions dividely, and I propose 2 solutions on those problems.

For the first question which is to predict a player id, 

Note that the raw data is difficult to handle, which spend the most of time of a solo player like me on organizing data/input and features. With more time, I think I can try more ideas because the fundamental codes had been establised and more "algorithmic" works can be easily done.

At last, thanks the organizors and sponsors for providing a amazing detailed Opta data which I have never seen before. I always want to make some things with my knowledge of AI. This can be a great start for a soccer-fan researcher as me.



