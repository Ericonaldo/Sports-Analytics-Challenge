# Sports-Analytics-Challenge

Sports Analytics Challenge sponsored by PSG. Link: https://www.agorize.com/zh/challenges/xpsg/pages/regles-du-jeu-futur?lang=en

==============================

## Project Organization
------------
    ├── README.md                             <- The top-level README for developers using this project.
    ├── LICENCE                               <- The Licence of this project.
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

Construct training sets:

1. Enter your terminal and cd into "code" dir

2. `python`

3. `from utils import *`

4. `construct_train_sets()`

5. `trans_train_set_to_seq_data()`

Train the deep model:

1. Enter your terminal and cd into "code" dir

2. `python train_team_event_rnn.py  --load` ('load' means if load a pre trained model (not need if train fron sketch), this model is for question 2-4)

3. `python train_player_rnn.py  --load` ('load' means if load a pre trained model (not need if train fron sketch), this model is for question 1))

Evaluate the deep model:

1. Enter your terminal and cd into "code" dir

2. `python evaluation_rnn.py --valid 100 --test_p --test_xyt --xyt_epoch 360 --p_epoch 500` ('valid' means the valid set num (if is not constructed before), 'test_\*' means which model do you want to test, 'xyt_epoch' means which epoch of model do you want to load for 'team_event_rnn' model, default as 360, 'p_epoch' means which epoch of model do you want to load for 'player_rnn' model, default as 500)


## Solution details

### Introduction

In this competition, I take the 4 (or 3) questions respectively, and propose 2 solutions for those problems.

For the No.1 question which is to predict the player id, I take it as a Fine-grained classification problem, for which the high level class is the position of a player and it gives information when predicting the low level class -- the true players. (I also want to take it as a metric learning problem but without enough time on that.)

For the No.2 question which is to predict the team (home 1 or away 0), I take it as a 2-class classification problem.

For the No.3 question which is to predict the ball position, I take it as a regression problem.

### Solutions and Results

#### GBDT(Lightgbm) Solution

This solution is only for the last 3 questions.

First, I construct the training set from all the competition xml data, in which every event is taken as a sample. For each sample, I construct space features, time features and detailed features from the raw event data like 'field', 'zone', 'time dis from the last event' and so on (codes are in /code/utils.py/construct_ball_team_df()). The targets/labels are: if the next event team id the same as current; the next x coordinate; the next y coordinate. 

Then, I train 3 gbdt models to predict above targets respectively. The best results I test offline is  score_team=0.4512471655328798, loss_xy=1294.2247845804986. Note that the score_team is worse than a random baseline 0.5.

The feature importance and faeture correlation are as follows.

![feature_importance_bst_next_ball_related](/fig/feature_importance_bst_next_ball_related.png)
![feature_importance_x](/fig/feature_importance_x.png)
![feature_importance_y](/fig/feature_importance_y.png)
![corr](/fig/out.jpg)

#### Deep Learning Solution

This solution is only for all the 4 questions.

For time limit, I can only design some simple networks without further fine-tuning.

For player id prediction, I choose a RNN (GRU) model to model the feature of events of this particular player, and another RNN (GRU) model to model the feature of events of his team. Then concat and through some FC layer to get a high level class and a low level class. The loss are weighted Cross Entropy loss. To inference from a given game xml, I only need the low level class (Codes are in /code/utils.py/train_player_rnn(), /code/utils.py/evaluation_rnn.py)

For team id and coordinate predictions, I choose a RNN (GRU) model to model the feature of events of the two teams, and another RNN (GRU) model to model the feature of the last 10 events. Then concat and through some FC layer. For each team, he get a confidence of "if the next event is mine" and the related (x, y) coordinates. The loss are a combined weighted Binary Cross Entropy loss and MSE loss. To inference from a given game xml, I exact 2 sequence of 2 teams respectively, the through the network and see whose confidence is bigger, then the results follow the bigger one. (Codes are in /code/train_team_event_rnn.py, /code/utils.py/evaluation_rnn.py)

Then I construct the training set from all the competition xml data.

For player id prediction, I choose a sequence of a sufficient player (who satisfy the requirements) from one game as a sample; for team id and coordinate predictions, I choose the start min as [0, 7.5, 15, ..., 75] for every competition and take every 15min event sequence as a team sequence sample, and every last 10 event sequence as a event sequence sample.

For each sample, I construct space features, time features and detailed features (codes are in /code/utils.py/construct_player_seq(), /code/utils.py/construct_team_seq() and /code/utils.py/construct_event_seq()). The targets/labels are: if the next event is mine; the next x coordinate; the next y coordinate. 

Then, I train 2 models to predict above targets respectively. The best results I test offline is score_team=0.5555555555555556, loss_xy=2209.658194428966. Note that the score_team is better however the regression are worse. The training curve are as follows, looks like great, but seems overfitted...

![team_event_rnn](/fig/team_event_rnn.png)
![player_rnn](/fig/player_rnn.png)

The inference time for one sample is less than 3 seconds because I use a mixture prediction of GBDT and RNN (GBDT needs 0.5s/sample and RNN 1.5s/sample). Experimens are all on a laptop with a Intel i5 8x CPU.

### Questions and Challenges

There are few problems and challenges in this task.

1. The corrdinate is related to the team. If we normalize the corrdinate based on period and team id, then we have to determine which team is related to the next event first before getting a true prediction, which introduce a tough cumulative error.

2. The corrdinate is related to the ball related events. Note that about 1/10 events are off-ball events which means they are not related with the ball ((x, y) = (0, 0)) and if it occurs in target then one have to predict if the next event is related to the ball, which is difficult for unbanlanced samples. Even though this event is rare (actually not 'that' rare), it can significantly influence the overall results (for the huge MSE error).

3. If there are additional statistic provided, it is better sice one do not need to spend time on calculating though these xml files.

### Future Work

1. More statistics can be added into the model.

2. Make the team prediction as an additional task for the player rnn network.

3. Consider a proper way to normalize the corrdinates.

4. Train different models based on the last event type. For example, to predict the next event, train one model for 'Pass' event, one model for 'Ball Recovery' and so on.

5. Consider and construct more features.

6. Test more proper models.


### Conclusion and Acknowledgement

Note that the raw data is difficult to handle, which spend the most of time of a solo player like me (can only work during spare time) on organizing data/input and features. With more time, I think I can try more ideas because the fundamental codes had been establised and more "algorithmic" works can be easily done. 

At last, thanks the organizers and sponsors for providing an amazing detailed Opta data which I have never seen before. I always want to make some things with my knowledge of AI. This can be a great start for a soccer-fan researcher as me (who want a chance to watch the live in the stadium, haha...).

## Mainly used packages

lxml==4.3.2
lightgbm==2.2.2
matplotlib==3.0.2
numpy==1.16.2
pandas==0.23.4
torch==1.0.1.post2
torchvision==0.2.2
tqdm==4.30.0
scikit-learn==0.20.1
seaborn==0.9.0

