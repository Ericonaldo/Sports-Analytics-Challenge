# Sports-Analytics-Challenge

Sports Analytics Challenge sponsored by PSG. Link: https://www.agorize.com/zh/challenges/xpsg/pages/regles-du-jeu-futur?lang=en

To Train the gbdt model:

1. Enter your terminal

2. python train_gbdt.py  --corr --xymodel ('corr' means saving corr img, '*model' means which model do you want to train)

To Evaluate the gbdt model:

1. Enter your terminal

2. python evaluation_gbdt.py --valid 100 --tmodel -- xymodel --pmodel ('valid' means the valid set num, '*model' means which model do you want to test)

To Evaluate your own gbdt model:

1. Enter your terminal

2. python

3. >>from utils import *

4. >>from evaluation_gbdt import evaluate_gbdt

5. >>construct_val_sets(val_num=100) (set the validation number that you need)

6. >>evaluate_gbdt(models=[p_model, x_model, y_model, t_model]) (remember to load your models first, args could be None)