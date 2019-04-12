# -*- coding:utf-8 -*-
from utils import *
import argparse

'''
pname = 'pmodel.pkl'
xname = 'xmodel.pkl'
yname = 'ymodel.pkl'
tname = 'tmodel.pkl'
next_ball_related_name = 'next_ball_related_name.pkl'
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--seq', action='store_true', help='is or not construct sequence data now')
    parser.add_argument('--pmodel', action='store_true')
    parser.add_argument('--xymodel', action='store_true')
    parser.add_argument('--tmodel', action='store_true')
    
    args = parser.parse_args()
    
    ## model of x,y and team
    if args.seq:
        trans_train_set_to_seq_data()
    
    