# -*- coding:utf-8 -*-
# basic
from datetime import datetime, timedelta
import datetime as dt
import time
import re
import sys
import pickle
import os
from tqdm import *
import multiprocessing
import math
import lxml
from lxml import etree

# data analysis and wrangling
import numpy as np
import pandas as pd
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import imgkit

# machine learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold,KFold
import torch
from torch import nn
# import xgboost
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import gc

data_path = "../data/XPSG - available resources/"
train_path = "../data/train/" # path of processed training data
valid_path = "../data/valid/" # path of processed validation data
processed_path = "../data/processed/"
model_path = '../models/'

train_dir = data_path+"French Ligue One 20162017 season - Match Day 1- 19/" # path of raw data
player_info = data_path+"Players and IDs - F40 - L1 20162017.xml"
test_before_change_eg = data_path+"Example test base before changes specified in the Rules -f24-24-2016-853139-eventdetails_test_hackathon_1.xml"
test_change_eg = data_path+"Example test base file - f24-24-2016-853285-eventdetails_test_hackathon_2.xml"

rules_team = [50,25,18,2,15,51,49,27]
rules_x_y = [50,25,18,27]
# ---------- general -------------
def get_player_data(is_changed=False):
    """
    Provide the player's data containing their ids, names, team ids and team names.
    args:
        is_changed: If True, then get a player list who have changed team; if not, then get all.
    return:
        player_list: The player list in DataFrame of pandas.
        
    """
    player_xml = lxml.etree.parse(player_info)
    
    player_id=[]
    player_name=[]
    team_id=[]
    team_name=[]
    
    if is_changed:
        new_team=[]
        leave_dates=[]
        for i in player_xml.xpath('//PlayerChanges//Player'):
            player_id.append(i.attrib['uID'])
            player_name.append(i.getchildren()[0].text)
            team_id.append(i.getparent().attrib['uID'])
            team_name.append(i.getparent().getchildren()[0].text)
            leave_dates.append(i.xpath("Stat[@Type='leave_date']")[0].text)
            new_team.append(i.xpath("Stat[@Type='new_team']")[0].text)
            
        save_name = "changed_player_data.csv"
        player_df = pd.DataFrame({'player_id':player_id, 'player_name':player_name, 
                                'old_team_id':team_id, 'old_team_name':team_name,
                                'leave_date':leave_dates,'new_team_name':new_team}) # Changed players' data
    else:
        jersey_num=[]
        join_date=[]
        # regular players
        for i in player_xml.xpath('//SoccerDocument/Team/Player'):
            player_id.append(i.attrib['uID'])
            player_name.append(i.xpath("Name")[0].text)
            team_id.append(i.getparent().attrib['uID'])
            team_name.append(i.getparent().attrib['short_club_name'])
            jersey_num.append(i.xpath("Stat[@Type='jersey_num']")[0].text)
            join_date.append(i.xpath("Stat[@Type='join_date']")[0].text)
            """
            jersey=""
            join=""
            for j in i.getchildren():
                if 'Type' in j.attrib.keys() and j.attrib['Type']=="jersey_num":
                    jersey = j.text
                elif 'Type' in j.attrib.keys() and j.attrib['Type']=="join_date":
                    join = j.text
            jersey_num.append(jersey)
            join_date.append(join)
            """
        
        # changed players
        for i in player_xml.xpath('//PlayerChanges//Player'):
            if i.attrib['uID'] not in player_id:
                player_id.append(i.attrib['uID'])
                player_name.append(i.xpath("Name")[0].text)
                team_id.append(i.getparent().attrib['uID'])
                team_name.append(i.getparent().getchildren()[0].text)
                jersey_num.append(i.xpath("Stat[@Type='jersey_num']")[0].text)
                join_date.append(i.xpath("Stat[@Type='join_date']")[0].text)
           
        save_name = "all_player_data.csv"
        player_df = pd.DataFrame({'player_id':player_id, 'player_name':player_name, 'jersey_num':jersey_num,
                                'team_id':team_id, 'team_name':team_name, 'join_date':join_date}) # All players' data
    
    player_df.to_csv(processed_path+save_name, index=False)
    return player_df

def get_play_time_in_one_game(comp_xml):
    """
    Provide the player's playing time data in one given game containing their ids, and on time, off time and playing time in this game.
    Ignore all the secs.
    args:
        comp_xml: the xml file of given game.
    return:
        play_time_df: The playing time data in DataFrame of pandas.
        
    """
    players=[]
    players_ele = comp_xml.xpath("//Q[@qualifier_id='30']/@value")
    for i in range(0,len(players_ele)):
        players_ele[i] = ['p'+j for j in players_ele[i].split(', ')]
        players += players_ele[i]
    players_num = len(players)    
    
    play_time_df = pd.DataFrame({"player_id":players,
                                 #"player_name":[],
                                "on_time":list(np.zeros(players_num)),
                                "on_period":list(np.zeros(players_num)),
                                "off_time":list(np.zeros(players_num)),
                                "off_period":list(np.zeros(players_num)),
                                "playing_time":list(np.zeros(players_num))})
    
    play_time_df.on_time = play_time_df.off_time = play_time_df.playing_time = pd.Timedelta(minutes=0)
    play_time_df.on_period = play_time_df.on_period.astype(int)
    play_time_df.off_period = play_time_df.off_period.astype(int)
    
    fir_half_time = int(comp_xml.xpath("//Event[@type_id='30' and @period_id='1']/@min")[0]) # find the time of first half
    sec_half_time = int(comp_xml.xpath("//Event[@type_id='30' and @period_id='2']/@min")[0]) # find the time of second half

    first_players = comp_xml.xpath("//Q[@qualifier_id='30']/@value") # find the first players
    for i in range(0,len(first_players)): # record the first players' on and off time, ignore the secs
        first_players[i] = ['p'+j for j in first_players[i].split(', ')[0:11]]
        play_time_df.loc[play_time_df.player_id.isin(first_players[i]), 'on_time'] = pd.Timedelta(minutes=0)
        play_time_df.loc[play_time_df.player_id.isin(first_players[i]), 'on_period'] = 1
        play_time_df.loc[play_time_df.player_id.isin(first_players[i]), 'off_time'] = pd.Timedelta(minutes=int(sec_half_time))
        play_time_df.loc[play_time_df.player_id.isin(first_players[i]), 'off_period'] = 2

    subs_players = comp_xml.xpath("//Event[@type_id='19']") # find the substitution players
    subs_ed_players = comp_xml.xpath("//Event[@type_id='18']") # find the substituted players   
    for i in subs_players: # record the substitution players' on and off time, ignore the secs
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'on_time'] = pd.Timedelta(minutes=int(i.attrib['min']))
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'on_period'] = int(i.attrib['period_id'])
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'off_time'] = pd.Timedelta(minutes=int(sec_half_time))
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'off_period'] = 2
    for i in subs_ed_players: # sum the substituted players' playing time, ignore the secs
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'off_time'] = pd.Timedelta(minutes=int(i.attrib['min']))
        play_time_df.loc[play_time_df.player_id==('p'+i.attrib['player_id']), 'off_period'] = int(i.attrib['period_id'])
    
    # player on and off in the same half
    play_time_df.loc[play_time_df.on_period == play_time_df.off_period, 'playing_time'] = \
        play_time_df.loc[play_time_df.on_period == play_time_df.off_period, 'off_time'] - \
        play_time_df.loc[play_time_df.on_period == play_time_df.off_period, 'on_time']
    # player on in the first half and off in the second
    play_time_df.loc[play_time_df.on_period != play_time_df.off_period, 'playing_time'] = \
        play_time_df.loc[play_time_df.on_period != play_time_df.off_period, 'off_time'] - \
        pd.Timedelta(minutes=45) + \
        pd.Timedelta(minutes=fir_half_time) - \
        play_time_df.loc[play_time_df.on_period != play_time_df.off_period, 'on_time']
        
    return play_time_df   
        
def get_play_time(player_df):
    """
    Provide the player's total playing time data in the training set containing their ids, names, and total playing time.
    Ignore all the secs.
    args:
        player_df: players who's playing time need to be shown.
    return:
        total_play_time_df: The total playing time data in DataFrame of pandas.
        
    """
    files= os.listdir(train_dir)
    files = [i for i in files if i[0:3]=='f24']
    l = len(files)
    
    total_play_time_df = pd.DataFrame({"player_id":player_df.player_id,
                                 "player_name":player_df.player_name,
                                 "total_playing_time":list(np.zeros(len(player_df)))})
    total_play_time_df.total_playing_time = pd.Timedelta(minutes=0)
    
    for file in files:
        comp_xml = lxml.etree.parse(train_dir+file)
        
        play_time_df = get_play_time_in_one_game(comp_xml)
        
        # record the total playing time od each player
        for i in range(0, len(play_time_df)):
            total_play_time_df.loc[total_play_time_df.player_id==play_time_df.iloc[i].player_id,'total_playing_time'] += \
                play_time_df.iloc[i].playing_time
        
    save_name = "total_play_time_data.csv"
    total_play_time_df.to_csv(processed_path+save_name, index=False)
    return total_play_time_df

def suff_plyr(choice_xml):
    """
    Provide the list of sufficient player ids in the chosen game.
    args:
        choice_xml: the xml of the chosen game whose type is lxml.etree._ElementTree.
    return:
        suff_plyr_list: the list of sufficient player ids in the chosen game.
        
    """
    player_in_the_game = []
    players_ele = choice_xml.xpath("//Q[@qualifier_id='30']/@value")
    for i in range(0,len(players_ele)):
        players_ele[i] = ['p'+j for j in players_ele[i].split(', ')]
        player_in_the_game += players_ele[i]
    
    # Get all player list and changed player list
    if (os.path.exists(processed_path+"all_player_data.csv")):
        all_player_df = pd.read_csv(processed_path+"all_player_data.csv")
    else:
        all_player_df = get_player_data()
    all_player_df.join_date = all_player_df.join_date.apply(
            lambda x:pd.to_datetime(x, format="%Y-%m-%d"))
    join_date_plyr = list(
        all_player_df[all_player_df.join_date < pd.to_datetime('2017-01-01', format="%Y-%m-%d")].player_id)
    
    if (os.path.exists(processed_path+"changed_player_data.csv")):
        change_player_df = pd.read_csv(processed_path+"changed_player_data.csv")
    else:
        change_player_df = get_player_data(True)
    change_player_df.leave_date = change_player_df.leave_date.apply(
            lambda x:pd.to_datetime(x, format="%Y-%m-%d"))
        
    # Get playing time data of all players
    if (os.path.exists(processed_path+"total_play_time_data.csv")):
        total_play_time_data = pd.read_csv(processed_path+"total_play_time_data.csv")
        total_play_time_data.total_playing_time = total_play_time_data.total_playing_time.apply(
            lambda x:pd.Timedelta(x))
    else:
        total_play_time_data = get_play_time(all_player_df)
    suff_time_plyr = list(total_play_time_data[total_play_time_data.total_playing_time > pd.Timedelta(minutes=800)].player_id)
    
    # Filter sufficient players
    suff_plyr_list = [ i for i in player_in_the_game if ((i in suff_time_plyr) and
                  (i in join_date_plyr) and (i in player_in_the_game))]
    
    return suff_plyr_list

def suff_plyr_event(pick_events, suff_plyr_list):
    """
    Justify if there are players in suff_plyr_list is in pick_events.
    args:
        pick_events: the picked events of the chosen game.
    return:
        Bool type.
        
    """
    for i in pick_events:
        if 'player_id' in i.attrib:
            player_id = 'p'+i.attrib['player_id']
            if player_id in suff_plyr_list:
                return True
    return False

def construct_one_val(choice_xml):
    """
    Construct one validation set.
    args:
        choice_xml: the xml of the chosen game whose type is lxml.etree._ElementTree.
    return:
        val_xml: the xml of one validation set whose type is lxml.etree._ElementTree.
        label_csv: the csv file of label results of this validation set.
        
    """
    suff_plyr_list = suff_plyr(choice_xml)
    if suff_plyr_list==[]:
        return [None, None]
    
    pick_plyr_id = suff_plyr_list[rnd.randint(0,len(suff_plyr_list)-1)] # randomly choose a sufficient player
    events = choice_xml.xpath('//Event')
    half_type = rnd.randint(1,2)
    half_events = [i for i in events if i.attrib['period_id']==str(half_type)] # randomly choose one half and its events
    min_pick = rnd.randint(0,45-15) + (half_type-1)*45  # randomly choose t and pick events
    pick_events = [i for i in half_events if min_pick<=int(i.attrib['min'])<=(min_pick+15)]
    # do while until there are suff_plyr in events
    while not suff_plyr_event(pick_events, suff_plyr_list):
        half_type = rnd.randint(1,2)
        half_events = [i for i in events if i.attrib['period_id']==str(half_type)] # randomly choose one half and its events
        min_pick = rnd.randint(0,45-15) + (half_type-1)*45  # randomly choose t and pick events
        pick_events = [i for i in half_events if min_pick<=int(i.attrib['min'])<=(min_pick+15)]

    game = choice_xml.xpath('//Game')[0]
    home_id = game.attrib['home_team_id']
    away_id = game.attrib['away_team_id']
    
    
    game.attrib['timestamp'] = "" 
    for i in game.attrib.keys(): # substitute before info
        game.attrib[i] = ""
        
    for i in pick_events: # substitute team_id, player_id and other info
        if 'player_id' in i.attrib:
            i.attrib['player_id'] = str(int(i.attrib['player_id']==pick_plyr_id))
        if 'team_id' in i.attrib:
            i.attrib['team_id'] = str(int(i.attrib['team_id']==home_id))
        if 'id' in i.attrib:
            i.attrib['id'] = ""
        if 'timestamp' in i.attrib:
            i.attrib['timestamp'] = ""
        if 'last_modified' in i.attrib:
            i.attrib['last_modified'] = ""
        if 'version' in i.attrib:
            i.attrib['version'] = ""
        
        for j in i.xpath('Q'):
            j.attrib['id'] = ""
            #When Type ID=140 and 141, qualifier_id=140/141 appears, replace the values by ""
            if ('qualifier_id' in j.attrib) and (j.attrib['qualifier_id'] == "140" or j.attrib['qualifier_id'] == "141"):
                j.attrib['qualifier_id'] = j.attrib['value'] = ""
        
    for i in pick_events[0:-10]: # substitute ball coor.
        i.attrib['x'] = i.attrib['y'] = "0.0"
    
    for i in pick_events[-10:]: # substitute the last 10 events
        i.attrib['outcome'] = ""
        for j in i.xpath('Q'):
            j.getparent().remove(j) # same as example
        #     j.attrib['qualifier_id'] = j.attrib['value'] = ""
    
    next_event_idx = choice_xml.xpath('//Event').index(pick_events[-1])+1 
    next_event = events[next_event_idx] # next event
    results = str(pick_plyr_id[1:]) + ',' + str(int(next_event.attrib['team_id']==home_id)) + ',' + \
                            str(next_event.attrib['x'])+ ',' + str(next_event.attrib['y']) # the label results
    label_csv = str(results)
    
    other_event = list(set(choice_xml.xpath('//Event')).difference(set(pick_events)))
    for j in other_event:
            j.getparent().remove(j)            
    val_xml = choice_xml # Construct val_xml
    
    return [val_xml, label_csv]

def construct_val_sets(train_dir=train_dir, save_path=valid_path, val_num=10):
    """
    Construct numbers of validation set.
    args:
        save_path: the path of saving validation sets.
        val_num: the number of validation sets.
    return:
        a bool if the construction is success.
        
    """
    files= os.listdir(train_dir)
    files = [i for i in files if i[0:3]=='f24']
    l = len(files)
    
    for i in tqdm(range(val_num)):
        choice_file_idx = rnd.randint(0,l-1)
        # do while loop until returns are not None
        choice_xml = lxml.etree.parse(train_dir+files[choice_file_idx]) # randomly choose one game
        val_xml, label_csv = construct_one_val(choice_xml)
        while val_xml==None:
            choice_file_idx = rnd.randint(0,l-1)
            choice_xml = lxml.etree.parse(train_dir+files[choice_file_idx]) # randomly choose one game
            val_xml, label_csv = construct_one_val(choice_xml)
        try:
            val_xml.write(save_path+'val_' + str(i) + '.xml')
            with open(save_path+'label_' + str(i) + '.csv', 'w') as f:
                f.write(label_csv)
        except IOError:
            print("Write error!")
            return False


def draw_corr(df):
    table_html = df.corr(
        method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1).render()
    imgkit.from_string(table_html, 'out.jpg')


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# ---------- for gbdt models -------------
def map_type_id(raw_id): 
    map_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 25.0, 27.0, 28.0, 30.0, 32.0, 40.0, 41.0, 42.0,
                43.0, 44.0, 45.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,
                59.0, 60.0, 61.0, 65.0, 68.0, 70.0, 74.0]
    if raw_id in map_list:
        return map_list.index(raw_id)
    else:
        return len(map_list)+1

def construct_one_ball_team_df(choice_xml):
    """
    Construct one training set of ball pos and team of given game.
    args:
        choice_xml: the xml of the chosen game whose type is lxml.etree._ElementTree.
    return:
        train_df: the training set of ball pos of given game with its features.
        
    """
    game = choice_xml.xpath('//Game')[0]
    home_id = game.attrib['home_team_id']
    away_id = game.attrib['away_team_id']
    
    train_df = pd.DataFrame({#"game_id":[],
                             "event_no":[],
                             "period_id":[],
                             "min":[],
                             "sec":[],                           
                             #"event_id":[],
                             "type_id":[],
                             #"player_id":[],
                             "team_id":[],
                             "keypass":[],
                             "assist":[],
                             "ball_related":[],
                             #"ball_slope":[],
                             "x":[],
                             "y":[],
                             })
    
    events = choice_xml.xpath("//Event[@period_id='1' or @period_id='2']") 
     
    last_x = last_y = np.nan    
    for i in range(len(events)):
        ball_related = 1
        keypass = int('keypass' in events[i].attrib)
        assist = int('assist' in events[i].attrib)       
        x = float(events[i].attrib['x'])
        y = float(events[i].attrib['y'])
        if (x==y==0):
            ball_related = 0
            x = last_x # get the last position of the ball
            y = last_y

        # TODO need? 
        '''
        if (int(events[i].attrib['period_id'])-1)^(int(events[i].attrib['team_id']==home_id)) and (ball_related == 1):
            x = 100-x
            y = 100-y
        '''
        '''
        # ball slope 
        if (y - last_y) == 0:
            ball_slope = 1
        else:
            ball_slope = (x - last_x) / (y - last_y)
        if ball_related == 1: # update last_x, last_y
            last_x = x
            last_y = y
        elif ball_related == 0:
            ball_slope = 0
        '''
        
        temp = pd.DataFrame({"event_no":[i],
                             "period_id":[int(events[i].attrib['period_id'])-1],
                             "min":[int(events[i].attrib['min'])],
                             "sec":[int(events[i].attrib['sec'])],
                             #"event_id":[int(events[i].attrib['event_id'])],
                             "type_id":[int(events[i].attrib['type_id'])],
                             #"player_id":[int(events[i].attrib['player_id'])],
                             "team_id":[int(events[i].attrib['team_id']==home_id)],
                             "keypass":[keypass],
                             "assist":[assist],
                             "ball_related":[ball_related],
                             #"ball_slope":[ball_slope],
                             "x":[x],
                             "y":[y],                         
                             })
        train_df = pd.concat([train_df, temp])
    
    return train_df


def construct_ball_team_df(save_path=train_path):
    """
    Construct the training set of ball pos and team id.
    args:
        save_path: the path of saving validation sets.
    return:
        train_df: the training set whose type is pd.DataFrame.
        
    """
    files= os.listdir(train_dir)
    files = [i for i in files if i[0:3]=='f24']
    
    train_df = pd.DataFrame()
    for i in tqdm(range(len(files))):
        game_xml = lxml.etree.parse(train_dir+files[i])
        one_train_df = construct_one_ball_team_df(game_xml)
        one_train_df.insert(0,'game_id',i*np.ones([len(one_train_df)]))
        # one_train_df['game_id'] = i
        train_df = pd.concat([train_df, one_train_df])
    train_df.reset_index(drop=True, inplace=True)
    # train_df['type_id'] = train_df['type_id'].apply(map_type_id)   
    
    ### space feature
    ## field
    train_df['field'] = (train_df['x']>50).astype('int32') # left=0, right=1, (0,0)=2
    train_df.loc[train_df[train_df.ball_related==0].index, 'field'] = 2
    ## zone         
    train_df['left_zone'] = ((0<=train_df['y']) & (train_df['y']<=21.1)).astype('int32')
    train_df['middle_zone'] = ((21.1<train_df['y']) & (train_df['y']<78.9)).astype('int32')
    train_df['right_zone'] = ((78.9<=train_df['y']) & (train_df['y']<=100)).astype('int32')
    # left=0, middle=1, right=2, (0,0)=3
    train_df['zone'] = train_df['right_zone']*0 + train_df['middle_zone']*1 + train_df['right_zone']*2
    train_df.drop(['left_zone', 'middle_zone', 'right_zone'], inplace=True, axis=1)
    train_df.loc[train_df[train_df.ball_related==0].index, 'zone'] = 3 
    ## penal zone            
    train_df['penal_zone_l'] = (((0<=train_df['x']) & (train_df['x']<=17)) & ((21.1<=train_df['y']) & (train_df['y']<=78.9))).astype('int32')
    train_df['penal_zone_r'] = (((83<=train_df['x']) & (train_df['x']<=100)) & ((21.1<=train_df['y']) & (train_df['y']<=78.9))).astype('int32')
    # left=0, right=1, (0,0)=2
    train_df['penal_zone'] = train_df['penal_zone_l']*0 + train_df['penal_zone_r']*1
    train_df.drop(['penal_zone_l', 'penal_zone_r'], inplace=True, axis=1)
    train_df.loc[train_df[train_df.ball_related==0].index, 'penal_zone'] = 2
                  
    ## is or not penal_point
    train_df['penal_point'] = ((train_df['x']==88.5) & (train_df['y']==50)).astype('int32')
    train_df['ball_pos'] = np.sqrt(train_df['x']*train_df['x']+train_df['y']*train_df['y'])
    
    ### time feature
    train_df['game_time'] = train_df['min']*60+train_df['sec']
    train_df['time_dis_last_event'] = train_df['game_time'].shift(1)
    train_df['period_last_event'] = train_df['period_id'].shift(1)
    train_df['game_last_event'] = train_df['game_id'].shift(1)
    train_df['time_dis_last_event'] = train_df.game_time - train_df.time_dis_last_event
    train_df.loc[train_df.period_id != train_df.period_last_event,'time_dis_last_event'] = 0
    train_df.loc[train_df.game_id != train_df.game_last_event,'time_dis_last_event'] = 0
    train_df = train_df.drop(['period_last_event','game_last_event'], axis=1)
    
    df_fea = train_df.copy()
    df_fea['event_no'] = df_fea['event_no'] + 1
    train_df = pd.merge(
        train_df, 
        df_fea.rename(columns=dict(zip(df_fea.columns[3:], [f'last_{col}' for col in df_fea.columns[3:]]))),
        how='left',
        left_on=['event_no','game_id','period_id'], 
        right_on=['event_no','game_id','period_id']
    )
    
    # train_df[['next_ball_related']] = train_df[['ball_related']].shift(-1)
    train_df[['next_x','next_y']] = train_df[['x','y']].shift(-1)
    train_df[['next_team']] = train_df[['team_id']].shift(-1)
    train_df[['next_team']] = (train_df['team_id']==train_df['next_team']).astype(int)
    
    train_df['period_next_event'] = train_df['period_id'].shift(-1)
    train_df['game_next_event'] = train_df['game_id'].shift(-1)
    train_df.loc[train_df.period_id != train_df.period_next_event,['next_x','next_y','next_team']] = np.nan
    train_df.loc[train_df.game_id != train_df.game_next_event,['next_x','next_y','next_team']] = np.nan
    train_df = train_df.drop(['period_next_event','game_next_event'], axis=1)

    train_df.to_csv(save_path+"ball_team_df.csv", index=False)
    return train_df

# ---------- for RNN models -------------



