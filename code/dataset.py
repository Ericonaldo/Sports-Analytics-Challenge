# -*- coding:utf-8 -*-
from utils import *
import torch
import torch.autograd as autograd 
import torch.nn as nn         
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset


# team_feature_dim = 26
# event_feature_dim = 37
# player_feature_dim = 26

total_team_seq = 100
total_event_seq = 10
total_player_seq = 30

class TeamEventData():
    def __init__(self, path, t, team_id):
        self.path = path
        self.t = round(float(t), 1)
        # team seq
        self.team_seq = None
        self.team_seq_len = np.array(0)
        self.stat_team = np.array([0])
        # last 10 event seq
        self.event_seq = None
        self.event_seq_len = np.array(0)
        self.stat_event = np.array([0])
        # labels
        self.label_team = None
        self.label_xy = None
        files = os.listdir(path)
        csv_files = [i for i in files if 'csv' in i]
        label_file = [i for i in csv_files if '_'+str(self.t) in i]
        if team_id == 0:
            team_seq_file = 'tr_'+str(self.t)+'_team0.tseq'
        else:
            team_seq_file = 'tr_'+str(self.t)+'_team1.tseq'
        event_seq_file = 'tr_'+str(self.t)+'_event.eseq'
        if len(label_file) == 0:
            print(str(self.t)+'seq do not exist!')
        else:
            label_file = pd.read_csv(path+label_file[0], header=None)
            self.label_team = np.array([int(label_file.iloc[0,1] == team_id)])
            self.label_xy  = np.array(label_file.iloc[0,2:]/100) # normalization
            team_seq = pd.read_csv(path+'team_seq/'+team_seq_file).values[0:total_team_seq] 
            self.team_seq_len = np.array(len(team_seq))
            # print(self.team_seq_len)
            # padding
            self.team_seq = np.pad(team_seq,((0,total_team_seq-len(team_seq)),(0,0)),'constant') # [total_team_seq, team_feature_dim]

            self.event_seq = pd.read_csv(path+'event_seq/'+event_seq_file).values # [time_step, team_feature_dim]
            self.event_seq_len = np.array(len(self.event_seq))           
        

class TeamEventDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        self.data = []
        event_seq = 'event_seq/'
        team_seq = 'team_seq/'
        player_seq = 'player_seq/'

        dirs = os.listdir(path)
        dirs = [i for i in dirs if '.' not in i]
        dirs = [str(i) for i in range(len(dirs))]

        for d in dirs:
            for t in tqdm(range(0, 11)):
                t *= 7.5
                self.data.append(TeamEventData(path+d+'/', t, 0))
                self.data.append(TeamEventData(path+d+'/', t, 1))
        
    def __getitem__(self, index):
        #index %= 2
        #print(index)
        team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, label_team, label_xy = \
            self.data[index].team_seq, self.data[index].team_seq_len, self.data[index].stat_team, self.data[index].event_seq, self.data[index].event_seq_len, self.data[index].stat_event, self.data[index].label_team, self.data[index].label_xy
        '''
        print('team_seq', np.shape(team_seq))
        print('team_seq_len', np.shape(team_seq_len))
        print('stat_team', stat_team)
        print('event_seq', np.shape(event_seq))
        print('event_seq_len', np.shape(event_seq_len))
        print('stat_event', stat_event)
        print('label_team', np.shape(label_team))
        print('label_xy', np.shape(label_xy))
        '''

        return team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, label_team, label_xy
        # return np.zeros([5, Config.team_feature_dim]), np.array(5), np.array([0]), np.zeros([10, Config.event_feature_dim]), np.array(10), np.array([0]), np.array([1]), np.array([0.5, 0.7]) # debug the model
    
    def __len__(self):
        return len(self.data)


class PlayerData():
    def __init__(self, path, t):
        self.path = path
        self.t = round(float(t), 1)
        # team seq
        self.team_seq = None
        self.team_seq_len = np.array(0)
        self.stat_team = np.array([0])
        # last 10 event seq
        self.event_seq = None
        self.event_seq_len = np.array(0)
        self.stat_event = np.array([0])
        # player seq
        self.player_seq = None
        self.player_seq_len = np.array(0)
        self.stat_player = np.array([0])
        # labels
        self.label_team = None
        self.label_xy = None
        self.label_player = None
        
        files = os.listdir(path)
        csv_files = [i for i in files if 'csv' in i]
        label_file = [i for i in csv_files if str(self.t) in i]
        if team_id == 0:
            team_seq_file = 'tr_'+str(self.t)+'_team0.tseq'
        else:
            team_seq_file = 'tr_'+str(self.t)+'_team1.tseq'
        event_seq_file = 'tr_'+str(self.t)+'_event.eseq'
        if len(label_file) == 0:
            print(str(self.t)+'seq do not exist!')
        else:
            label_file = pd.read_csv(path+label_file[0], header=None)
            self.label_team = np.array(label_file.iloc[0,1])
            self.label_xy  = np.array(label_file.iloc[0,2:])

            team_seq = pd.read_csv(path+'team_seq/'+team_seq_file).values # [time_step, team_feature_dim]
            self.team_seq_len = np.array(len(team_seq))
            # padding
            self.team_seq = np.pad(team_seq,((0,total_team_seq-len(team_seq)),(0,0)),'constant')  

            self.event_seq = pd.read_csv(path+'event_seq/'+event_seq_file).values # [time_step, team_feature_dim]
            self.event_seq_len = np.array(len(self.event_seq))

class PlayerDataset(Dataset):

    def __init__(self, path):
        for d in dirs:
            for t in tqdm(range(0, 11)):
                t *= 7.5
            self.data.append(PlayerData(path+d+'/', t))

    def __getitem__(self,index):
        pass
        return 
        
    def __len__(self):
        return len(self.data)