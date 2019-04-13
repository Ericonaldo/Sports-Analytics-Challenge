# -*- coding:utf-8 -*-
from utils import *
import torch
import torch.autograd as autograd 
import torch.nn as nn         
import torch.nn.functional as F 
import torch.optim as optim


# team_feature_dim = 26
# event_feature_dim = 37
# player_feature_dim = 26

class TeamEventData():
    def __init__(self, path, t, team_id):
        self.path = path
        self.t = t
        # team seq
        self.team_seq = None
        self.team_seq_len = 0
        self.stat_team = None
        # last 10 event seq
        self.event_seq = None
        self.event_seq_len = 0
        self.stat_event = None
        # labels
        self.label_team = None
        self.label_xy = None
        files = os.listdir(path)
        csv_files = [i for i in files if 'csv' in i]
        label_file = [i for i in csv_files if str(t) in i]
        if team_id == 0:
            team_seq_file = 'tr_'+str(t)+'team0.tseq'
        else:
            team_seq_file = 'tr_'+str(t)+'team1.tseq'
        event_seq_file = 'tr_'+str(t)+'event.eseq'
        if len(label_file) == 0:
            print(str(t)+'seq do not exist!')
        else:
            label_file = pd.read_csv(path+label_file, header=None)
            self.label_team = label_file.iloc[0,1]
            self.label_xy  = label_file.iloc[0,2:3]
            self.team_seq = pd.read_csv(path+'team_seq'+team_seq_file).values # [time_step, team_feature_dim]
            self.team_seq_len = len(self.team_seq)
            self.event_seq = pd.read_csv(path+'event_seq'+event_seq_file).values # [time_step, team_feature_dim]
            self.event_seq_len = len(self.event_seq)


class PlayerData():
    def __init__(self, path, t):
        self.path = path
        self.t = t
        # team seq
        self.team_seq = None
        self.team_seq_len = 0
        self.stat_team = None
        # last 10 event seq
        self.event_seq = None
        self.event_seq_len = 0
        self.stat_event = None
        # player seq
        self.player_seq = None
        self.player_seq_len = 0
        self.stat_player = None
        # labels
        self.label_team = None
        self.label_xy = None
        self.label_player = None
                
        files = os.listdir(path)
        csv_files = [i for i in files if 'csv' in i]
        label_file = [i for i in csv_files if str(t) in i]
        if team_id == 0:
            team_seq_file = 'tr_'+str(t)+'team0.tseq'
        else:
            team_seq_file = 'tr_'+str(t)+'team1.tseq'
        event_seq_file = 'tr_'+str(t)+'event.eseq'
        if len(label_file) == 0:
            print(str(t)+'seq do not exist!')
        else:
            label_file = pd.read_csv(path+label_file, header=None)
            self.label_team = label_file.iloc[0,1]
            self.label_team = label_file.iloc[0,1]
            self.label_xy  = label_file.iloc[0,2:3]

            self.team_seq = pd.read_csv(path+'team_seq'+team_seq_file).values # [time_step, team_feature_dim]
            self.team_seq_len = len(self.team_seq)

            self.event_seq = pd.read_csv(path+'event_seq'+event_seq_file).values # [time_step, team_feature_dim]
            self.event_seq_len = len(self.event_seq)
            
        

class TeamEventDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        self.data = []
        event_seq = 'event_seq/'
        team_seq = 'team_seq/'
        player_seq = 'player_seq/'

        dirs = os.listdir(path)
        dirs = [i for i in dirs if '.' not in i]
        dirs = list(range(len(dirs))

        for d in dirs:
            for t in range(0, 12):
                t *= 7.5
                self.data.append(TeamEventData(path+d+'/', t, 0))
                self.data.append(TeamEventData(path+d+'/', t, 1))
        
        
    def __getitem__(self, index):

        team_seq, stat_team, event_seq, stat_event, label_team, label_xy = \
            self.data[index].x_team, self.data[index].stat_team, self.data[index].x_event, self.data[index].stat_event, self.data[index].label_team, self.data[index].label_xy

        return x_team, stat_team, x_event, stat_event, label_team, label_xy = data
    
    def __len__(self):
        return len(self.data)

class PlayerDataset(Dataset):

    def __init__(self, path):
        for d in dirs:
            for t in range(0, 12):
                t *= 7.5
            self.data.append(PlayerData(path+d+'/', t))

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        
    def __len__(self):
        return len(self.data)