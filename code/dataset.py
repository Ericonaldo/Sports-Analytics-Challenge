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

total_team_seq = 150
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

            self.event_seq = pd.read_csv(path+'event_seq/'+event_seq_file).values[0:total_event_seq] # [time_step, team_feature_dim]
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
        #index %= 2000
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
        '''
        if index ==0 :
            return np.zeros([5, Config.team_feature_dim]), np.array(5), np.array([0]), np.zeros([10, Config.event_feature_dim]), np.array(10), np.array([0]), np.array([1]), np.array([0.2, 0.7]) # debug the model

        if index ==1 :
            return np.ones([5, Config.team_feature_dim]), np.array(4), np.array([0]), np.zeros([10, Config.event_feature_dim]), np.array(10), np.array([0]), np.array([0]), np.array([0.8, 0.3]) # debug the model
        '''
        return team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, label_team, label_xy
    
    def __len__(self):
        return len(self.data)


class PlayerData():
    def __init__(self, path, player_seq_file, all_player_df, suff_plyr):
        self.path = path
        self.player_seq_file= player_seq_file
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
        self.label_player = None
        self.label_pos = None

        self.team_id = int(self.player_seq_file[-6]) # tr_0.0_p37740.0.pseq
        if self.player_seq_file[6]=='_':
            self.t = self.player_seq_file[3:6]
        else:
            self.t = self.player_seq_file[3:7] 

        if self.team_id == 0:
            team_seq_file = 'tr_'+str(self.t)+'_team0.tseq'
        else:
            team_seq_file = 'tr_'+str(self.t)+'_team1.tseq'
        event_seq_file = 'tr_'+str(self.t)+'_event.eseq'

        files = os.listdir(path)
        csv_files = [i for i in files if 'csv' in i]
        label_file = [i for i in csv_files if str(self.t) in i]

        if len(label_file) == 0:
            print(str(self.t)+' seq do not exist!')
        else:
            #labels
            label_file = pd.read_csv(path+label_file[0], header=None)
            player_id = str(label_file.iloc[0,0])
            player_pos = all_player_df[all_player_df.player_id==('p'+player_id)].iloc[0]['position']
            self.label_pos = np.array(Config.pos_name.index(player_pos))
            
            self.label_player = np.array(suff_plyr.index('p'+player_id))

            ## team seq
            team_seq = pd.read_csv(path+'team_seq/'+team_seq_file).values[0:total_team_seq] # [time_step, team_feature_dim]
            self.team_seq_len = np.array(len(team_seq))
            # padding
            self.team_seq = np.pad(team_seq,((0,total_team_seq-len(team_seq)),(0,0)),'constant')  
            ## event seq
            self.event_seq = pd.read_csv(path+'event_seq/'+event_seq_file).values[0:total_event_seq] # [time_step, team_feature_dim]
            self.event_seq_len = np.array(len(self.event_seq))
            ## player seq
            player_seq = pd.read_csv(path+'player_seq/'+self.player_seq_file).values[0:total_player_seq]
            self.player_seq_len = np.array(len(player_seq))
            # padding
            self.player_seq = np.pad(player_seq,((0,total_player_seq-len(player_seq)),(0,0)),'constant') 

class PlayerDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.data = []
        event_seq = 'event_seq/'
        team_seq = 'team_seq/'
        player_seq = 'player_seq/'

        dirs = os.listdir(path)
        dirs = [i for i in dirs if '.' not in i]
        dirs = [str(i) for i in range(len(dirs))]

        # Get all player list
        if (os.path.exists(processed_path+"all_player_data.csv")):
            all_player_df = pd.read_csv(processed_path+"all_player_data.csv")
        else:
            all_player_df = get_player_data()

        # Get playing time data of all players
        if (os.path.exists(processed_path+"total_play_time_data.csv")):
            total_play_time_data = pd.read_csv(processed_path+"total_play_time_data.csv")
            total_play_time_data.total_playing_time = total_play_time_data.total_playing_time.apply(
                lambda x:pd.Timedelta(x))
        else:
            total_play_time_data = get_play_time(all_player_df)
        suff_time_plyr = list(total_play_time_data[total_play_time_data.total_playing_time > pd.Timedelta(minutes=800)].player_id)

        all_player_df.join_date = all_player_df.join_date.apply(
            lambda x:pd.to_datetime(x, format="%Y-%m-%d"))
        join_date_plyr = list(
            all_player_df[all_player_df.join_date < pd.to_datetime('2017-01-01', format="%Y-%m-%d")].player_id)

        suff_plyr = [_ for _ in all_player_df if (_ in join_date_plyr) and (_ in suff_time_plyr)]

        for d in dirs:
            player_files = os.listdir(path+d+'/'+player_seq)
            player_files = [i for i in player_files if '.pseq' in i]
            for p in tqdm(range(len(player_files))):
                self.data.append(PlayerData(path+d+'/', player_files[p], all_player_df, suff_plyr))
        
    def __getitem__(self, index):
        #index %= 2000
        team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, player_seq, player_seq_len, stat_player, label_pos, label_player = \
            self.data[index].team_seq, self.data[index].team_seq_len, self.data[index].stat_team, self.data[index].event_seq, self.data[index].event_seq_len, self.data[index].stat_event, self.data[index].player_seq, self.data[index].player_seq_len, self.data[index].stat_player, self.data[index].label_pos, self.data[index].label_player
        #print(team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, player_seq, player_seq_len, stat_player, label_pos, label_player)
        
        return team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, player_seq, player_seq_len, stat_player, label_pos, label_player

    def __len__(self):
        return len(self.data)