# -*- coding:utf-8 -*-
from utils import *
from dataset import *
from train_team_event_rnn import TeamEventRNN, LastEventRNN
import torch
import torch.autograd as autograd 
import torch.nn as nn         
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as Data

class PlayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TeamEventRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,         # rnn hidden unit
            num_layers=1,                         # number of rnn layer
            batch_first=True,                     # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x, x_lens, h0=None):
        # x shape (batch, team_time_step, team_input_size)
        # x_lens shape (batch, seq_len)
        # r_out shape (batch, team_time_step, team_output_size)
        # h_0, h_n shape (n_layers, batch, team_hidden_size)

        x, x_lens, unsorted_idx =sort_sequences(x, x_lens)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        r_out, h_n = self.rnn(x, h0)     # h0 = initial hidden state

        return h_n


class PlayerClassifyNetwork(nn.Module):
    def __init__(self, team_input_size, team_hidden_size, event_input_size, event_hidden_size, player_input_size, player_hidden_size):
        super(TeamEventNetwork, self).__init__()
        self.team_input_size = team_input_size
        self.team_hidden_size = team_hidden_size
        self.event_input_size = event_input_size
        self.event_hidden_size = event_hidden_size
        self.player_input_size = event_input_size
        self.player_hidden_size = event_hidden_size

        self.team_rnn = TeamEventRNN(
            input_size=self.team_input_size,
            hidden_size=self.team_hidden_size,
        )
        self.event_rnn = LastEventRNN(
            input_size=self.event_input_size,
            hidden_size=self.event_hidden_size, # should be team_hidden_size + stat_team_size
        )

        self.player_rnn = PlayerRNN(
            input_size=self.player_input_size,
            hidden_size=self.player_hidden_size, # should be team_hidden_size + stat_team_size
        )
        # multi head for team and pos
        self.out_pos = torch.nn.Linear(self.player_hidden_size+self.stat_player_size, Config.pos_class)
        self.out_player = torch.nn.Linear(self.player_hidden_size+self.stat_player_size, Config.max_class_num)

    def forward(self, team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event):
        # x_team shape (batch, team_time_step, team_input_size)
        # stat_team shape (batch, team_stat_size)
        # x_event shape (batch, event_time_step, event_input_size)
        # stat_event shape (batch, event_stat_size)
        # h_team shape (batch, team_hidden_size)
        # h_event shape (batch, event_hidden_size)

        h_team = self.team_rnn(x_team, team_seq_len)
        h_event = self.event_rnn(x_event, event_seq_len, h_team)

        out_pos = torch.sigmoid(self.out_team(h_event)) # predict the next team (batch, 1)
        out_player = torch.sigmoid(self.out_team(h_event)) # predict the position of the ball (batch, 2)
        return [out_team, out_xy]


class MultiCrossEntropyLoss(torch.nn.Module):
    """
    Multiple CrossEntropyLoss loss function.
    """

    def __init__(self):
        super(BinaryRegressionLoss, self).__init__()

    def forward(self, out_team, out_xy, label_team, label_xy):
        bin_logloss = nn.BCELoss(out_team, label_team)
        reg_loss = nn.MSELoss(out_xy, label_xy)

        
        loss = bin_logloss + label_xy*reg_loss
        return loss

if __name__ == '__main__':   
    net = TeamEventNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        event_input_size=Config.event_feature_dim, event_hidden_size=Config.event_hidden_size)
    criterion = BinaryRegressionLoss()
    optimizer = optim.Adam(net.parameters(), lr = Config.lr)

    print('Loading data...')
    if os.path.exists(Config.processed_path+'TeamEventDataset.pkl'):
        team_event_dataset = pickle.load(open(Config.processed_path+'TeamEventDataset.pkl', 'rb'))
    else:
        team_event_dataset = TeamEventDataset(train_path)
        pickle.dump(team_event_dataset, open(Config.processed_path+'TeamEventDataset.pkl', 'wb'))
    print('Data Loaded!')
    train_dataloader = Data.DataLoader(team_event_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.batch_size)
    
    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event, label_team, label_xy = data
            optimizer.zero_grad()
            out_team, out_xy = net(team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event)
            loss_contrastive = criterion(out_team, out_xy, label_team, label_xy)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history)