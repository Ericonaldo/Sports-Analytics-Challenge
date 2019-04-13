# -*- coding:utf-8 -*-
from utils import *
from dataset import *
import torch
import torch.autograd as autograd 
import torch.nn as nn         
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


class TeamEventRNN(nn.Module):
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
        '''
        print('1',x_lens)
        x_lens = torch.tensor(x_lens)
        print('2',x_lens)
        '''

        x, x_lens, unsorted_idx =sort_sequences(x, x_lens)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        r_out, h_n = self.rnn(x.float(), h0)     # h0 = initial hidden state

        return h_n[0]

class LastEventRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LastEventRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timestep = 10                        # time step should be 10

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,         # rnn hidden unit
            num_layers=1,                         # number of rnn layer
            batch_first=True,                     # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x, x_lens, h0=None):
        # x shape (batch, event_time_step, event_input_size)
        # x_lens shape (batch, seq_len)
        # r_out shape (batch, event_time_step, event_output_size)
        # h_n shape (n_layers, batch, event_hidden_size)

        # x = pack_padded_sequence(x, x_lens, batch_first=True)
        r_out, h_n = self.rnn(x.float(), h0)   # h0 = initial hidden state
        
        return h_n[0]

class TeamEventNetwork(nn.Module):
    def __init__(self, team_input_size, team_hidden_size, event_input_size, event_hidden_size, team_stat_dim, event_stat_dim):
        super(TeamEventNetwork, self).__init__()
        self.team_input_size = team_input_size
        self.team_hidden_size = team_hidden_size
        self.event_input_size = event_input_size
        self.event_hidden_size = event_hidden_size
        self.team_stat_size = team_stat_dim
        self.event_stat_size = event_stat_dim

        self.team_rnn = TeamEventRNN(
            input_size=self.team_input_size,
            hidden_size=self.team_hidden_size,
        )
        self.event_rnn = LastEventRNN(
            input_size=self.event_input_size,
            hidden_size=self.event_hidden_size, # should be team_hidden_size + stat_team_size
        )
        # multi head for team and pos
        self.fc1 = torch.nn.Linear(self.event_hidden_size+self.event_stat_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)

        out_hidden_size = 64
        self.out_team = torch.nn.Linear(out_hidden_size, 1)
        self.out_xy = torch.nn.Linear(out_hidden_size, 2)

    def forward(self, team_seq, team_seq_len, stat_team, event_seq, event_seq_len, stat_event):
        # x_team shape (batch, team_time_step, team_input_size)
        # stat_team shape (batch, team_stat_size)
        # x_event shape (batch, event_time_step, event_input_size)
        # stat_event shape (batch, event_stat_size)
        # h_team shape (batch, team_hidden_size)
        # h_event shape (batch, event_hidden_size)

        h_team = self.team_rnn(team_seq, team_seq_len)
        h_team = torch.cat((h_team, stat_team.float()), 1)
        h_team = torch.unsqueeze(h_team, 0)
        h_event = self.event_rnn(event_seq, event_seq_len, h_team)

        h_output = torch.cat((h_event, stat_event.float()), 1)
        h_output = self.fc1(h_output)
        h_output = self.fc2(h_output)

        out_team = torch.sigmoid(self.out_team(h_output)) # predict the next team (batch, 1)
        out_xy = torch.sigmoid(self.out_team(h_output)) # predict the position of the ball (batch, 2)
        return [out_team, out_xy]


class BinaryRegressionLoss(torch.nn.Module):
    """
    Binary and Regression loss function.
    """

    def __init__(self):
        super(BinaryRegressionLoss, self).__init__()

    def forward(self, out_team, out_xy, label_team, label_xy):
        bin_logloss = nn.BCELoss()(out_team, label_team.float())
        reg_loss = torch.nn.MSELoss(reduction='none')(out_xy, label_xy.float())
        
        loss = bin_logloss + torch.sum(label_xy.float()*reg_loss)
        return loss

if __name__ == '__main__':   
    net = TeamEventNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        event_input_size=Config.event_feature_dim, event_hidden_size=Config.event_hidden_size,
        team_stat_dim=Config.team_stat_dim, event_stat_dim=Config.event_stat_dim)
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
            if i % 10 == 0 :
                print("Epoch {}\t Step {}\t Loss {}\t".format(epoch,i,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot('team_event_rnn', counter,loss_history)