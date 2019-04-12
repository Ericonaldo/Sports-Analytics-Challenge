# -*- coding:utf-8 -*-
from utils import *
import torch
import torch.autograd as autograd 
import torch.nn as nn         
import torch.nn.functional as F 
import torch.optim as optim

# Hyper Parameters and other config
class Config():
    data_path = "../data/XPSG - available resources/"
    train_path = "../data/train/" # path of processed training data
    valid_path = "../data/valid/" # path of processed validation data
    processed_path = "../data/processed/"
    model_path = '../models/'
    batch_size = 64
    number_epochs = 500
    lr = 0.01  

class TeamRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TeamRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,         # rnn hidden unit
            num_layers=1,                         # number of rnn layer
            batch_first=True,                     # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        # x shape (batch, team_time_step, team_input_size)
        # r_out shape (batch, team_time_step, team_output_size)
        # h_n shape (n_layers, batch, team_hidden_size)
        # h_c shape (n_layers, batch, team_hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)     # None represents zero initial hidden state

        return h_n

class LastEventRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LastEventRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timestep = 10                        # time step should be 10

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,         # rnn hidden unit
            num_layers=1,                         # number of rnn layer
            batch_first=True,                     # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        # x shape (batch, event_time_step, event_input_size)
        # r_out shape (batch, event_time_step, event_output_size)
        # h_n shape (n_layers, batch, event_hidden_size)
        # h_c shape (n_layers, batch, event_hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        
        return h_n

class TeamEventNetwork(nn.Module):
    def __init__(self, team_input_size, team_hidden_size, event_input_size, event_hidden_size):
        super(TeamEventNetwork, self).__init__()
        self.team_input_size = team_input_size
        self.team_hidden_size = team_hidden_size
        self.event_input_size = event_input_size
        self.event_hidden_size = event_hidden_size

        self.team_rnn = TeamRNN(
            input_size=self.team_input_size,
            hidden_size=self.team_hidden_size,
        )
        self.event_rnn = EventRNN(
            input_size=self.event_input_size,
            hidden_size=self.event_hidden_size, # should be team_hidden_size + stat_team_size
        )
        # multi head for team and pos
        self.out_team = self.torch.nn.Linear(self.event_hidden_size, 1)
        self.out_xy = self.torch.nn.Linear(self.event_hidden_size, 2)

    def forward(self, x_team, stat_team, x_event, stat_event):
        # x_team shape (batch, team_time_step, team_input_size)
        # stat_team shape (batch, team_stat_size)
        # x_event shape (batch, event_time_step, event_input_size)
        # stat_event shape (batch, event_stat_size)
        # h_team shape (batch, team_hidden_size)
        # h_event shape (batch, event_hidden_size)

        h_team = self.team_rnn(x_team, None)   # None represents zero initial hidden state
        h_event = self.event_rnn(x_event, torch.cat[h_team, stat_team]) 

        out_team = F.sigmoid(self.out_team(torch.cat[h_event, event_stat])) # predict the next team (batch, 1)
        out_xy = F.sigmoid(self.out_team(torch.cat[h_event, event_stat])) # predict the position of the ball (batch, 2)
        return [out_team, out_xy]


class BinaryRegressionLoss(torch.nn.Module):
    """
    Binary and Regression loss function.
    """

    def __init__(self):
        super(BinaryRegressionLoss, self).__init__()

    def forward(self, out_team, out_xy, label_team, label_xy):
        bin_logloss = -nn.BCELoss(out_team, label_xy)
        reg_loss = torch.mse_loss(out_xy, label_xy)
        
        loss = bin_logloss + label_xy*reg_loss
        return loss

if __name__ == '__main__':   
    net = TeamEventNetwork()
    criterion = BinaryRegressionLoss()
    optimizer = optim.Adam(net.parameters(), lr = config.lr)

    team_dataset = TeamEventDataset()
    train_dataloader = DataLoader(team_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.batch_size)
    
    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            x_team, stat_team, x_event, stat_event, label_team, label_xy = data
            optimizer.zero_grad()
            out_team, out_xy = net(x_team, stat_team, x_event, stat_event)
            loss_contrastive = criterion(out_team, out_xy, label_team, label_xy)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history)