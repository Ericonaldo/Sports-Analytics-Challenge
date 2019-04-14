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
import argparse

class PlayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PlayerRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, 16)
        self.rnn = nn.GRU(
            input_size=16,
            hidden_size=self.hidden_size,         # rnn hidden unit
            num_layers=1,                         # number of rnn layer
            batch_first=True,                     # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x, x_lens, h0=None):
        # x shape (batch, team_time_step, team_input_size)
        # x_lens shape (batch, seq_len)
        # r_out shape (batch, team_time_step, team_output_size)
        # h_0, h_n shape (n_layers, batch, team_hidden_size)
        
        x = self.fc1(x.float())
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        r_out, h_n = self.rnn(x.float(), h0)     # h0 = initial hidden state

        return h_n[0]


class PlayerClassifyNetwork(nn.Module):
    def __init__(self, team_input_size, team_hidden_size, player_input_size, player_hidden_size, team_stat_dim, player_stat_dim):
        super(PlayerClassifyNetwork, self).__init__()
        self.team_input_size = team_input_size
        self.team_hidden_size = team_hidden_size
        self.player_input_size = player_input_size
        self.player_hidden_size = player_hidden_size
        self.team_stat_size = team_stat_dim
        self.player_stat_size = player_stat_dim

        self.team_rnn = TeamEventRNN(
            input_size=self.team_input_size,
            hidden_size=self.team_hidden_size,
        )

        self.player_rnn = PlayerRNN(
            input_size=self.player_input_size,
            hidden_size=self.player_hidden_size,
        )

        out_hidden_size = 16

        self.fc1 = torch.nn.Linear(self.team_hidden_size+self.player_hidden_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, out_hidden_size)

        out_hidden_size = 32
        self.out_pos = torch.nn.Linear(out_hidden_size, Config.pos_class)
        self.out_player = torch.nn.Linear(out_hidden_size, Config.sum_class_num)

    def forward(self, team_seq, team_seq_len, stat_team, player_seq, player_seq_len, stat_player):
        # team_seq shape (batch, team_time_step, team_input_size)
        # stat_team shape (batch, team_stat_size)
        # player_seq shape (batch, player_time_step, player_input_size)
        # stat_player shape (batch, player_stat_size)
        # h_team shape (batch, team_hidden_size)
        # h_player shape (batch, player_hidden_size)

        h_team = self.team_rnn(team_seq, team_seq_len)
        # h_team = torch.cat((h_team, stat_team.float()), 1)
        h_player = self.player_rnn(player_seq, player_seq_len)

        # h_output = torch.cat((h_player, player_event.float()), 1)
        h_output = torch.cat((h_team, h_player), 1)
        h_output = torch.relu(self.fc1(h_output))
        h_output = torch.relu(self.fc2(h_output))
        h_output = torch.relu(self.fc3(h_output))
        h_output = torch.relu(self.fc4(h_output))
        # print(h_output)

        out_pos = self.out_team(h_event) # predict the pos class
        out_player = self.out_team(h_event) # predict the player
        return [out_team, out_xy]


class MultiCrossEntropyLoss(torch.nn.Module):
    """
    Multiple CrossEntropyLoss loss function.
    """

    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, out_pos, out_player, label_pos, label_player):
        # out_pos shape (batch, team_time_step, team_input_size)
        # out_player shape (batch, team_stat_size)
        # label_pos shape (batch, event_time_step, event_input_size)
        # label_player shape (batch, event_stat_size)
        pos_loss = nn.NLLLoss()(out_pos, label_pos)
        player_loss = nn.NLLLoss(out_player, label_player)
         
        loss = Config.weight_pos_loss * pos_loss + Config.weight_player_loss * player_loss
        return loss

def adjust_lr(optimizer, interation):
    if (interation+1) % Config.player_decay_iter == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= Config.player_decay_value
            print('current lr', param_group['lr'])

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--load', type=int, default=0)

    args = parser.parse_args()

    net = PlayerClassifyNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        player_input_size=Config.player_feature_dim, player_hidden_size=Config.player_hidden_size,
        team_stat_dim=Config.team_stat_dim, player_stat_dim=Config.player_stat_dim)

    if args.load>0:
        if not os.path.exists(Config.model_path+str(args.load)+'_player_rnn.pkl'):
            print("Model doesn't exist!")
            exit(0)
        net.load_state_dict(torch.load(Config.model_path+str(args.load)+'_player_rnn.pkl'))
    
    criterion = MultiCrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = Config.player_lr)

    print('Loading data...')
    if os.path.exists(Config.processed_path+'PlayerDataset.pkl'):
        player_dataset = pickle.load(open(Config.processed_path+'PlayerDataset.pkl', 'rb'))
    else:
        player_dataset = PlayerDataset(train_path)
        pickle.dump(player_dataset, open(Config.processed_path+'PlayerDataset.pkl', 'wb'))
    print('Data Loaded!')
    train_dataloader = Data.DataLoader(player_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.batch_size)
    
    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.player_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            team_seq, team_seq_len, stat_team, event_seq, event_seq_len, player_seq, player_seq_len, stat_player, label_pos, label_player = data
            optimizer.zero_grad()
            out_pos, out_player = net(team_seq, team_seq_len, stat_team, player_seq, player_seq_len, stat_player)
            myloss = criterion(out_pos, out_player, label_pos, label_player)
            myloss.backward()
            optimizer.step()
            mov_ave_loss = [(0.8*mov_ave_loss + 0.2*myloss.item()) if i>0 else myloss.item()][0]
            iteration_number += 1
            adjust_lr(optimizer, iteration_number)
            if i % 10 == 0 :
                print("Epoch {}\t Step {}\t Loss {}\t".format(epoch, i, mov_ave_loss))
                counter.append(iteration_number)
                loss_history.append(mov_ave_loss)
        
        if (epoch+1) % 50 == 0:
            np.savetxt(Config.model_path+'player_counter.txt', np.array(counter))
            np.savetxt(Config.model_path+'player_loss_history.txt', np.array(loss_history))
            torch.save(net.state_dict(), Config.model_path+str(epoch+1)+'_player_rnn.pkl')
            
    show_plot('player_rnn', counter,loss_history)