from utils import *
from train_gbdt import *
from train_team_event_rnn import *
from evaluation_gbdt import *
from evaluation_rnn import *
import random

def Result(xml_1, choice='default'):

    if choice=='default':
        [pred_player, pred_team, _, _] = get_result_rnn(xml_1)
        [_, _, pred_x, pred_y] = get_result_gbdt(xml_1)
    elif choice=='rnn':
        [pred_player, pred_team, pred_x, pred_y] = get_result_rnn(xml_1)
    elif choice=='gbdt':
        [pred_player, pred_team, pred_x, pred_y] = get_result_gbdt(xml_1)
    elif choice=='rules':
        [pred_player, pred_team, pred_x, pred_y] = get_result_rules(xml_1)

    results = str(pred_player) + ',' + str(pred_team) + ',' + \
                            str(pred_x)+ ',' + str(pred_y) # the results
    results = str(results)

    with open('res_psgx.csv', 'w') as f:
        f.write(results)

def get_result_rnn(xml_1, epoch_team=360, epoch_player=500):
    ### team and (x,y)-------------------------
    net = TeamEventNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        event_input_size=Config.event_feature_dim, event_hidden_size=Config.event_hidden_size,
        team_stat_dim=Config.team_stat_dim, event_stat_dim=Config.event_stat_dim)

    if not os.path.exists(Config.model_path+str(epoch_team)+'_team_events_rnn.pkl'):
        print("Model doesn't exist!")
        exit(0)
    net.load_state_dict(torch.load(Config.model_path+str(epoch_team)+'_team_events_rnn.pkl'))

    [team0_seq, team0_seq_len, team1_seq, team1_seq_len] = construct_xml_to_team_seq(xml_1) # (1,T,D), (1), (1,T,D), (1)
    event_seq, event_seq_len = construct_xml_to_event_seq(xml_1) # (1,10,D), (1), (1,10,D), (1)
    stat_team0 = stat_team1 = stat_event = np.array([0])

    out_team0, out_xy0 = net(team0_seq, team0_seq_len, stat_team0, event_seq, event_seq_len, stat_event) # (1,1), (1,2)
    out_team1, out_xy1 = net(team1_seq, team1_seq_len, stat_team1, event_seq, event_seq_len, stat_event) # (1,1), (1,2)

    out_team = [0 if (out_team0[0,0] >= out_team1[0,0]) else 1][0]
    out_xy = [out_xy0 if (out_team0[0,0] >= out_team1[0,0]) else out_xy1][0][0].data.numpy() 

    out_xy *= 100
    out_xy = np.around(out_xy, 1)
    
    pred_team = out_team
    pred_x, pred_y = out_xy[0], out_xy[1]
    ## rules
    # get the feature dataframe
    df = construct_one_ball_team_df(xml_1)
    df = get_eva_fea(df)
    gc.collect()
    period_id = df.iloc[-1]['period_id']
    train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
    # get the last event
    df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]

    if (df_test_X.type_id == 50):
        pred_team = 1-df_test_X.team_id
        pred_x = 100-df_test_X.x
        pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 4) or (df_test_X.type_id == 5) or (df_test_X.type_id == 44):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 100-df_test_X.x
            pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 2) or (df_test_X.type_id == 51)or (df_test_X.type_id == 15):
        pred_team = 1-df_test_X.team_id
    elif (df_test_X.type_id == 49):
        pred_team = df_test_X.team_id
    elif (df_test_X.type_id == 25)or (df_test_X.type_id == 18):
        pred_team = df_test_X.team_id
        pred_x = 0
        pred_y = 0
    elif (df_test_X.type_id == 28) or (df_test_X.type_id == 68) or (df_test_X.type_id == 70):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 0
            pred_y = 0
    elif (df_test_X.type_id == 49):
        pred_team = 1-df_test_X.team_id
        pred_x = 0
        pred_y = 0
    
    ### player-------------------------
    # pred_player = get_random_player()
    net = PlayerClassifyNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        player_input_size=Config.player_feature_dim, player_hidden_size=Config.player_hidden_size,
        team_stat_dim=Config.team_stat_dim, player_stat_dim=Config.player_stat_dim)

    if not os.path.exists(Config.model_path+str(epoch_player)+'_player_rnn.pkl'):
        print("Model doesn't exist!")
        exit(0)
    net.load_state_dict(torch.load(Config.model_path+str(epoch_player)+'_player_rnn.pkl'))

    pdata = construct_xml_to_player_seq(xml_1)

    [team0_seq, team0_seq_len, team1_seq, team1_seq_len] = construct_xml_to_team_seq(xml_1) # (1,T,D), (1), (1,T,D), (1)
    event_seq, event_seq_len = construct_xml_to_event_seq(xml_1) # (1,10,D), (1), (1,10,D), (1)
    player_seq, player_seq_len, team_id = pdata
    stat_team0 = stat_team1 = stat_event = stat_player = np.array([0])

    if team_id == 0:
        out_pos, out_player = net(team0_seq, team0_seq_len, stat_team0, player_seq, player_seq_len, stat_player) # (1,4), (1,779)
    elif team_id == 1:
        out_pos, out_player = net(team1_seq, team1_seq_len, stat_team1, player_seq, player_seq_len, stat_player) # (1,4), (1,779)

    out_player = out_player.max(1, keepdim=True)[1]
    out_pos = out_pos.max(1, keepdim=True)[1]
    out_pos = int(out_pos[0][0])
    out_player = int(out_player[0][0])

    # Get all player list
    if (os.path.exists(processed_path+"all_player_data.csv")):
        all_player_df = pd.read_csv(processed_path+"all_player_data.csv")
    else:
        all_player_df = get_player_data()

    pred_player = all_player_df.iloc[out_player].player_id
    if pred_player[0] == 'p':
        pred_player = int(pred_player[1:])
    
    return [pred_player, pred_team, pred_x, pred_y]

def get_result_gbdt(xml_1):
    nbr_gate = 0.44
    team_gate = 0.5 # 0.45

    x_model = pickle.load(open(model_path+xname, 'rb'))
    y_model = pickle.load(open(model_path+yname, 'rb'))
    team_model = pickle.load(open(model_path+tname, 'rb'))

    # get the feature dataframe
    df = construct_one_ball_team_df(xml_1)
    df = get_eva_fea(df)
    gc.collect()
    period_id = df.iloc[-1]['period_id']
    train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
    # get the last event
    df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]
    
    pred_player = pred_x = pred_y = pred_team = np.nan
    
    # get the team prediction
    pred_team = team_model.predict(df_test_X)[0]
    if pred_team>=team_gate:
        pred_team = df_test_X.team_id
    else:
        pred_team = 1-df_test_X.team_id
           
    # predict_xy
    pred_x = x_model.predict(df_test_X)[0].round(1)
    pred_y = y_model.predict(df_test_X)[0].round(1)      

    # rules
    if (df_test_X.type_id == 50):
        pred_team = 1-df_test_X.team_id
        pred_x = 100-df_test_X.x
        pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 4) or (df_test_X.type_id == 5) or (df_test_X.type_id == 44):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 100-df_test_X.x
            pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 2) or (df_test_X.type_id == 51)or (df_test_X.type_id == 15):
        pred_team = 1-df_test_X.team_id
    elif (df_test_X.type_id == 49):
        pred_team = df_test_X.team_id
    elif (df_test_X.type_id == 25)or (df_test_X.type_id == 18):
        pred_team = df_test_X.team_id
        pred_x = 0
        pred_y = 0
    elif (df_test_X.type_id == 28) or (df_test_X.type_id == 68) or (df_test_X.type_id == 70):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 0
            pred_y = 0
    elif (df_test_X.type_id == 49):
        pred_team = 1-df_test_X.team_id
        pred_x = 0
        pred_y = 0

    pred_player = get_random_player()
    return [pred_player, pred_team, pred_x, pred_y]

def get_result_rules(xml_1):
    df = construct_one_ball_team_df(game_xml)
    df = get_eva_fea(df)
    gc.collect()
    
    period_id = df.iloc[-1]['period_id']
    train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
    
    df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]
    
    pred_player = pred_x = pred_y = pred_team = np.nan
    
    ## test teams
    pred_team = rnd.randint(0,1)
    
    ## test xy
    pred_x = pred_y = 0

    # rules
    if (df_test_X.type_id == 50):
        pred_team = 1-df_test_X.team_id
        pred_x = 100-df_test_X.x
        pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 4) or (df_test_X.type_id == 5) or (df_test_X.type_id == 44):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 100-df_test_X.x
            pred_y = 100-df_test_X.y
    elif (df_test_X.type_id == 2) or (df_test_X.type_id == 51)or (df_test_X.type_id == 15):
        pred_team = 1-df_test_X.team_id
    elif (df_test_X.type_id == 49):
        pred_team = df_test_X.team_id
    elif (df_test_X.type_id == 25)or (df_test_X.type_id == 18):
        pred_team = df_test_X.team_id
        pred_x = 0
        pred_y = 0
    elif (df_test_X.type_id == 28) or (df_test_X.type_id == 68) or (df_test_X.type_id == 70):
        if df_test_X.type_id != df_test_X.last_type_id:
            pred_team = 1-df_test_X.team_id
            pred_x = 0
            pred_y = 0
    elif (df_test_X.type_id == 49):
        pred_team = 1-df_test_X.team_id
        pred_x = 0
        pred_y = 0

    pred_player = get_random_player()
    return [pred_player, pred_team, pred_x, pred_y]

def get_random_player():
    # Get all player list and changed player list
    if (os.path.exists(processed_path+"all_player_data.csv")):
        all_player_df = pd.read_csv(processed_path+"all_player_data.csv")
    else:
        all_player_df = get_player_data()
    all_player_df.join_date = all_player_df.join_date.apply(
            lambda x:pd.to_datetime(x, format="%Y-%m-%d"))
    join_date_plyr = list(
        all_player_df[all_player_df.join_date < pd.to_datetime('2017-01-01', format="%Y-%m-%d")].player_id)
    
    # Get playing time data of all players
    if (os.path.exists(processed_path+"total_play_time_data.csv")):
        total_play_time_data = pd.read_csv(processed_path+"total_play_time_data.csv")
        total_play_time_data.total_playing_time = total_play_time_data.total_playing_time.apply(
            lambda x:pd.Timedelta(x))
    else:
        total_play_time_data = get_play_time(all_player_df)
    suff_time_plyr = list(total_play_time_data[total_play_time_data.total_playing_time > pd.Timedelta(minutes=800)].player_id)

    

    players = [_ for _ in suff_time_plyr if _ in join_date_plyr]

    rnd_idx = random.randint(0,len(players))
    plyr_id = players[rnd_idx]
    if 'p' == plyr_id[0]:
        plyr_id = plyr_id[1:]

    return plyr_id