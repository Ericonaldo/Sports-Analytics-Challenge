from utils import *
import argparse
from train_team_event_rnn import *
from train_player_rnn import *
from evaluation_gbdt import get_eva_fea

def construct_xml_to_team_seq(choice_xml):
    """
    Construct the team event sequence from a xml file.
    args:
        path: the given game's xml file.
    return:
        The team event sequence from a given xml file.
        
    """
    
    team0_df = pd.DataFrame({"min":[],
                            "sec":[],  
                            "type_id":[],
                            "q_num":[],
                            "keypass":[],
                            "nokeypass":[],
                            "assist":[],
                            "noassist":[],
                            })
    team1_df = pd.DataFrame({"min":[],
                            "sec":[],  
                            "type_id":[],
                            "q_num":[],
                            "keypass":[],
                            "nokeypass":[],
                            "assist":[],
                            "noassist":[],
                            })    
            
    
    events = choice_xml.xpath('//Event')[0:-10]
    team_0_events = [i for i in events if i.attrib['team_id']=='0']
    team_1_events = [i for i in events if i.attrib['team_id']=='1']
    
    for i in team_0_events:
        mins = int(i.attrib['min'])
        secs = int(i.attrib['sec'])
        type_id = int(i.attrib['type_id'])
        keypass = int('keypass' in i.attrib)
        assist = int('assist' in i.attrib)
        q_num = len(i.xpath('Q'))/200
        temp = pd.DataFrame({"min":[mins],
                            "sec":[secs],  
                            "type_id":[type_id],
                            "q_num":[q_num],
                            "keypass":[keypass],
                            "nokeypass":[1-keypass],
                            "assist":[assist],
                            "noassist":[1-assist],
                            })
        team0_df = pd.concat([team0_df,temp])
        
    for i in team_1_events:
        mins = int(i.attrib['min'])
        secs = int(i.attrib['sec'])
        keypass = int('keypass' in i.attrib)
        assist = int('assist' in i.attrib)
        q_num = len(i.xpath('Q'))/200
        temp = pd.DataFrame({"min":[mins],
                            "sec":[secs],  
                            "type_id":[type_id],
                            "q_num":[q_num],
                            "keypass":[keypass],
                            "nokeypass":[1-keypass],
                            "assist":[assist],
                            "noassist":[1-assist],
                            })
        team1_df = pd.concat([team1_df,temp])
        
    team0_df = get_time_fea(team0_df)
    team1_df = get_time_fea(team1_df)
    
    team0_df = get_type_fea(team0_df)
    team1_df = get_type_fea(team1_df)
    
    if 'type_id' in list(team0_df.columns):
        team1_df.drop(['type_id'], axis=1, inplace=True)
        team0_df.drop(['type_id'], axis=1, inplace=True)

    gc.collect()

    team0_seq_len = len(team0_df)
    team1_seq_len = len(team1_df)
        
    return [torch.unsqueeze(torch.tensor(team0_df.values), 0), torch.tensor([team0_seq_len]), torch.unsqueeze(torch.tensor(team1_df.values), 0), torch.tensor([team1_seq_len])]
        

def construct_xml_to_event_seq(choice_xml):
    """
    Construct the last 10 event sequence from a xml file.
    args:
        path: the given game's xml file.
    return:
        The last 10 event sequence from a given xml file.
    """
    
    event_df = pd.DataFrame({"min":[],
                            "sec":[],  
                            "type_id":[],
                            "q_num":[],
                            "x":[],
                            "y":[],
                            "keypass":[],
                            "nokeypass":[],
                            "assist":[],
                            "noassist":[],
                            })
    events = choice_xml.xpath('//Event')[-10:]
    
    for i in events:
        mins = int(i.attrib['min'])
        secs = int(i.attrib['sec'])
        type_id = int(i.attrib['type_id'])
        keypass = int('keypass' in i.attrib)
        assist = int('assist' in i.attrib)
        q_num = len(i.xpath('Q'))/200
        x = float(i.attrib['x'])/100
        y = float(i.attrib['y'])/100
        temp = pd.DataFrame({"min":[mins],
                            "sec":[secs],  
                            "type_id":[type_id],
                            "q_num":[q_num],
                            "x":[x],
                            "y":[y],
                            "keypass":[keypass],
                            "nokeypass":[1-keypass],
                            "assist":[assist],
                            "noassist":[1-assist],
                            })
        event_df = pd.concat([event_df, temp])
    
    event_df = get_time_fea(event_df)       
    event_df = get_type_fea(event_df)
    event_df = get_space_fea(event_df)
    gc.collect()

    if 'type_id' in list(event_df.columns):
        event_df.drop(['type_id'], axis=1, inplace=True)
    
    return torch.unsqueeze(torch.tensor(event_df.values), 0), torch.tensor([10])

def construct_xml_to_player_seq(choice_xml):
    """
    Construct the player sequence from a xml file.
    args:
        path: the given game's xml file.
    return:
        The player sequence from a given xml file.
        
    """
    
    events = choice_xml.xpath('//Event')[0:-10]
    
    player_df = pd.DataFrame({"min":[],
                        "sec":[],  
                        "type_id":[],
                        "q_num":[],
                        "keypass":[],
                        "nokeypass":[],
                        "assist":[],
                        "noassist":[],
                        })
    
    p_events = [_ for _ in events if 'player_id' in _.attrib]
    
    p_events = [_ for _ in p_events if (int(_.attrib['player_id']) == 1)]
    
    if len(p_events)==0:
        print('len(players)=0!')
        return [0]  

    team_id = p_events[0].attrib['team_id']
    for i in p_events:
        mins = int(i.attrib['min'])
        secs = int(i.attrib['sec'])
        type_id = int(i.attrib['type_id'])
        keypass = int('keypass' in i.attrib)
        assist = int('assist' in i.attrib)
        q_num = len(i.xpath('Q'))/200
        temp = pd.DataFrame({"min":[mins],
                            "sec":[secs],  
                            "type_id":[type_id],
                            "q_num":[q_num],
                            "keypass":[keypass],
                            "nokeypass":[1-keypass],
                            "assist":[assist],
                            "noassist":[1-assist],
                            })
        player_df = pd.concat([player_df, temp])

    player_df = get_time_fea(player_df)       
    player_df = get_type_fea(player_df)
    
    if 'type_id' in list(player_df.columns):
            player_df.drop(['type_id'], axis=1, inplace=True)
        
    gc.collect()
    #player_df['player_id'] = p
    player_seq_len = len(player_df)

    return [torch.unsqueeze(torch.tensor(player_df.values), 0), torch.tensor([player_seq_len]), int(team_id)]
        

def evaluate_xyt_rnn(valid_dir=valid_path, epoch = 500):
    """
    Evaluate the given model.
    args:
        valid_dir: the dir path of validation sets.
        models: a list of [player_model, next_ball_related_model, x_model, y_model, team_model]
    return:
        test_df: the test data whose type is pd.DataFrame.
        
    """
    net = TeamEventNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        event_input_size=Config.event_feature_dim, event_hidden_size=Config.event_hidden_size,
        team_stat_dim=Config.team_stat_dim, event_stat_dim=Config.event_stat_dim)

    if not os.path.exists(Config.model_path+str(epoch)+'_team_events_rnn.pkl'):
        print("Model doesn't exist!")
        exit(0)
    net.load_state_dict(torch.load(Config.model_path+str(epoch)+'_team_events_rnn.pkl'))

    files= os.listdir(valid_dir)
    xml_files = [i for i in files if i[-3:]=='xml']
    csv_files = [i for i in files if i[-3:]=='csv']
    xml_files.sort()
    csv_files.sort()
    file_num = len(xml_files)

    score_player = loss_xy = score_team = count_num = 0
    for i in tqdm(range(file_num)):
        ground_truth = pd.read_csv(valid_dir+csv_files[i], header=None)
        if ground_truth.iloc[0,2] == ground_truth.iloc[0,3] == 0:
            continue

        choice_xml = lxml.etree.parse(valid_dir+xml_files[i])

        [team0_seq, team0_seq_len, team1_seq, team1_seq_len] = construct_xml_to_team_seq(choice_xml) # (1,T,D), (1), (1,T,D), (1)
        event_seq, event_seq_len = construct_xml_to_event_seq(choice_xml) # (1,10,D), (1), (1,10,D), (1)
        stat_team0 = stat_team1 = stat_event = np.array([0])

        out_team0, out_xy0 = net(team0_seq, team0_seq_len, stat_team0, event_seq, event_seq_len, stat_event) # (1,1), (1,2)
        out_team1, out_xy1 = net(team1_seq, team1_seq_len, stat_team1, event_seq, event_seq_len, stat_event) # (1,1), (1,2)

        out_team = [0 if (out_team0[0,0] >= out_team1[0,0]) else 1][0]
        out_xy = [out_xy0 if (out_team0[0,0] >= out_team1[0,0]) else out_xy1][0][0].data.numpy() 

        out_xy *= 100
        out_xy = np.around(out_xy, 1)

        pred_x = out_xy[0]
        pred_y = out_xy[1]
        pred_team = out_team

        # get the feature dataframe
        df = construct_one_ball_team_df(choice_xml)
        df = get_eva_fea(df)
        gc.collect()
        period_id = df.iloc[-1]['period_id']
        train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
        # get the last event
        df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]

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

        # compute result 
        if pred_team == ground_truth.iloc[0,1]:
            score_team +=1
        loss_xy += (pred_x - ground_truth.iloc[0,2])**2+\
                   (pred_y - ground_truth.iloc[0,3])**2
            
        print('-------------------------------------')
        print('label team={}, x={}, y={}'.format(
            ground_truth.iloc[0,1],ground_truth.iloc[0,2],ground_truth.iloc[0,3]))
        print('prdct team={}, x={}, y={}'.format(
            pred_team, pred_x, pred_y))
        print('-------------------------------------')
        count_num += 1

    print('\n ave scores/loss score_team={}, loss_xy={}'.format(
        float(score_team)/count_num, loss_xy/count_num))
    print('count_num: ', count_num)

def evaluate_p_rnn(all_player_df, valid_dir=valid_path, epoch = 500):
    """
    Evaluate the given model.
    args:
        valid_dir: the dir path of validation sets.
        models: a list of [player_model, next_ball_related_model, x_model, y_model, team_model]
    return:
        test_df: the test data whose type is pd.DataFrame.
        
    """
    net = PlayerClassifyNetwork(team_input_size=Config.team_feature_dim, team_hidden_size=Config.team_hidden_size, 
        player_input_size=Config.player_feature_dim, player_hidden_size=Config.player_hidden_size,
        team_stat_dim=Config.team_stat_dim, player_stat_dim=Config.player_stat_dim)

    if not os.path.exists(Config.model_path+str(epoch)+'_player_rnn.pkl'):
        print("Model doesn't exist!")
        exit(0)
    net.load_state_dict(torch.load(Config.model_path+str(epoch)+'_player_rnn.pkl'))

    files= os.listdir(valid_dir)
    xml_files = [i for i in files if i[-3:]=='xml']
    csv_files = [i for i in files if i[-3:]=='csv']
    xml_files.sort()
    csv_files.sort()
    file_num = len(xml_files)

    score_player = score_pos = count_num = 0
    for i in tqdm(range(file_num)):
        ground_truth = pd.read_csv(valid_dir+csv_files[i], header=None)
        if ground_truth.iloc[0,2] == ground_truth.iloc[0,3] == 0:
            continue

        choice_xml = lxml.etree.parse(valid_dir+xml_files[i])

        pdata = construct_xml_to_player_seq(choice_xml)
        if (len(pdata) == 1) and (pdata[0] == 0):
            continue

        [team0_seq, team0_seq_len, team1_seq, team1_seq_len] = construct_xml_to_team_seq(choice_xml) # (1,T,D), (1), (1,T,D), (1)
        event_seq, event_seq_len = construct_xml_to_event_seq(choice_xml) # (1,10,D), (1), (1,10,D), (1)
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
        pred_player = all_player_df.iloc[out_player].player_id
        if pred_player[0] == 'p':
            pred_player = int(pred_player[1:])
        
        label_player = ground_truth.iloc[0,0]
        # print(all_player_df)
        # print(label_player)
        label_pos = all_player_df[all_player_df.player_id == ('p'+str(label_player))]['position'].values[0]
        pred_pos = Config.pos_name[out_pos]

        # compute result 
        if pred_player == int(label_player):
            score_player +=1
        if pred_pos == label_pos:
            score_pos +=1

        print('-------------------------------------')
        print('label position={}, player={}'.format(label_pos, label_player))
        print('prdct position={}, player={}'.format(pred_pos, pred_player))
        print('-------------------------------------')
        count_num += 1

    print('\n ave scores/loss score_position={}, score_player={}'.format(float(score_pos)/count_num, float(score_player)/count_num))
    print('count_num: ', count_num)
    
     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--valid', type=int, default=0)
    parser.add_argument('--test_p', action='store_true')
    parser.add_argument('--test_xyt', action='store_true')
    parser.add_argument('--xyt_epoch', type=int, default=360)
    parser.add_argument('--p_epoch', type=int, default=500)
    
    args = parser.parse_args()
    if args.valid != 0:
        construct_val_sets(val_num=args.valid)
    if args.test_xyt:
        print("Evaluating xyt...")
        evaluate_xyt_rnn(epoch = args.xyt_epoch)

    if args.test_p:
        # Get all player list
        if (os.path.exists(processed_path+"all_player_data.csv")):
            all_player_df = pd.read_csv(processed_path+"all_player_data.csv")
        else:
            all_player_df = get_player_data()
        print("Evaluating players...")
        evaluate_p_rnn(epoch = args.p_epoch, all_player_df = all_player_df)