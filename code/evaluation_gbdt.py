# -*- coding:utf-8 -*-
from utils import *
import argparse
nbr_gate = 0.44
team_gate = 0.5 # 0.45

def get_eva_fea(train_df):
    """
    Get the feature in order to evaluate the model.
    args:
        train_df: the training df to be constructed features.
    return:
        train_df: the training set with constructed features.
        
    """
    train_df.insert(0,'game_id',np.ones([len(train_df)]))
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
    
    return train_df


def evaluation_with_next_ball(save_path=train_path, valid_dir=valid_path, models=[None, None, None, None, None]):
    """
    Evaluate the given model.
    args:
        valid_dir: the dir path of validation sets.
        models: a list of [player_model, next_ball_related_model, x_model, y_model, team_model]
    return:
        test_df: the test data whose type is pd.DataFrame.
        
    """
    test_df = pd.DataFrame({"pre_next_ball_related":[],
                            "pre_x":[],
                            "pre_y":[],
                            "pre_team":[],
                            "next_ball_related":[],
                            "x":[],
                            "y":[],
                            "team_id":[],})
    [player_model, next_ball_related_model, x_model, y_model, team_model] = models
    
    files= os.listdir(valid_dir)
    xml_files = [i for i in files if i[-3:]=='xml']
    csv_files = [i for i in files if i[-3:]=='csv']
    file_num = len(xml_files)
    
    score_player = loss_xy = score_team = 0
    
    for i in tqdm(range(file_num)):
        ground_truth = pd.read_csv(valid_dir+csv_files[i], header=None)
        
        game_xml = lxml.etree.parse(valid_dir+xml_files[i])
        df = construct_one_ball_team_df(game_xml)
        df = get_eva_fea(df)
        gc.collect()
        
        period_id = df.iloc[-1]['period_id']
        train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
        
        df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]
        
        pred_player = pred_x = pred_y = pred_team = None
        
        if (next_ball_related_model is not None):
            if team_model is not None:
                print("Testing team model...")
                # predict next ball related
                pred_next_ball_related = next_ball_related_model.predict(df_test_X)[0]
                df_test_X['next_ball_related'] = [1 if pred_next_ball_related>=nbr_gate else 0][0]
                pred_team = team_model.predict(df_test_X)[0]
                team = [1 if pred_team>=team_gate else 0][0]
                
                # team = rnd.randint(0,1)

                # rules
                if df_test_X.type_id == map_type_id(25):
                    team = df_test_X.team_id
                elif (df_test_X.type_id == 16) or (df_test_X.type_id == 19):
                    team = df_test_X.team_id
                elif df_test_X.type_id == map_type_id(18):
                    team = df_test_X.team_id
                elif df_test_X.type_id == map_type_id(15):
                    team = 1-df_test_X.team_id
                # compute result
                if team == ground_truth.iloc[0,1]:
                    score_team +=1

            if (x_model is not None) and (y_model is not None):
                print("Testing x,y model...")
                # predict next ball related
                pred_next_ball_related = next_ball_related_model.predict(df_test_X)[0]
                next_ball_related = [1 if pred_next_ball_related>=nbr_gate else 0][0]
                if next_ball_related==1:
                    # predict_xy
                    pred_x = x_model.predict(df_test_X)[0].round(1)
                    pred_y = y_model.predict(df_test_X)[0].round(1)
                    if int(period_id)^int(team):
                        pred_x = 100-pred_x
                        pred_y = 100-pred_y
                else:
                    pred_x = pred_y = 0
                # rules
                if df_test_X.type_id == map_type_id(15):
                        pred_x = df_test_X.x
                        pred_y = df_test_X.y
                        if (int(period_id)^int(team)) and not(int(period_id)^int(df_test_X.team_id)):
                            pred_x = 100-pred_x
                            pred_y = 100-pred_y
                if (df_test_X.type_id == 16) and (df_test_X.ball_related == 0):
                    pred_x = pred_y = 0
                if (df_test_X.type_id == 34) and (df_test_X.ball_related == 1):
                    pred_x = pred_y = 0
                if df_test_X.type_id == map_type_id(18):
                    pred_x = pred_y = 0
                # compute result 
                loss_xy += (pred_x - ground_truth.iloc[0,2])*(pred_x - ground_truth.iloc[0,2])+\
                                    (pred_y - ground_truth.iloc[0,3])*(pred_y - ground_truth.iloc[0,3])
        else:
            print("next_ball_related_model is None!")

        if player_model is not None:
            pred_player = player_model.predict(df_test_X)[0]
            if pred_player == ground_truth.iloc[0,0]:
                score_player +=1
            
        print('ground truth player={}, team={}, x={}, y={}'.format(
            ground_truth.iloc[0,0],ground_truth.iloc[0,1],ground_truth.iloc[0,2],ground_truth.iloc[0,3]))
        print('predicted results player={}, team={}, x={}, y={}'.format(
            pred_player, team, pred_x, pred_y))

        temp = pd.DataFrame({"pre_next_ball_related":[pred_next_ball_related],
                            "pre_x":[pred_x],
                            "pre_y":[pred_y],
                            "pre_team":[pred_team],
                            "next_ball_related":[int(not(ground_truth.iloc[0,2]==ground_truth.iloc[0,3]==0))],
                            "x":[ground_truth.iloc[0,2]],
                            "y":[ground_truth.iloc[0,3]],
                            "team_id":[ground_truth.iloc[0,1]],})
        test_df = pd.concat([test_df, temp])

    print('\n ave scores/loss score_player={}, score_team={}, loss_xy={}'.format(
        float(score_player)/file_num, float(score_team)/file_num, loss_xy/file_num))
    test_df.to_csv(save_path+"test_df.csv", index=False)

    return test_df

def evaluate_gbdt(save_path=train_path, valid_dir=valid_path, models=[None, None, None, None]):
    """
    Evaluate the given model.
    args:
        valid_dir: the dir path of validation sets.
        models: a list of [player_model, next_ball_related_model, x_model, y_model, team_model]
    return:
        test_df: the test data whose type is pd.DataFrame.
        
    """
    test_df = pd.DataFrame({#"pre_next_ball_related":[],
                            "pre_x":[],
                            "pre_y":[],
                            "pre_team":[],
                            "next_ball_related":[],
                            "x":[],
                            "y":[],
                            "team_id":[],})
    [player_model, x_model, y_model, team_model] = models
    
    files= os.listdir(valid_dir)
    xml_files = [i for i in files if i[-3:]=='xml']
    csv_files = [i for i in files if i[-3:]=='csv']
    xml_files.sort()
    csv_files.sort()
    file_num = len(xml_files)
    
    score_player = loss_xy = score_team = count_num = 0
    
    for i in tqdm(range(file_num)):
        # read the label file
        ground_truth = pd.read_csv(valid_dir+csv_files[i], header=None)
        if ground_truth.iloc[0,2] == ground_truth.iloc[0,3] == 0:
            continue
        game_xml = lxml.etree.parse(valid_dir+xml_files[i])
        
        # get the feature dataframe
        df = construct_one_ball_team_df(game_xml)
        df = get_eva_fea(df)
        gc.collect()
        period_id = df.iloc[-1]['period_id']
        train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1)
        # get the last event
        df_test_X = train_df.iloc[-1][train_df.columns[0:-3]]
        
        pred_player = pred_x = pred_y = pred_team = np.nan
        
        # get the team prediction
        if team_model is not None:
            print("Testing team model...")
            # predict next ball related
            pred_team = team_model.predict(df_test_X)[0]
            if pred_team>=team_gate:
                team = df_test_X.team_id
            else:
                team = 1-df_test_X.team_id
            

        if (x_model is not None) and (y_model is not None):
            print("Testing x,y model...")          
            # predict_xy
            pred_x = x_model.predict(df_test_X)[0].round(1)
            pred_y = y_model.predict(df_test_X)[0].round(1)
        
        if player_model is not None:
            pred_player = player_model.predict(df_test_X)[0]
            

        # rules
        if (df_test_X.type_id == 50):
            team = 1-df_test_X.team_id
            pred_x = 100-df_test_X.x
            pred_y = 100-df_test_X.y
        elif (df_test_X.type_id == 4) or (df_test_X.type_id == 5) or (df_test_X.type_id == 44):
            if df_test_X.type_id != df_test_X.last_type_id:
                team = 1-df_test_X.team_id
                pred_x = 100-df_test_X.x
                pred_y = 100-df_test_X.y
        elif (df_test_X.type_id == 2) or (df_test_X.type_id == 51)or (df_test_X.type_id == 15):
            team = 1-df_test_X.team_id
        elif (df_test_X.type_id == 49):
            team = df_test_X.team_id
        elif (df_test_X.type_id == 25)or (df_test_X.type_id == 18):
            team = df_test_X.team_id
            pred_x = 0
            pred_y = 0
        elif (df_test_X.type_id == 28) or (df_test_X.type_id == 68) or (df_test_X.type_id == 70):
            if df_test_X.type_id != df_test_X.last_type_id:
                team = 1-df_test_X.team_id
                pred_x = 0
                pred_y = 0
        elif (df_test_X.type_id == 49):
            team = 1-df_test_X.team_id
            pred_x = 0
            pred_y = 0
        

        # compute result 
        if team == ground_truth.iloc[0,1]:
            score_team +=1
        loss_xy += (pred_x - ground_truth.iloc[0,2])*(pred_x - ground_truth.iloc[0,2])+\
                            (pred_y - ground_truth.iloc[0,3])*(pred_y - ground_truth.iloc[0,3])
        if pred_player == ground_truth.iloc[0,0]:
                score_player +=1
            
        print('-------------------------------------')
        print('label player={}, team={}, x={}, y={}'.format(
            ground_truth.iloc[0,0],ground_truth.iloc[0,1],ground_truth.iloc[0,2],ground_truth.iloc[0,3]))
        print('prdct player={}, team={}, x={}, y={}'.format(
            pred_player, team, pred_x, pred_y))
        print('-------------------------------------')
        count_num += 1

        temp = pd.DataFrame({#"pre_next_ball_related":[pred_next_ball_related],
                            "pre_x":[pred_x],
                            "pre_y":[pred_y],
                            "pre_team":[pred_team],
                            "next_ball_related":[int(not(ground_truth.iloc[0,2]==ground_truth.iloc[0,3]==0))],
                            "x":[ground_truth.iloc[0,2]],
                            "y":[ground_truth.iloc[0,3]],
                            "team_id":[ground_truth.iloc[0,1]],})
        test_df = pd.concat([test_df, temp])

    print('\n ave scores/loss score_player={}, score_team={}, loss_xy={}'.format(
        float(score_player)/count_num, float(score_team)/count_num, loss_xy/count_num))
    print('count_num: ', count_num)
    test_df.to_csv(save_path+"gbdt_test_df.csv", index=False)

    return test_df



pname = 'pmodel.pkl'
xname = 'xmodel.pkl'
yname = 'ymodel.pkl'
tname = 'tmodel.pkl' 
next_ball_related_name = 'next_ball_related_name.pkl'                   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--valid', type=int, default=0)
    parser.add_argument('--test_p', action='store_true')
    parser.add_argument('--test_xy', action='store_true')
    parser.add_argument('--test_t', action='store_true')
    
    args = parser.parse_args()
    if args.valid != 0:
        construct_val_sets(val_num=args.valid)
    
    bst_x = bst_y = bst_t = bst_p = None
    if args.test_p:
        pass
    if args.test_xy:
        # bst_next_ball_related = pickle.load(open(model_path+next_ball_related_name, 'rb'))
        bst_x = pickle.load(open(model_path+xname, 'rb'))
        bst_y = pickle.load(open(model_path+yname, 'rb'))
    if args.test_t:
        bst_t = pickle.load(open(model_path+tname, 'rb'))
    #evaluation_with_next_ball(models=[None, bst_next_ball_related, bst_x, bst_y, bst_t])
    evaluate_gbdt(models=[bst_p, bst_x, bst_y, bst_t])