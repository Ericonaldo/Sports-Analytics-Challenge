# -*- coding:utf-8 -*-
from utils import *
import argparse

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

def evaluation(save_path=train_path, valid_dir=valid_path):
    """
    Evaluate the rules.
    args:
        valid_dir: the dir path of validation sets.
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
        
        ## test teams
        print("Testing team model...")
        team = rnd.randint(0,1)

        # rules
        if (df_test_X.type_id == 18) or (df_test_X.type_id == 25):
            team = df_test_X.loc[0, 'team_id']
        elif df_test_X.type_id == 15:
            team = 1-df_test_X.loc[0, 'team_id']
        elif (df_test_X.type_id == 4) or (df_test_X.type_id == 5):
            if df_test_X.type_id == df_test_X.last_type_id:
                team = 1-df_test_X.loc[0, 'team_id']
        # compute result
        if team == ground_truth.iloc[0,1]:
            score_team +=1
        
        ## test xy
        print("Testing x,y model...")
        pred_x = pred_y = 0
        # rules
        if df_test_X.type_id == map_type_id(15):
                pred_x = df_test_X.loc[0, 'x']
                pred_y = df_test_X.loc[0, 'y']
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
         
        ## test players
        # pass
        '''
        pred_player = player_model.predict(df_test_X)[0]
        if pred_player == ground_truth.iloc[0,0]:
            score_player +=1
        '''
            
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

                  
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--valid', type=int, default=0)
    
    args = parser.parse_args()
    if args.valid != 0:
        construct_val_sets(val_num=args.valid)

    evaluation()