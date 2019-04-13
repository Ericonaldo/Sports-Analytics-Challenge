# -*- coding:utf-8 -*-
from utils import *
import argparse

categorical_feature = ['team_id', 'game_time','last_type_id', 'last_team_id',
                       'keypass', 'last_keypass', 'assist', 'last_assist', 'field' ,'penal_zone', 
                       'penal_point', 'zone', 'last_field', 'last_penal_zone', 'last_zone', 'last_penal_point',
                       'ball_related']#, 'type_id', 'next_ball_related']
pname = 'pmodel.pkl'
xname = 'xmodel.pkl'
yname = 'ymodel.pkl'
tname = 'tmodel.pkl'
next_ball_related_name = 'next_ball_related_name.pkl'

def train_bst_class(X_tr, y_tr, X_val, y_val, feature_names='auto', categorical_feature=categorical_feature, tr_w=None, val_w=None):
    params = {  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    #'metric': {'binary_logloss', 'auc'},  #二进制对数损失
    'num_leaves': 31,  
    #'max_depth': 6,  
    'min_data_in_leaf': 300,  
    'learning_rate': 0.01,  
    'feature_fraction': 0.8,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 2,  
    'lambda_l1': 1,    
    'lambda_l2': 0.01,  # 越小l2正则程度越高  
    #min_gain_to_split': 0.2, 
    'min_data': 1, 
    'min_data_in_bin': 1,
    #'is_unbalance': True  
    }  

    MAX_ROUNDS = 15000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature = categorical_feature,
        weight=tr_w,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
        categorical_feature = categorical_feature,
        weight=val_w,
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100,
        feature_name=list(feature_names),
    )
    return bst

def train_bst_reg(X_tr, y_tr, X_val, y_val, feature_names='auto', categorical_feature=categorical_feature):
    params = {
        'boosting': 'gbdt',
        'num_leaves': 31,
        'objective': 'regression',
        'min_data_in_leaf': 300,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'metric': 'l2',
        #'max_depth': 6,
        'num_threads': 8, 
        'min_data': 1, 
        'min_data_in_bin': 1, 
        'lambda_l2': 0.01,  # 越小l2正则程度越高 
        # 'device': 'gpu', 
    }
    
    MAX_ROUNDS = 10000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature = categorical_feature
    )

    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
        categorical_feature = categorical_feature
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100,
        feature_name=list(feature_names),
    )
    return bst

def get_train_val(tr_X, tr_y):
    val_num = int(len(tr_X) / 10)
    return tr_X.iloc[:-val_num].values, tr_y[:-val_num].values, tr_X.iloc[-val_num:].values, tr_y.iloc[-val_num:].values


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--corr', action='store_true')
    parser.add_argument('--pmodel', action='store_true')
    parser.add_argument('--xymodel', action='store_true')
    parser.add_argument('--tmodel', action='store_true')
    
    args = parser.parse_args()
    
    ## model of x,y and team
    if not os.path.exists(train_path+"ball_team_df.csv"):
        df = construct_ball_team_df()
    else:
        df = pd.read_csv(train_path+"ball_team_df.csv")
    df = df.drop(df[(df['min']<=10) & (df['period_id']==0)].index, axis=0) # drop the events before 10mins in period 1
    df = df.drop(df[(df['min']<=55) & (df['period_id']==1)].index, axis=0) # drop the events before 10mins in period 2
    train_df = df.dropna().drop(['game_id','event_no','period_id'], axis=1).sample(frac=1).reset_index(drop=True)
    
    if args.corr:
        draw_corr(train_df)
    """
    next_ball_train_df = train_df.copy()
    next_ball_train_df['weight_column'] = next_ball_train_df.next_ball_related
    next_ball_train_df.loc[next_ball_train_df.weight_column==0, 'weight_column'] = 2
    next_ball_train_df.loc[next_ball_train_df.weight_column==1, 'weight_column'] = 1
    # resample
    length = len(next_ball_train_df[next_ball_train_df.next_ball_related==0].index)
    choice_idx = np.random.randint(low=0, high=length-1, size=15*length)
    choice_idx = next_ball_train_df[next_ball_train_df.next_ball_related==0].index[choice_idx]
    next_ball_train_df = pd.concat([next_ball_train_df, next_ball_train_df.iloc[choice_idx]]).sample(frac=1).reset_index(drop=True)
    
    df_train_X = next_ball_train_df[next_ball_train_df.columns[0:-4]]
    df_train_y = next_ball_train_df[next_ball_train_df.columns[-5:]]
    X_tr, y_tr, X_val, y_val = get_train_val(df_train_X, df_train_y)
    tr_w, val_w = y_tr[:,-1], y_val[:,-1]
    
    bst_next_ball_related = train_bst_class(X_tr[:,0:-1], y_tr[:,0], X_val[:,0:-1], y_val[:,0], feature_names=df_train_X.columns[0:-1]
        , categorical_feature=categorical_feature[0:-1], tr_w=tr_w, val_w=val_w)
    pickle.dump(bst_next_ball_related, open(model_path+next_ball_related_name, 'wb'))
    lgb.plot_importance(bst_next_ball_related, max_num_features=30)
    plt.title("Feature importance of model bst_next_ball_related")
    plt.savefig("feature_importance_bst_next_ball_related.png")
    """
    df_train_X = train_df[train_df.columns[0:-3]]
    df_train_y = train_df[train_df.columns[-3:]]
    X_tr, y_tr, X_val, y_val = get_train_val(df_train_X, df_train_y)
    if args.tmodel:
        print("Training team model...")
        bst_t = train_bst_class(X_tr, y_tr[:,2], X_val, y_val[:,2], feature_names=df_train_X.columns, categorical_feature=categorical_feature)
        pickle.dump(bst_t, open(model_path+tname, 'wb'))
        lgb.plot_importance(bst_t, max_num_features=30)
        plt.title("Feature importance of model team")
        plt.savefig("feature_importance_x.png")
    
    
    # drop those whose next ball related = 0
    """
    df_train_X.drop(df_train_y[df_train_y.next_ball_related==0].index, inplace=True)
    df_train_y.drop(df_train_y[df_train_y.next_ball_related==0].index, inplace=True)
    X_tr, y_tr, X_val, y_val = get_train_val(df_train_X, df_train_y)
    """
    df_train_X.drop(df_train_X[(df_train_y.next_x==0) & (df_train_y.next_y==0)].index, inplace=True)
    df_train_y.drop(df_train_X[(df_train_y.next_x==0) & (df_train_y.next_y==0)].index, inplace=True)
    # df_train_X.drop(df_train_X[df_train_X.type_id in rules_x_y].index, inplace=True)
    # df_train_y.drop(df_train_X[df_train_X.type_id in rules_x_y].index, inplace=True)
    X_tr, y_tr, X_val, y_val = get_train_val(df_train_X, df_train_y)

    if args.xymodel:
        print("Training x model...")
        bst_x = train_bst_reg(X_tr, y_tr[:,0], X_val, y_val[:,0], feature_names=df_train_X.columns
            , categorical_feature=categorical_feature)
        pickle.dump(bst_x, open(model_path+xname, 'wb'))
        lgb.plot_importance(bst_x, max_num_features=30)
        plt.title("Feature importance of model x")
        plt.savefig("feature_importance_x.png")

        print("Training y model...")
        bst_y = train_bst_reg(X_tr, y_tr[:,1], X_val, y_val[:,1], feature_names=df_train_X.columns
            , categorical_feature=categorical_feature)
        pickle.dump(bst_y, open(model_path+yname, 'wb'))
        lgb.plot_importance(bst_y, max_num_features=30)
        plt.title("Feature importance of model y")
        plt.savefig("feature_importance_y.png")

    ## TODO something about player model
    if args.pmodel:
        pass