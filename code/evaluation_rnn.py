from utils import *
from train_team_event_rnn import *

model_name = 'team_events_rnn.pkl'                   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--valid', type=int, default=0)
    parser.add_argument('--test_p', action='store_true')
    parser.add_argument('--test_xyt', action='store_true')
    
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