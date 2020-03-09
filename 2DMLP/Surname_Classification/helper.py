import numpy as np
import torch as t
def make_train_state(args):
    """return the information during the training
    
    Arguments:
        args {[Namespace]} -- [includes model parameters]
    
    Returns:
        [dictionary] -- [description]
    """
    return {
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val':1e8,
        'learning_rate':args.learning_rate,
        'epoch_index':0,
        'train_loss':[],
        'train_acc':[],
        'val_loss':[],
        'val_acc':[],
        'test_loss':-1,
        'test_acc':-1,
        'model_filename': args.model_file
    }

def update_train_state(args,model,train_state):

    # if the current epoch is the first epoch, then at least store one model
    if train_state['epoch_index'] == 0:
        t.save(model,train_state['model_filename'])
    
    elif train_state['epoch_index'] >= 1:

        last_val_loss = train_state['val_loss'][-1]
        
        # if the loss worsened (if the loss is too large)
        if last_val_loss > train_state['early_stopping_best_val']:
            train_state['early_stopping_step'] += 1
        
        else:
            # if the loss decreases
            if last_val_loss < train_state['early_stopping_best_val']:
                t.save(model,train_state['model_filename'])
            
            train_state['early_stopping_step'] = 0
        
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria
    return train_state

def compute_accuracy(y_pred,y_target):


    # tensor.max(dim = 1) will return the maximal value in each row
    # it will return a tuple (values[type:tensor], indexes[type:tensor])  
    _,y_pred_idx = y_pred.max(dim = 1)
    # item will return the value of the tensor
    n_correct = t.eq(y_target,y_pred_idx).sum().item()
    return n_correct/ len(y_target) * 100

def set_seed_everywhere(seed):
    np.random.seed(seed)
    t.manual_seed(seed)









