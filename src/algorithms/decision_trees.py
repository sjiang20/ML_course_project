import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils import evaluation
from utils import print_info


def experiment_decision_trees(data_name, data_train, data_test, target_train, target_test,
                              experiment_name, depth=1, criteria=None):

    params_dict = {'Depth': depth,
                   'Training samples': len(data_train),
                   'Testing samples': len(data_test),
                   'Creteria': 'entropy'}
    print_info.print_experiment_start(experiment_name, params_dict)

    dt = DecisionTreeClassifier(max_depth=depth,criterion=criteria)
    dt.fit(data_train, target_train)
    y_train = dt.predict(data_train)
    y_test = dt.predict(data_test)

    # Get evaluation for training
    eva_metrics = ['accuracy', 'f1', 'precision', 'recall']
    evaluations = evaluation.get_evaluations(target_train, y_train, eva_metrics)
    print_info.print_evaluation(evaluations, eva_metrics, is_test=False)

    # Get evaluation for testing
    eva_metrics = ['accuracy', 'f1', 'precision', 'recall']
    evaluations = evaluation.get_evaluations(target_test, y_test, eva_metrics)
    print_info.print_evaluation(evaluations, eva_metrics, is_test=True)
    print_info.print_experiment_end()

    str_config = ''
    for key in params_dict:
        str_config = str_config+'_'+key+'_'+str(params_dict[key])


def experiment_original(data_name, data_train, data_test, target_train, target_test):
    criteria = 'entropy'
    experiment_name = 'Dec ision trees original'
    full_depth = get_full_depth(data_train, target_train)
    experiment_decision_trees(data_name, data_train, data_test, target_train, target_test, \
                              experiment_name, full_depth, criteria=criteria)


def get_full_depth(data_train, target_train):
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(data_train, target_train)
    depth = dt.tree_.max_depth
    return depth

def experiment_pruning(data_name, data_train, data_test, target_train, target_test):
    full_depth = get_full_depth(data_train, target_train)

    experiment_name = 'Decition trees pruning'
    criteria = 'entropy'

    for depth in range(1, full_depth+1):
        experiment_decision_trees(data_name, data_train, data_test, target_train, target_test, \
                                  experiment_name, depth, criteria=criteria)

def experiment_tr_portion(data_name, data_train, data_test, target_train, target_test):
    import random
    l = len(data_train)
    idx = range(l)
    random.shuffle(idx)

    for i in range(10):
        p = 1.0*(i+1)/10.0

        index = idx[0:np.int16(l*p)]
        data_train = np.asarray(data_train)
        target_train = np.asarray(target_train)
        data_train_portion = data_train[index, :]
        target_train_portion = target_train[index]
        experiment_name = 'boosting-tr-portion-'+str(i)+'/10'

        full_depth =  get_full_depth(data_train_portion, target_train_portion)

        experiment_decision_trees(data_name, data_train_portion, data_test, target_train_portion, target_test, experiment_name, depth=max(1, full_depth-1), criteria='entropy')

def run_decision_trees_experiments(data_name, data_train, data_test, target_train, target_test, tr_portion=False):
    if tr_portion:
        experiment_tr_portion(data_name, data_train, data_test, target_train, target_test)
        return True
    experiment_original(data_name, data_train, data_test, target_train, target_test)
    experiment_pruning(data_name, data_train, data_test, target_train, target_test)