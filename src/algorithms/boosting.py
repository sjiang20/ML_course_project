import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from utils import evaluation
from utils import print_info

def experiment_boosting(data_name, data_train, data_test, target_train, target_test, experiment_name, depth=3, n_classifiers = 100):
    params_dict = {}
    params_dict['n_classifiers'] = n_classifiers
    params_dict['depth'] = depth
    params_dict['Training samples'] = len(data_train)
    params_dict['Testing samples'] = len(data_test)

    print_info.print_experiment_start(experiment_name, params_dict)

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth),
                             n_estimators=n_classifiers,
                             algorithm ="SAMME")
    clf.fit(data_train, target_train)

    y_train = clf.predict(data_train)
    y_test = clf.predict(data_test)

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
        str_config = str_config + '_' + key + '_' + str(params_dict[key])

def experiment_n_classifiers(data_name, data_train, data_test, target_train, target_test):

    for n_classifier in range(20, 110, 20):
        experiment_name = 'boosting-k-'+str(n_classifier)
        experiment_boosting(data_name, data_train, data_test, target_train, target_test, experiment_name,
                            n_classifiers=n_classifier)

    return True

def get_full_depth(data_train, target_train):
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(data_train, target_train)
    depth = dt.tree_.max_depth
    return depth

def experiment_depth(data_name, data_train, data_test, target_train, target_test):
    full_depth = get_full_depth(data_train, target_train)
    for depth in range(1, full_depth, 2):
        experiment_name = 'boosting-depth-'+str(depth)
        experiment_boosting(data_name, data_train, data_test, target_train, target_test, experiment_name, depth=depth)

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
        experiment_boosting(data_name, data_train_portion, data_test, target_train_portion, target_test, experiment_name, depth=3,
                            n_classifiers=100)

def run_boosting_experiments(data_name, data_train, data_test, target_train, target_test, tr_portion=False):
    if tr_portion:
        experiment_tr_portion(data_name, data_train, data_test, target_train, target_test)
        return True
    experiment_n_classifiers(data_name, data_train, data_test, target_train, target_test)
    experiment_depth(data_name, data_train, data_test, target_train, target_test)