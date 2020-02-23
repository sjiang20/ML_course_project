import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import tree
from utils import evaluation
from utils import print_info


def experiment_knn(data_name, data_train, data_test, target_train, target_test, experiment_name, n_neighbors=5, weights = 'uniform'):
    params_dict = {}

    params_dict['n_neighbors'] = n_neighbors
    params_dict['weights'] = weights
    params_dict['Training samples'] = len(data_train)
    params_dict['Testing samples'] = len(data_test)

    print_info.print_experiment_start(experiment_name, params_dict)

    nn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    nn.fit(data_train, target_train)
    y_test = nn.predict(data_test)
    y_train = nn.predict(data_train)

    # Get evaluation for training
    eva_metrics = ['accuracy', 'f1', 'precision', 'recall']
    evaluations = evaluation.get_evaluations(target_train, y_train, eva_metrics)
    print_info.print_evaluation(evaluations, eva_metrics, is_test=False)

    # Get evaluation for testing
    eva_metrics = ['accuracy', 'f1', 'precision', 'recall']
    evaluations = evaluation.get_evaluations(target_test, y_test, eva_metrics)
    print_info.print_evaluation(evaluations, eva_metrics, is_test=True)
    print_info.print_experiment_end()

def experiment_n_neighbors(data_name, data_train, data_test, target_train, target_test):
    for n_neighbors in range(5, 50, 5):
        experiment_name = 'KNN-n_neighbors-k-'+str(n_neighbors)
        experiment_knn(data_name, data_train, data_test, target_train, target_test, experiment_name, n_neighbors)

def experiment_weights(data_name, data_train, data_test, target_train, target_test):
    weights = 'uniform'
    experiment_name = 'KNN-weights-uniform'
    experiment_knn(data_name, data_train, data_test, target_train, target_test, experiment_name, weights=weights)

    weights = 'distance'
    experiment_name = 'KNN-weights-distance'
    experiment_knn(data_name, data_train, data_test, target_train, target_test, experiment_name, weights=weights)

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
        experiment_knn(data_name, data_train_portion, data_test, target_train_portion, target_test, experiment_name, n_neighbors=5, weights = 'uniform')

def run_knn_experiments(data_name, data_train, data_test, target_train, target_test, tr_portion=False):
    if tr_portion:
        experiment_tr_portion(data_name, data_train, data_test, target_train, target_test)
        return True
    experiment_n_neighbors(data_name, data_train, data_test, target_train, target_test)