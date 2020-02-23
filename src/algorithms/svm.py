from sklearn import svm
from utils import evaluation
from utils import print_info
import numpy as np

def experiment_svm(data_name, data_train, data_test, target_train, target_test,
                              experiment_name, C=1.0, gamma=1.0, kernel='rbf'):
    params_dict = {}

    params_dict['c'] = C
    params_dict['gamma'] = gamma
    params_dict['kernel'] = kernel
    params_dict['Training samples'] = len(data_train)
    params_dict['Testing samples'] = len(data_test)

    print_info.print_experiment_start(experiment_name, params_dict)

    nn = svm.SVC(C=C, gamma = gamma, kernel=kernel)
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

def experiment_gamma(data_name, data_train, data_test, target_train, target_test):
    g_pool = []
    g_pool.append(0.0001)
    g_pool.append(0.001)
    g_pool.append(0.01)
    g_pool.append(0.1)
    g_pool.append(1)

    for g in g_pool:
        experiment_name = 'svm-gamma-'+str(g)
        experiment_svm(data_name, data_train, data_test, target_train, target_test, experiment_name, gamma=g)

    return True


def experiment_c(data_name, data_train, data_test, target_train, target_test):
    c_pool = []
    c_pool.append(0.0001)
    c_pool.append(0.001)
    c_pool.append(0.01)
    c_pool.append(0.1)
    c_pool.append(1)

    for c in c_pool:
        experiment_name = 'svm-c-'+str(c)
        experiment_svm(data_name, data_train, data_test, target_train, target_test, experiment_name, C=c)
    return True

def experiment_kernel(data_name, data_train, data_test, target_train, target_test):
    kernel_pool = {'rbf', 'poly', 'linear'}

    for k in kernel_pool:
        experiment_name = 'svm-kernel-'+k
        experiment_svm(data_name, data_train, data_test, target_train, target_test, experiment_name, kernel=k)
    return True

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
        experiment_svm(data_name, data_train_portion, data_test, target_train_portion, target_test, experiment_name, C=0.1, gamma=0.1, kernel='rbf')

def run_svm_experiments(data_name, data_train, data_test, target_train, target_test, tr_portion=False):
    if tr_portion:
        experiment_tr_portion(data_name, data_train, data_test, target_train, target_test)
        return True
    experiment_c(data_name, data_train, data_test, target_train, target_test)
    experiment_kernel(data_name, data_train, data_test, target_train, target_test)