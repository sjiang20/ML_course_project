import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import tree
from utils import evaluation
from utils import print_info

def experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=(100), solver='adam', alpha=0.0001, activation='relu'):
    params_dict = {}

    if isinstance(layers, tuple):
        for i in range(len(layers)):
            params_dict['Layer ' + str(i)]=layers[i]
    else:
        params_dict['Layers 1']=layers

    params_dict['solver'] = solver
    params_dict['alpha'] = alpha
    params_dict['activation'] = activation
    params_dict['Training samples'] = len(data_train)
    params_dict['Testing samples'] = len(data_test)

    print_info.print_experiment_start(experiment_name, params_dict)

    nn = MLPClassifier(hidden_layer_sizes=layers, solver = solver, alpha = alpha, activation = activation)
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

def experiment_layers(data_name, data_train, data_test, target_train, target_test):
    layers = (128)
    experiment_name = 'NN-layers-l1-128'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=layers)

    layers = (128, 64)
    experiment_name = 'NN-layers-l1-128-l2-64'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=layers)

    layers = (128, 64, 32)
    experiment_name = 'NN-layers-l1-128-l2-64-l3-32'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=layers)

    layers = (128, 64, 32, 16)
    experiment_name = 'NN-layers-l1-128-l2-64-l3-32-l4-16'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=layers)

    layers = (128, 64, 32, 16, 8)
    experiment_name = 'NN-layers-l1-128-l2-64-l3-32-l4-16-l5-8'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, layers=layers)

def experiment_regularization(data_name, data_train, data_test, target_train, target_test):
    layers = (128, 64, 32)

    alpha = 0.0001
    experiment_name = 'NN-regularization-0.0001'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, alpha = alpha, layers=layers)

    alpha = 0.001
    experiment_name = 'NN-regularization-0.001'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, alpha=alpha, layers=layers)

    alpha = 0.01
    experiment_name = 'NN-regularization-0.01'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, alpha=alpha, layers=layers)

    alpha = 0.1
    experiment_name = 'NN-regularization-0.1'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, alpha=alpha, layers=layers)

def experiment_activation(data_name, data_train, data_test, target_train, target_test):
    layers = (128, 64, 32)
    alpha = 0.01
    activation = 'relu'
    experiment_name = 'NN-activation-relu'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, activation=activation, layers=layers, alpha=alpha)

    activation = 'identity'
    experiment_name = 'NN-activation-identity'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, activation=activation, layers=layers, alpha=alpha)

    activation = 'logistic'
    experiment_name = 'NN-activation-logistic'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, activation=activation, layers=layers, alpha=alpha)

    activation = 'tanh'
    experiment_name = 'NN-activation-tanh'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, activation=activation, layers=layers, alpha=alpha)

def experiment_solver(data_name, data_train, data_test, target_train, target_test):
    solver = 'adam'
    experiment_name = 'NN-solver-adam'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, solver=solver)

    solver = 'sgd'
    experiment_name = 'NN-solver-sgd'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, solver=solver)

    solver = 'lbfgs'
    experiment_name = 'NN-solver-lbfgs'
    experiment_nn(data_name, data_train, data_test, target_train, target_test, experiment_name, solver=solver)

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
        experiment_nn(data_name, data_train_portion, data_test, target_train_portion, target_test, experiment_name,layers=(128, 64, 32), solver='adam', alpha=0.0001, activation='tanh')

def run_nn_experiments(data_name, data_train, data_test, target_train, target_test, tr_portion=False):
    if tr_portion:
        experiment_tr_portion(data_name, data_train, data_test, target_train, target_test)
        return True
    experiment_layers(data_name, data_train, data_test, target_train, target_test)
    experiment_regularization(data_name, data_train, data_test, target_train, target_test)
    experiment_activation(data_name, data_train, data_test, target_train, target_test)
    experiment_solver(data_name, data_train, data_test, target_train, target_test)