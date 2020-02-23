import argparse
import os
from utils import input_parser, data_loader
import algorithms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def main():
    args = input_parser.get_input()
    data_path = os.path.join(os.path.join('../data', args.data), args.data+'.csv')
    titels, X, y = data_loader.get_data(data_path)

    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    if args.tr_portion:
        algorithms.__dict__['run_'+args.algorithm+'_experiments'](args.data, X_train, X_test, y_train, y_test, args.tr_portion)
    else:
        algorithms.__dict__['run_' + args.algorithm + '_experiments'](args.data, X_train, X_test, y_train, y_test)

    return True

if __name__ == '__main__':
    main()
