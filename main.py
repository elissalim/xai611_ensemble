import numpy as np
from config import arg
from data_loader import data_loader
from build_net import build_net
from make_solver import make_solver
from utils import control_random
from sklearn.metrics import accuracy_score
import json


def main(run_mode, subject):
    model_lst = ['deep4', 'EEGNet']
    prob_lst = []
    for model in model_lst:
        args = arg(model, run_mode, subject)

        # seed control
        if args.seed:
            control_random(args)

        # load train / test dataset
        # TODO: change how test should be really done, split train-val-test
        train_loader, val_loader = data_loader(args)

        # import backbone model
        net = build_net(args, train_loader.dataset.X.shape)

        # make solver (runner)
        solver = make_solver(args, net, train_loader, val_loader)

        # train/test
        probs = solver.experiment()
        prob_lst.append(probs)

    if run_mode == 'test':
        # combine the predictions using a weighted average
        ensemble_pred = (prob_lst[0] + prob_lst[1]) / len(model_lst)

        # convert the predictions to class labels
        ensemble_pred_labels = np.argmax(ensemble_pred, axis=1)

        # calculate the accuracy of the ensemble model
        ensemble_accuracy = accuracy_score(val_loader.dataset.y, ensemble_pred_labels)
        print('Subject:'+str(subject)+'Ensemble Accuracy:', ensemble_accuracy)

        return ensemble_accuracy

if __name__ == '__main__':
    run_mode = 'test' # 'train' or 'test'
    accuracy = {}
    if run_mode == 'test':
        for subject in range(1, 10):
            accuracy[str(subject)] = main(run_mode, subject)
        print(accuracy)

        with open('result.json', 'w') as fp:
            json.dump(accuracy, fp)
        print("====================================Finish====================================")

        print('Mean accuracy:', np.mean(list(accuracy.values())))
    else:
        for subject in range(1, 10):
            main(run_mode, subject)
