import os
import json
import argparse
import datetime
import time
from utils import createFolder, str2list_int, str2list, print_info, band_list


def arg(model, run_mode, subject):
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_subject', default=subject, help="Please do not enter any value.")
    # parser.add_argument('--val_subject', type=int, nargs='+', default=[1])
    parser.add_argument('--batch_size', type=int, default=72)

    if model=='EEGNet':
        # Net
        parser.add_argument('--net', default="EEGNet")

        if run_mode=='train':
            parser.add_argument('--mode', default="train", choices=['test', 'debug'])
            parser.add_argument('--pretrained_path', type=str, default=None,
                                help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
            parser.add_argument('--epochs', type=int, default=200)
            parser.add_argument('--stamp', default='EEGNet_200epoch', help="Please enter stamp.")
        elif run_mode=='test':
            parser.add_argument('--mode', default="test", choices=['train', 'debug'])
            parser.add_argument('--pretrained_path', type=str, default='EEGNet_200epoch',
                                help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
            parser.add_argument('--epochs', type=int, default=50)
            parser.add_argument('--stamp', default='EEGNet_50epoch', help="Please enter stamp.")
        else:
            raise ValueError("Please enter run_mode correctly.")

    elif model=='deep4':
        # Net
        parser.add_argument('--net', default="deep4")

        if run_mode == 'train':
            parser.add_argument('--mode', default="train", choices=['test', 'debug'])
            parser.add_argument('--pretrained_path', type=str, default=None,
                                help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
            parser.add_argument('--epochs', type=int, default=200)
            parser.add_argument('--stamp', default='deep4_200epoch', help="Please enter stamp.")
        elif run_mode == 'test':
            parser.add_argument('--mode', default="test", choices=['train', 'debug'])
            parser.add_argument('--pretrained_path', type=str, default='deep4_200epoch',
                                help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
            parser.add_argument('--epochs', type=int, default=50)
            parser.add_argument('--stamp', default='deep4_50epoch', help="Please enter stamp.")
        else:
            raise ValueError("Please enter run_mode correctly.")

    else:
        raise ValueError("Please enter net correctly.")

    # Time
    parser.add_argument('--start_time', default=time.time(), help="Please do not enter any value.")
    parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
    parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")

    # Mode
    parser.add_argument('--train_cont_path', type=int, default=None, help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
    parser.add_argument('--paradigm', default="session", choices=['ind'], help="Please enter ind or not")

    # Data
    parser.add_argument('--dataset', default='bcic4_2a')
    parser.add_argument('--band', type=band_list, default=[[0, 42]], help="Please connect it with a comma.")
    parser.add_argument('--chans', default='all', type=str2list, help="Please connect it with a comma.")
    parser.add_argument('--labels', default='0,1,2,3', type=str2list_int, help="Please connect it with a comma.")

    # Train
    parser.add_argument('--criterion', default='CEE', help="Please enter loss function you want to use.")
    parser.add_argument('--opt', default='Adam', help="Please enter optimizer you want to use.")
    parser.add_argument('--metrics', default='loss,acc', type=str2list, help="Please connect it with a comma.")
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=2e-04)
    parser.add_argument('--weight_decay', '-wd', dest='wd', type=float, default=2e-04)

    parser.add_argument('--scheduler', '-sch', default='exp', choices=['exp', 'cos'])
    if parser.parse_known_args()[0].scheduler == 'exp':
        parser.add_argument('--gamma', type=float, default=0.999)
    elif parser.parse_known_args()[0].scheduler == 'cos':
        parser.add_argument('--eta_min', type=float, required=True)

    # Path
    parser.add_argument('--save_path')

    # Miscellaneous
    parser.add_argument('--gpu', default=0, help="multi / 0 / 1 / cpu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_step', default=5, type=int, help="Number of print results per epoch.")
    parser.add_argument('--signature', help="To filter the results.")

    # Parsing
    args = parser.parse_args()

    # Set train subject
    args.train_subject = [int(args.train_subject)]

    # Set save_path
    assert args.stamp is not None, "You Should enter stamp."
    if args.train_cont_path:
        args.save_path = os.path.dirname(os.path.dirname(args.train_cont_path))
    else:
        if args.pretrained_path:
            args.save_path = f"./result/{args.stamp}/{args.train_subject[0]}"
        else:
            args.save_path = f"./pretrained/{args.stamp}/{args.train_subject[0]}"
    createFolder(args.save_path)

    # Print
    print_info(vars(args))
    return args
