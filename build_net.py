import torch
import torch.nn as nn
from Net import EEGNet_net, deep4_net


def build_net(args, shape):
    print("[Build Net]")

    if args.net == 'EEGNet':
        net = EEGNet_net.EEGNet(args, shape)
    elif args.net == 'deep4':
        net = deep4_net.Deep4Net(
            shape[2],
            len(args.labels),
            shape[3],
            final_conv_length='auto'
        )
    else:
        raise NotImplementedError

    # load pretrained parameters
    if args.mode == 'test':
        # Do test using pretrained model
        if args.net == 'EEGNet' or args.net == 'deep4':
            m_path = "./pretrained/" + args.pretrained_path + "/"
            param = torch.load(m_path + f"{args.train_subject[0]}/checkpoint/200.tar")
            net.load_state_dict(param['net_state_dict'])
        else:
            raise NotImplementedError
    # Set GPU
    if args.gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if args.gpu == "multi":
            device = args.gpu
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
        net.cuda()

    # Set CPU
    else:
        device = torch.device("cpu")

    # Print
    print(f"device: {device}")
    print("")

    return net
