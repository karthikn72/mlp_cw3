import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Epoch you want to continue training from while restarting an experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='Total number of epochs for model training')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-3,
                        help='Learning rate to use for Adam') # ---------------------> TASK 3
    parser.add_argument('--model', type=str, default='resnet50',
                        help='The model to use for the experiment')
    parser.add_argument('--pretrain', type=str, default='imagenet',
                        help='The pretraining scheme to use for the experiment')
    parser.add_argument('--dataloader', type=str, default='birds',
                        help='The dataset to finetune on for the experiment')
    parser.add_argument('height', type=int, default=224, help='The height of the input image')
    parser.add_argument('width', type=int, default=224, help='The width of the input image')
    
    args = parser.parse_args()
    print(args)
    return args
