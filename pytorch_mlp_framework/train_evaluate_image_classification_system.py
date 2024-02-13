import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import *

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Config(object):
    def __init__(self):
        # Parameters for training target network
        self.batch_size = 16
        self.num_workers = 4
        self.epochs = 25
        self.num_classes = 10

opt = Config()

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

def __gray2RGB(img):
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

train_transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: __gray2RGB(img)),  # 将灰度图转换成3通道
        transforms.ToTensor(),
    ]
)
test_transform = train_transform

data_path = '../../data'
train_dataset = datasets.MNIST(root=data_path, train=True, transform=train_transform, download=True)
train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
test_dataset = datasets.MNIST(root=data_path, train=False, transform=test_transform, download=True)

dataloaders = {}
dataset_sizes = {}

train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)
val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)
test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)

blocks = {'conv_block': [ConvolutionalProcessingBlock, ConvolutionalDimensionalityReductionBlock], 
          'bn_conv_block': [BN_ConvolutionalProcessingBlock, BN_ConvolutionalDimensionalityReductionBlock], 
          'bn_rc_conv_block': [BN_RC_ConvolutionalProcessingBlock, BN_RC_ConvolutionalDimensionalityReductionBlock], 
          'empty_block': [EmptyBlock, EmptyBlock]}


if args.block_type in blocks:
    processing_block_type, dim_reduction_block_type = blocks[args.block_type]
else:
    raise ModuleNotFoundError(f'"{args.block_type}" not found')

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    num_output_classes=args.num_classes, 
    num_filters=args.num_filters, 
    use_bias=False,
    num_blocks_per_stage=args.num_blocks_per_stage, num_stages=args.num_stages,
    processing_block_type=processing_block_type, 
    dimensionality_reduction_block_type=dim_reduction_block_type)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate, # TASK 3: Added initial_lr parameter to adjust initial learning rate
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
