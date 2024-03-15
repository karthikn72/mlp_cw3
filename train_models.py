
import numpy as np
from cub2011 import Cub2011
from pytorch_mlp_framework.experiment_builder import ExperimentBuilder
from pytorch_mlp_framework.arg_extractor import get_args

import torch
from torchvision import transforms, datasets
from torchvision.models import resnet50

def get_dataloaders(dataloader, transform, batch_size=32):
    if dataloader == 'aircrafts':
        train_data = datasets.FGVCAircraft(root='data', download=False, transform=transform, split='train')
        val_data = datasets.FGVCAircraft(root='data', download=False, transform=transform, split='val')
        test_data = datasets.FGVCAircraft(root='data', download=False, transform=transform, split='test')
        targets = train_data.classes
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(val_data))
        print("Number of testing samples: ", len(test_data))
        
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
    elif dataloader == 'birds':
        train_data = Cub2011(root='data', train=True, transform=transform, download=False)
        targets = train_data.classes
        # Split train_bird_data into training and validation sets
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
        test_data = Cub2011(root='data', train=False, transform=transform, download=False)
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(val_data))
        print("Number of testing samples: ", len(test_data))
        # Create the bird DataLoader
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # print(train_data._labels)
    return train_dataloader, val_dataloader, test_dataloader, targets

def get_models(model, pretrain_scheme):
    if model == 'resnet50':
        if pretrain_scheme == 'imagenet':
            model = resnet50(pretrained=True)
        else:
            model = resnet50(pretrained=False)
    
    return model

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((args.height, args.width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders(dataloader=args.dataloader, 
                                                                                 transform=transform, 
                                                                                 batch_size=args.batch_size)
    model = get_models(args.model, args.pretrain)
    num_classes = len(classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    conv_experiment = ExperimentBuilder(network_model=model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        learning_rate=args.learning_rate,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=args.continue_from_epoch,
                                        train_data=train_dataloader, 
                                        val_data=val_dataloader,
                                        test_data=test_dataloader)  # build an experiment object
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics