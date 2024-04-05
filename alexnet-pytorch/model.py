"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
import json

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 30  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_INIT = 0.001
LR_DECAY = 0.0005
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 51 
DEVICE_IDS = [0]  # GPUs to use

INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'C:/Users/klouc/Desktop/cogrob/data/rgbd-dataset/train'
VAL_IMG_DIR = 'C:/Users/klouc/Desktop/cogrob/data/rgbd-dataset/val'
TEST_IMG_DIR = 'C:/Users/klouc/Desktop/cogrob/data/rgbd-dataset/test'

OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.autograd.set_detect_anomaly(True)

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    train_dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    val_dataset = datasets.ImageFolder(VAL_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    test_dataset = datasets.ImageFolder(TEST_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    
    val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    result = []

    # start training!!
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        train_loss_all = []
        train_accuracy_all = []

        lr_scheduler.step()
        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            train_output = alexnet(imgs)
            train_loss = F.cross_entropy(train_output, classes)

            # update the parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():

                    _, train_preds = torch.max(train_output, 1)
                    train_accuracy = torch.sum(train_preds == classes).item() / train_preds.size(dim=0)

                    train_loss_all.append(train_loss.item())
                    train_accuracy_all.append(train_accuracy)

                    print('Epoch: {} \tStep: {} \tTrain Loss: {} \tTrain Acc: {}'
                        .format(epoch + 1, total_steps, train_loss.item(), train_accuracy))
                    tbwriter.add_scalar('loss', train_loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', train_accuracy, total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    # print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            # print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            # print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1

        with torch.no_grad():            
            val_loss_all = []
            val_accuracy_all = []
            for val_imgs, val_classes in val_dataloader:
                val_imgs, val_classes = val_imgs.to(device), val_classes.to(device)
                
                val_output = alexnet(val_imgs)
                val_loss = F.cross_entropy(val_output, val_classes)

                _, val_preds = torch.max(val_output, 1)
                val_accuracy = torch.sum(val_preds == val_classes).item() / val_preds.size(dim=0)

                val_loss_all.append(val_loss.item())
                val_accuracy_all.append(val_accuracy)

            print('VALIDATION', epoch + 1, 'loss: ', np.mean(val_loss_all), 'acc: ', np.mean(val_accuracy_all))

            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': np.mean(train_loss_all),
                'train_acc': np.mean(train_accuracy_all),
                'val_loss': np.mean(val_loss_all),
                'val_acc': np.mean(val_accuracy_all)
            }

            print(epoch_result)
            result.append(epoch_result)

        # save checkpoints
        # checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        # state = {
        #     'epoch': epoch,
        #     'total_steps': total_steps,
        #     'optimizer': optimizer.state_dict(),
        #     'model': alexnet.state_dict(),
        #     'seed': seed,
        # }
        # torch.save(state, checkpoint_path)

    with open('C:/Users/klouc/Desktop/cogrob/alexnet-pytorch/alex_result.json', "w") as json_file:
        json.dump(result, json_file)
