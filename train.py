import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CelebA, MNIST

import sys
import os
from args import get_train_args
import utils
import json
import trainers.conv_autoencoder as conv_autoencoder



def main():
    args = get_train_args()

    #set up save directory
    save_dir = os.path.join('./save', args.name)
    log_dir = os.path.join(save_dir, 'logging')
    example_dir = os.path.join(save_dir, 'examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(log_dir)
        os.makedirs(example_dir)

    #set up logger
    log = utils.get_logger(save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    # args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #set up data loader
    img_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    download = True
    train_dataset = CelebA('./data', split = 'train', transform=img_transform, download=download)
    val_dataset = CelebA('./data', split = 'valid', transform=img_transform, download=download)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    log.info("Data Loaded")
    # I do not use the validation data loader yet but I might implement this in training later


    #set up loss
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    def loss(pred = None, label = None, real = None, recon = None):
        loss = 0
        if pred is not None and label is not None:
            loss += bce(pred, label)
        if real is not None and recon is not None:
            loss += mse(recon, real)
        return loss


    #do training
    gen_model = None
    disc_model = None
    if args.type in ['ae_base', 'ae_exp', 'ae_small']:
        gen_model, disc_model = conv_autoencoder.train(args, train_dataloader, val_dataloader, loss, log, example_dir, save_dir)

    #save models
    #TODO: change this
    torch.save(gen_model.state_dict(), save_dir+'/gen_model.pt')
    torch.save(disc_model.state_dict(), save_dir+'/disc_model.pt')

if __name__ == '__main__':
    main()
