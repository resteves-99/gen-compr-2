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
sys.path.insert(0, '08-AutoEncoder')
sys.path.insert(0, '09-Generative Adversarial network')
import os
from args import get_train_test_args
import utils
import json
import vae
import conv_autoencoder



def vae_loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """

    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def main():
    args = get_train_test_args()

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

    # if os.path.exists('./data/celeba'):
    #     download = False
    if args.mnist:
        train_dataset = MNIST('./data', transform=img_transform, download=download)
    else:
        train_dataset = CelebA('./data', split = 'train', transform=img_transform, download=download)
        val_dataset = CelebA('./data', split = 'valid', transform=img_transform, download=download)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    log.info("Data Loaded")


    #set up loss
    loss = None
    if args.type in ['ae_base', 'ae_exp', 'ae_small']:
        loss = nn.MSELoss()

    #do training
    model = None
    if args.type in ['ae_base', 'ae_exp', 'ae_small']:
        model = conv_autoencoder.train(args, train_dataloader, val_dataloader, loss, log, example_dir)

    torch.save(model.state_dict(), save_dir+'/model_'+args.name+'.pt')

if __name__ == '__main__':
    main()
