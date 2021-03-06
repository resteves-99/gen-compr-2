__author__ = 'RafaelEsteves,SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from models.ae_experimental import experimental_autoencoder
from models.ae_short import small_autoencoder
from models.ae_baseline import baseline_autoencoder
import os


def to_img(x):
    x = 0.5 * (x + 1) #?
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 216, 178)
    return x


def train(args, train_dataloader, val_dataloader, loss_function, log, example_dir):
    #init model
    model = baseline_autoencoder(args)
    if args.type == 'ae_exp':
        model = experimental_autoencoder(args)
    elif args.type == 'ae_small':
        model = small_autoencoder(args)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #weight_decay

    #train
    for epoch in range(args.num_epochs):
        log.info(f'Epoch: {epoch}')
        model.train() # maybe move into for loop
        train_loss = 0
        total_est = 50000000 #figure this out
        with torch.enable_grad(), tqdm(total=total_est) as progress_bar:
            for batch_idx, data in enumerate(train_dataloader):
                # process batch
                img, _ = data
                img.resize_((16,3,218,178))
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                optimizer.zero_grad()

                #run model and train
                recon_batch = model(img)
                img.resize_((16,3,216,178))
                # recon_batch.resize_((16, 3, 218, 178))
                loss = loss_function(recon_batch, img)
                loss.backward()
                train_loss += loss
                optimizer.step()

                #logging
                input_ids = batch_idx #batch['input_ids'].to('cuda')
                progress_bar.update(input_ids)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item())

                #TODO: update logging

        #end of epcoh logging
        log.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))
        if epoch % 5 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, example_dir + f'/image_epcoh_{epoch}.png')

    #finished training
    return model