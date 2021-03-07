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
from models.ae_baseline import baseline_autoencoder, baseline_discriminator
import os


def to_img(x):
    x = 0.5 * (x + 1) #?
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 216, 178)
    return x


def train(args, train_dataloader, val_dataloader, loss_function, log, example_dir):
    #init model
    gen_model = baseline_autoencoder(args)
    disc_model = baseline_discriminator(args)
    if args.type == 'ae_exp':
        model = experimental_autoencoder(args)
    elif args.type == 'ae_small':
        model = small_autoencoder(args)
    if torch.cuda.is_available():
        gen_model.cuda()
        disc_model.cuda()

    disc_optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr) #weight_decay
    gen_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr)
    real_label = Variable(torch.ones(args.batch_size)).cuda()
    fake_label = Variable(torch.zeros(args.batch_size)).cuda()

    #train
    for epoch in range(args.num_epochs):
        log.info(f'Epoch: {epoch}')
        gen_model.train() # maybe move into for loop
        disc_model.train
        train_loss = (0,0)
        index = 0
        with torch.enable_grad(), tqdm(total=152000) as progress_bar:
            torch.autograd.set_detect_anomaly(True)
            for batch_idx, data in enumerate(train_dataloader):
                # process batch
                img, _ = data
                img.resize_((args.batch_size,3,218,178))
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()

                #run generator
                recon_batch = gen_model(img)

                #process batch again
                #TODO: all models must output same size?

                recon_batch = Variable(recon_batch)
                img.resize_((16,3,216,178))
                # img = Variable(img)
                # recon_batch.resize_((16, 3, 218, 178))

                #run discriminator
                real_output = disc_model(img)
                fake_output = disc_model(recon_batch)

                #calc discriminator loss and train
                d_loss_fake = loss_function(fake_output, fake_label)
                d_loss_real = loss_function(real_output, real_label)
                d_loss = d_loss_real + d_loss_fake
                if index%5 == 0:
                    disc_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)
                    disc_optimizer.step()
                    # fake_output = disc_model(recon_batch)

                #calc gen loss and train
                g_loss = loss_function(fake_output, real_label)
                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()


                #logging
                input_ids = batch_idx #batch['input_ids'].to('cuda')
                progress_bar.update(index)
                progress_bar.set_postfix(epoch=epoch, Gen_Loss=float(g_loss), Disc_Loss=float(d_loss))
                train_loss += (float(g_loss), float(d_loss))
                index += 1

                #TODO: update logging

        #end of epcoh logging
        log.info('====> Epoch: {} Average gen loss: {:.4f} Average Dis Loss: {:.4f}'.format(
            epoch, train_loss[0] / len(train_dataloader.dataset), train_loss[1] / len(train_dataloader.dataset)))
        if epoch % 5 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, example_dir + f'/recon_image_epcoh_{epoch}.png')

            save = to_img(img.cpu().data)
            save_image(save, example_dir + f'/real_image_epcoh_{epoch}.png')


    #finished training
    return gen_model, disc_model
