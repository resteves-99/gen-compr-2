__author__ = 'RafaelEsteves,SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from models.ae_experimental import experimental_autoencoder, experimental_discriminator
from models.ae_short import small_autoencoder, small_discriminator
from models.ae_baseline import baseline_autoencoder, baseline_discriminator
import os
import copy
import matplotlib.pyplot as plt


def disc_loss(args, gen, disc, criterion, real_image):
    real_label = torch.ones(args.batch_size, device='cuda')
    fake_label = torch.zeros(args.batch_size, device='cuda')
    real_images = copy.copy(real_image)

    #get reconstructed images
    fake_images = gen(real_images).detach()

    #reshape target so it is same size as reconstruction
    tgt_size = [args.batch_size]
    for elem in list(fake_images.shape[1:]): tgt_size.append(elem)
    real_images = real_images[:,:,:fake_images.shape[2],:fake_images.shape[3]]
    assert real_images.shape == fake_images.shape

    #predict for both real and reconstructed images
    fake_preds = disc(fake_images)
    real_preds = disc(real_images)

    #compare predictions with target label
    fake_loss = criterion(fake_preds, fake_label)
    real_loss = criterion(real_preds, real_label)
    disc_loss = 0.5*(fake_loss + real_loss)

    return fake_loss, real_loss

def gen_loss(args, gen, disc, criterion, real_image):
    real_label = torch.ones(args.batch_size).cuda()
    real_images = copy.copy(real_image)

    #generate fake images and make predictions on them
    fake_images = gen(real_images)
    fake_preds = disc(fake_images)

    #compare predictions with target label
    gen_loss = criterion(fake_preds, real_label)

    return gen_loss, fake_images


def train(args, train_dataloader, val_dataloader, criterion, log, example_dir, save_dir):
    #init model based on type of model
    gen_model = baseline_autoencoder(args)
    disc_model = baseline_discriminator(args)
    if args.type == 'ae_exp':
        gen_model = experimental_autoencoder(args)
        disc_model = experimental_discriminator(args)
    elif args.type == 'ae_small':
        gen_model = small_autoencoder(args)
        disc_model = small_discriminator(args)
    if args.load_dir is not None:
        #this wont work yet because of naming issues
        gen_dir = os.path.join(args.load_dir, 'gen_model.pt')
        gen_model.load_state_dict(torch.load(gen_dir))
        disc_dir = os.path.join(args.load_dir, 'disc_model.pt')
        disc_model.load_state_dict(torch.load(disc_dir))
        log.info("Continuing training based on ", args.load_dir)

    if torch.cuda.is_available():
        gen_model.cuda()
        disc_model.cuda()

    #init optimizers
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr) #weight_decay
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)

    #for every epoch train
    for epoch in range(args.num_epochs):
        log.info(f'Epoch: {epoch}')
        disc_model.train()
        train_loss = (0,0)
        index = 0
        with torch.enable_grad(), tqdm(total=51750051) as progress_bar:
            torch.autograd.set_detect_anomaly(True)
            for batch_idx, data in enumerate(train_dataloader):
                # process batch
                real_image, _ = data

                real_image.resize_((args.batch_size,3,218,178))
                if torch.cuda.is_available():
                    real_image = real_image.cuda()

                #calc discriminator loss and train
                fake_loss, real_loss = disc_loss(args, gen_model, disc_model, criterion, real_image)
                d_loss = fake_loss + real_loss
                if index%5 == 0:
                    disc_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)
                    disc_optimizer.step()

                #calc gen loss and train
                g_loss, recon_batch = gen_loss(args, gen_model, disc_model, criterion, real_image)
                gen_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                gen_optimizer.step()

                #logging, update progress bar and epoch loss
                input_ids = batch_idx #batch['input_ids'].to('cuda')
                progress_bar.update(index)
                progress_bar.set_postfix(epoch=epoch, Gen_Loss=float(g_loss), Disc_Loss=(float(fake_loss), float(real_loss)))
                train_loss += (float(g_loss), float(d_loss))
                index += 1


        #end of epcoh logging. average losses, save image examples, save model
        log.info('====> Epoch: {} Average gen loss: {:.4f} Average Dis Loss: {:.4f}'.format(
            epoch, train_loss[0] / len(train_dataloader.dataset), train_loss[1] / len(train_dataloader.dataset)))
        if epoch % 2 == 0:
            save = recon_batch.cpu().data
            save_image(save, example_dir + f'/recon_image_epcoh_{epoch}.png')

            save = real_image.cpu().data
            save_image(save, example_dir + f'/real_image_epcoh_{epoch}.png')
        if epoch % 10 == 0:
            torch.save(gen_model.state_dict(), save_dir + '/gen_model.pt')
            torch.save(disc_model.state_dict(), save_dir + '/disc_model.pt')


    #finished training. return trained models
    return gen_model, disc_model
