import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CelebA
import os
import utils
import json
from trainers.conv_autoencoder import  disc_loss, gen_loss
from args import get_eval_args
from tqdm import tqdm
from models.ae_experimental import experimental_autoencoder, experimental_discriminator
from models.ae_short import small_autoencoder, small_discriminator
from models.ae_baseline import baseline_autoencoder, baseline_discriminator
from train import loss
import numpy as np


def main():
    args = get_eval_args()

    #set up directories

    save_dir = os.path.join('./save', args.name)
    log_dir = os.path.join(save_dir, 'logging')
    example_dir = os.path.join(save_dir, 'examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(log_dir)
        os.makedirs(example_dir)

    # set up logger
    log = utils.get_logger(save_dir, 'log_test')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Testing Data...")

    #load data
    img_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = CelebA('./data', split = args.split, transform=img_transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    log.info("Data Loaded")

    #load model
    if args.type == 'ae_base':
        gen_model = baseline_autoencoder(args)
        disc_model = baseline_discriminator(args)
    elif args.type == 'ae_exp':
        gen_model = experimental_autoencoder(args)
        disc_model = experimental_discriminator(args)
    elif args.type == 'ae_small':
        gen_model = small_autoencoder(args)
        disc_model = small_discriminator(args)
    gen_dir = os.path.join(save_dir, 'gen_model.pt')
    gen_model.load_state_dict(torch.load(gen_dir))
    disc_dir = os.path.join(save_dir, 'disc_model.pt')
    disc_model.load_state_dict(torch.load(disc_dir))

    # log.info("Testing model in ", args.load_dir)

    if torch.cuda.is_available():
        gen_model.cuda()
        disc_model.cuda()

    #Loop over the test set
    with torch.enable_grad(), tqdm(total=15*args.batch_size) as progress_bar:
        torch.autograd.set_detect_anomaly(True)
        total_g_loss = np.array([0.0,0.0])
        total_d_loss = np.array([0.0,0.0])
        index = 0
        for batch_idx, data in enumerate(test_dataloader):
            # process batch
            real_image, _ = data
            real_image.resize_((args.batch_size, 3, 218, 178))
            if torch.cuda.is_available():
                real_image = real_image.cuda()

            # calc discriminator loss and train
            fake_loss, real_loss = disc_loss(args, gen_model, disc_model, loss, real_image)

            # calc gen loss and train
            pred_loss, mse_loss, recon_batch = gen_loss(args, gen_model, disc_model, loss, real_image, face_seg = True)

            input_ids = batch_idx  # batch['input_ids'].to('cuda')
            progress_bar.update(index)
            progress_bar.set_postfix(epoch=1)
            total_g_loss += (float(pred_loss),float(mse_loss))
            total_d_loss += (float(fake_loss),float(real_loss))

            if index == 30:
                break
            index += 1

        avg_pred_loss = total_g_loss[0]/index
        avg_mse_loss = total_g_loss[1]/index
        avg_fake_loss = total_d_loss[0]/index
        avg_real_loss = total_d_loss[1]/index

        log.info(f"Across one test batch the generator loss had average discriminator loss  {avg_pred_loss} and faciel MSE loss  {avg_mse_loss}\n"
                 f"The discriminator had average reconstructed image loss {avg_fake_loss} and average real image  loss {avg_real_loss}")

        #save the reconstructed example and the real example
        save = recon_batch.cpu().data
        save_image(save, example_dir + f'/recon_image_test.png')

        save = real_image.cpu().data
        save_image(save, example_dir + f'/real_image_test.png')

if __name__ == '__main__':
    main()