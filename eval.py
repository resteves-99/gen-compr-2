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

def main():
    args = get_eval_args()

    # set up logger
    log = utils.get_logger(args.load_dir, 'log_test')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Testing Data...")

    #load data
    img_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = CelebA('./data', split = 'test', transform=img_transform, download=True, target_type=None)
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
    if args.load_dir is None:
        log.info("--load-dir must be provided")
    gen_dir = os.path.join(args.load_dir, 'gen_model.pt')
    gen_model.load_state_dict(torch.load(gen_dir))
    disc_dir = os.path.join(args.load_dir, 'disc_model.pt')
    disc_model.load_state_dict(torch.load(disc_dir))
    log.info("Testing model in ", args.load_dir)

    if torch.cuda.is_available():
        gen_model.cuda()
        disc_model.cuda()

        # set up loss
        criterion = None
        if args.type in ['ae_base', 'ae_exp', 'ae_small']:
            criterion = nn.BCELoss()

    #run testing
    with torch.enable_grad(), tqdm(total=51750051) as progress_bar:
        torch.autograd.set_detect_anomaly(True)
        total_g_loss = 0
        total_d_loss = 0
        index = 0
        for batch_idx, data in enumerate(test_dataloader):
            # process batch
            real_image, _ = data
            real_image.resize_((args.batch_size, 3, 218, 178))
            # real_image = Variable(real_image)
            if torch.cuda.is_available():
                real_image = real_image.cuda()

            # calc discriminator loss and train
            fake_loss, real_loss = disc_loss(args, gen_model, disc_model, criterion, real_image)
            d_loss = fake_loss + real_loss

            # calc gen loss and train
            g_loss, recon_batch = gen_loss(args, gen_model, disc_model, criterion, real_image)

            # logging
            input_ids = batch_idx  # batch['input_ids'].to('cuda')
            progress_bar.update(index)
            progress_bar.set_postfix(epoch=1, Gen_Loss=float(g_loss),
                                     Disc_Loss=(float(fake_loss), float(real_loss)))
            total_g_loss += float(g_loss)
            total_d_loss += float(d_loss)
            index += 1

if __name__ == '__main__':
    main()