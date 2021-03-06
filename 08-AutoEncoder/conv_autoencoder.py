__author__ = 'RafaelEsteves,SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os


def to_img(x):
    x = 0.5 * (x + 1) #?
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 216, 178)
    return x


class autoencoder(nn.Module):
    def __init__(self, args):
        super(autoencoder, self).__init__()

        #TODO: add skip connection, attention
        #TODO: add option for not progressive
        # can channels be different sizes
        self.args = args

        self.enc_layer_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=1),  # b, 16, 72, 59
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, ...
        )
        self.enc_layer_2 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1),  # b, 4, 36, 30
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, ...
        )
        self.enc_layer_3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), # b, 8, 18, 15
        )
        self.enc_layer_2_def = nn.Sequential(
            nn.Conv2d(16, 12, kernel_size=3, stride=2),# padding=1), # b, 12, 35, 29
        )

        self.dec_layer_1 = nn.Sequential(
            #only use this on smallest output
            nn.ConvTranspose2d(12, 12, kernel_size=3, stride=2, padding=1),# padding=1),# output_padding=1),  # b, 16, 35, 29
            nn.ReLU(True),
        )
        self.dec_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(12, 8, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)), # padding=1),# output_padding=(1,0)),  # b, 8, 73, 61
            nn.ReLU(True),
        )
        self.dec_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=3, padding=(1,2)),  # b, 3, 219, 185
            nn.Tanh()
        )

    def encoder(self, x):
        out_1 = self.enc_layer_1(x)
        # print(out_1.shape)
        if self.args.prog_grow == True:
            out_2 = self.enc_layer_2(out_1)
            out_3 = self.enc_layer_3(out_2)
            embed = (out_2, out_3)
        else:
            embed = (self.enc_layer_2_def(out_1))
        # print(embed.shape)
        return embed

    def decoder(self, x):
        if len(x) == 2:
            out_2, out_3 = x
            #error
            out_2 = self.dec_layer_1(out_2)
            out = torch.cat((out_2, out_3), dim=1)
        else:
            out = x
        # print(out.shape)
        out = self.dec_layer_2(out)
        # print(out.shape)
        recon_x = self.dec_layer_3(out)
        # print(recon_x.shape)
        return recon_x

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x


def train(args, train_dataloader, val_dataloader, loss_function, log, example_dir):
    model = autoencoder(args)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #weight_decay

    for epoch in range(args.num_epochs):
        log.info(f'Epoch: {epoch}')
        model.train() # maybe move into for loop
        train_loss = 0
        with torch.enable_grad(), tqdm(total=len(enumerate(train_dataloader.dataset))) as progress_bar:
            for batch_idx, data in enumerate(train_dataloader):
                # print('b', batch_idx)
                img, _ = data
                # print(img.shape)
                # img = img.view(img.size(0), -1)
                # print(img.shape)
                img.resize_((16,3,218,178))
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                optimizer.zero_grad()

                recon_batch = model(img)
                img.resize_((16,3,216,178))
                # recon_batch.resize_((16, 3, 218, 178))
                loss = loss_function(recon_batch, img)
                loss.backward()
                train_loss += loss
                optimizer.step()

                input_ids = batch_idx #batch['input_ids'].to('cuda')
                progress_bar.update(input_ids)
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item())

                #TODO: update logging
                # if batch_idx % 100 == 0:
                #     # log.info(f'Evaluating at step {batch_idx}...')
                #     # # preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                #     # # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                #     #
                #     # # log.info('Visualizing in TensorBoard...')
                #     # # for k, v in curr_score.items():
                #     # #     tbx.add_scalar(f'val/{k}', v, batch_idx)
                #     # log.info(f'Eval {results_str}')
                #     log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch,
                #         batch_idx * len(img),
                #         len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader),
                #         loss / len(img)))

        log.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))
        if epoch % 1 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, example_dir + f'/image_epcoh_{epoch}.png')


