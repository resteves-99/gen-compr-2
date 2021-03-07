__author__ = 'RafaelEsteves,SherlockLiao'

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.forward1 = nn.Conv2d()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def train(args, train_dataloader, val_dataloader, loss_function, log, example_dir):
    model = VAE()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        log.info(f'Epoch: {epoch}')
        model.train() # maybe move into for loop
        train_loss = 0
        with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
            for batch_idx, data in enumerate(train_dataloader):
                img, _ = data
                img = img.view(img.size(0), -1)
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                optimizer.zero_grad()

                recon_batch, mu, logvar = model(img)
                loss = loss_function(recon_batch, img, mu, logvar)
                loss.backward()
                train_loss += loss.data[0]
                optimizer.step()

                input_ids = batch_idx #batch['input_ids'].to('cuda')
                progress_bar.update(len(input_ids))
                progress_bar.set_postfix(epoch=epoch, NLL=loss.item())

                if batch_idx % 100 == 0:
                    # log.info(f'Evaluating at step {batch_idx}...')
                    # # preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                    # # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    #
                    # # log.info('Visualizing in TensorBoard...')
                    # # for k, v in curr_score.items():
                    # #     tbx.add_scalar(f'val/{k}', v, batch_idx)
                    # log.info(f'Eval {results_str}')
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(img),
                        len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader),
                        loss.data[0] / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, example_dir + f'/image_{epoch}')


