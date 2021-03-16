import torch.nn as nn
import torch
from gsa_pytorch import GSA

class base_autoencoder(nn.Module):
    def __init__(self, args):
        super(base_autoencoder, self).__init__()

        self.args = args

        #initialize encoding layers
        self.enc_layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=1),  # b, 16, 72, 59
            nn.ReLU(True),
        )
        self.enc_layer_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),# padding=1), # b, 16, 35, 29
        )
        self.enc_layer_large = nn.Sequential(
            nn.Conv2d(16, 6, kernel_size=3, stride=1),# padding=1), # b, 12, 35, 29
        )
        #encode only to big encoding


        #initialize decoding layers
        self.dec_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(6, 8, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)), # padding=1),# output_padding=(1,0)),  # b, 8, 73, 61
            nn.ReLU(True),
        )
        self.dec_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=3, padding=(1,2)),  # b, 3, 219, 185
            nn.Tanh()
        )

    def encoder(self, x):
        out = self.enc_layer_1(x)
        out = self.enc_layer_2(out)
        embed_large = self.enc_layer_large(out)
        embed = embed_large
        return embed

    def decoder(self, x):
        out = x
        out = self.dec_layer_2(out)
        recon_x = self.dec_layer_3(out)
        return recon_x

    def forward(self, x):
        #encode then reconstruct
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

class base_discriminator(nn.Module):
    def __init__(self, args):
        super(base_discriminator, self).__init__()

        #construct discriminator
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # batch, 32, 218, 178
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # batch, 32, 108, 89
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),  # batch, 32, 108, 89
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=2, stride=2)  # batch, 64, 54, 44
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # batch, 64, 27, 22
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=5, stride=3),  # batch, 64, 8, 6
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, channel, height, width
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()