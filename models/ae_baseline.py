import torch.nn as nn
import torch

class baseline_autoencoder(nn.Module):
    def __init__(self, args):
        super(baseline_autoencoder, self).__init__()

        self.args = args

        self.enc_layer_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=1),  # b, 16, 72, 59
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, ...
        )
        self.enc_layer_2_def = nn.Sequential(
            nn.Conv2d(16, 12, kernel_size=3, stride=2),# padding=1), # b, 12, 35, 29
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
        embed = (self.enc_layer_2_def(out_1))
        return embed

    def decoder(self, x):
        out = x
        out = self.dec_layer_2(out)
        recon_x = self.dec_layer_3(out)
        return recon_x

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

class baseline_discriminator(nn.Module):
    def __init__(self, args):
        super(baseline_discriminator, self).__init__()
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
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()