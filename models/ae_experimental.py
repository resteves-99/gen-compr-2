import torch.nn as nn
import torch

class experimental_autoencoder(nn.Module):
    def __init__(self, args):
        super(experimental_autoencoder, self).__init__()


        self.args = args

        # initialize encoding layers
        self.enc_layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1),  # b, 32, 214, 174
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        )
        self.enc_layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # b, 64, 107, 87
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.enc_layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # b, 128, 54, 44
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        )
        self.enc_layer_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # b, 128, 27, 22
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        )
        self.enc_layer_large = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),  # b, 64, 27, 22
        )
        self.enc_layer_small = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=2),  # b, 64, 13, 10
        )
        # encode only t
        # encode only to small encoding

        # initialize decoding processing layers
        self.dec_layer_small = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=(0, 1)),  # b, 32, 27, 22
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.dec_layer_process_1 = nn.Sequential(
            # only use this on smallest output
            nn.InstanceNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),  # b, 128, 27, 22
            nn.InstanceNorm2d(128),
        )
        self.dec_layer_process_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # b, 128, 27, 22
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # b, 128, 27, 22
            nn.InstanceNorm2d(128),
        )
        # #plus from layer 1
        # self.dec_layer_process_3 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),  # b, 8, 72, 60
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),  # b, 8, 72, 60
        #     nn.InstanceNorm2d(128),
        # )

        # initialize decoding layers
        # plus from process layer 1
        self.dec_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # b, 64, 54, 44
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.dec_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # b, 32, 107, 87
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        )
        self.dec_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1),  # b, 3, 213, 173
        )

    def encoder(self, x):
        out = self.enc_layer_1(x)
        out = self.enc_layer_2(out)
        out = self.enc_layer_3(out)
        out = self.enc_layer_4(out)
        large_out = self.enc_layer_large(out)
        small_out = self.enc_layer_small(out)
        embed = (small_out, large_out)

        return embed

    def decoder(self, x):
        small_out, large_out = x
        out = self.dec_layer_small(small_out)
        out = torch.cat((large_out, out), dim=1)
        out_1 = self.dec_layer_process_1(out)
        out = self.dec_layer_process_2(out_1)
        out = out + out_1 #skip connection
        out = self.dec_layer_1(out)
        out = self.dec_layer_2(out)
        out = self.dec_layer_3(out)
        recon_x = out
        return recon_x

    def forward(self, x):
        #encode then reconstruct
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

class experimental_discriminator(nn.Module):
    def __init__(self, args):
        super(experimental_discriminator, self).__init__()

        #construct discriminator
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),  # batch, 32, 105, 85
            nn.LeakyReLU(0.2, True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # batch, 63, 51, 41
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # batch, 128, 24, 19
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=2),  # batch, 128, 11, 8
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*11*8, 1024),
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
        x = self.conv4(x)
        #flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()