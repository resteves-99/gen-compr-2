import torch.nn as nn
import torch

class experimental_autoencoder(nn.Module):
    def __init__(self, args):
        super(experimental_autoencoder, self).__init__()

        #TODO: kernel size
        #TODO: stride
        #TODO: bpp
        #TODO: add attention


        self.args = args

        #initialize encoding layers
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
        #construct both a large and small embedding

        #initialize decoding layers
        self.dec_layer_1 = nn.Sequential(
            #only use this on smallest output
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),# padding=1),# output_padding=1),  # b, 16, 36, 30
            nn.ReLU(True),
        )
        self.dec_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(12, 8, kernel_size=5, stride=2, padding=1), # padding=1),# output_padding=(1,0)),  # b, 8, 73, 61
            nn.ReLU(True),
        )
        self.dec_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=3, padding=(2,4)),  # b, 3, 217, 179
            nn.Tanh()
        )

    def encoder(self, x):
        out_1 = self.enc_layer_1(x)
        out_2 = self.enc_layer_2(out_1)
        out_3 = self.enc_layer_3(out_2)
        embed = (out_2, out_3)
        # print(embed.shape)
        return embed

    def decoder(self, x):
        out_2, out_3 = x
        #take smaller encoding and make it same size as bigger encoding
        out_3 = self.dec_layer_1(out_3)
        #join the two encodings along the channel dimension
        out = torch.cat((out_2, out_3), dim=1)
        out = self.dec_layer_2(out)
        recon_x = self.dec_layer_3(out)
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
        #flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()