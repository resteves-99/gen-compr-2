import torch.nn as nn
import torch

class small_autoencoder(nn.Module):
    def __init__(self, args):
        super(small_autoencoder, self).__init__()


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

        self.dec_layer_1 = nn.Sequential(
            #only use this on smallest output
            nn.ConvTranspose2d(8, 12, kernel_size=3, stride=2, padding=1),# padding=1),# output_padding=1),  # b, 16, 35, 29
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
        out_2 = self.enc_layer_2(out_1)
        out_3 = self.enc_layer_3(out_2)
        embed = out_3

        return embed

    def decoder(self, x):
        out = x
        out = self.dec_layer_1(out)
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