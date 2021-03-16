import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CelebA, MNIST

import numpy as np
from facenet_pytorch import MTCNN
import sys
import os
from args import get_train_args
import utils
import json
import trainers.conv_autoencoder as conv_autoencoder


#set up loss
bce = nn.BCELoss()
mse = nn.MSELoss()
mt = None
to_pil = transforms.ToPILImage()
def loss(pred = None, label = None, real = None, recon = None, face_seg = False):
    global mt
    loss = 0
    if pred is not None and label is not None:
        loss += bce(pred, label)
    if real is not None and recon is not None:
        if face_seg:
            #loss on faces only
            if mt is None:
                mt = MTCNN(image_size = list(real[0,:,:,:].squeeze().shape)[0])
            height, width = list(real[0,0,:,:].squeeze().shape)
            for batch_idx in range(real.shape[0]):
                real_img = real[batch_idx, :, :, :].squeeze() #get one image
                real_img_cpy = real_img
                real_img_cpy = to_pil(real_img_cpy)
                face_tensor, prob = mt.detect(real_img_cpy) #detect faces
                if face_tensor is not None: #if face found make ints, use only most likely face
                    face_tensor = np.squeeze(face_tensor.astype(int)[0])
                else:#if face not found use entire picture
                    face_tensor = [0, 0, width, height]
                real_crop = real_img[:, max(face_tensor[0]-25,0):min(face_tensor[3]+25, height), max(face_tensor[1]-25, 0): min(face_tensor[2]+25,width)]

                recon_img = recon[batch_idx, :, :, :].squeeze()
                #use the same face coordinates for the reconstructed image
                recon_crop = recon_img[:, max(face_tensor[0]-25,0):min(face_tensor[3]+25, height), max(face_tensor[1]-25, 0): min(face_tensor[2]+25,width)]

                loss += mse(recon_crop, real_crop)
        else:
            loss += mse(recon, real)
    return loss

def main():
    args = get_train_args()

    #set up save directory
    save_dir = os.path.join('./save', args.name)
    log_dir = os.path.join(save_dir, 'logging')
    example_dir = os.path.join(save_dir, 'examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(log_dir)
        os.makedirs(example_dir)

    #set up logger
    log = utils.get_logger(save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    # args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #set up data loader
    img_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    download = True
    train_dataset = CelebA('./data', split = 'train', transform=img_transform, download=download)
    val_dataset = CelebA('./data', split = 'valid', transform=img_transform, download=download)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    log.info("Data Loaded")
    # I do not use the validation data loader yet but I might implement this in training later


    #do training
    gen_model = None
    disc_model = None
    if args.type in ['ae_base', 'ae_exp', 'ae_small']:
        gen_model, disc_model = conv_autoencoder.train(args, train_dataloader, val_dataloader, loss, log, example_dir, save_dir)

    #save models
    torch.save(gen_model.state_dict(), save_dir+'/gen_model.pt')
    torch.save(disc_model.state_dict(), save_dir+'/disc_model.pt')

if __name__ == '__main__':
    main()
