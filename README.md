# installation
run the following comands in a new conda environment. 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pandas tqdm requests matplotlib
pip install facenet-pytorch
```

# train
to train an adversarial autoencoder model use the following command
``` 
python train.py --name model_name
```
The other possible options for training are given below
```
--batch-size number
--num-epochs number
--lr number
--type [ae_base, ae_exp, ae_small] default ae_base
```

# test
to see an example output from the test set run and to get face reconstruction loss use
```
python eval.py --name model_name --load-dir model_dir
```


# models
ae_base is a baseline model which embeds images into a larger size than our experimental model.
ae_exp is our experimental model.\
ae_small is a baseline model which embeds images into a smaller size than our experimental mode.


All of these models are composed of an encoder, a decoder, and a discriminator.\
The differences between these models is further specified in my writeup.

You can run my model with upsampling as opposed to tranpose convolution by switching to the checkerboard fix branch.
You can run my model with an added attention layer by switching to the attention branch

# evalutation
You can find the test log which includes test loss as well as example reconstruction images under ```./save/model_name/examples/```
The first iteration of models is the ```[large_1, exp_1, small_1]```
The improved base configuration group is ```[updated_model_large, updated_model_exp, updated_model_small]```
The models with attention implemented are ```[large_attn, exp_attn, small_attn]```
Finally, the models with upsampling are ```[large_upsample, exp_upsample, small_upsample]```e

# old readme

## AutoEncoder and VAE

![autoencoder.png](http://upload-images.jianshu.io/upload_images/3623720-5e46977d7f8905f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

This is the principal of autoencoder.

- - -
#### Simple multilayer perceptron as encoder and decoder we can get the result as follow:

![auto0.png](http://upload-images.jianshu.io/upload_images/3623720-8609665d5484ca28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![auto1.png](http://upload-images.jianshu.io/upload_images/3623720-55b3cba386f1e3a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The encoder output is actual like this one

![auto4.png](http://upload-images.jianshu.io/upload_images/3623720-bdb6aa9b7e99ba4a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### Convolution autoencoder result:

![image_0.png](http://upload-images.jianshu.io/upload_images/3623720-1ab9ed4ec16f4a26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image_20.png](http://upload-images.jianshu.io/upload_images/3623720-95c793a566cf287c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image_90.png](http://upload-images.jianshu.io/upload_images/3623720-20d520008d5722f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### Variational autoencoder result:

![image_0.png](http://upload-images.jianshu.io/upload_images/3623720-eed315cf84c0b879.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image_10.png](http://upload-images.jianshu.io/upload_images/3623720-b6fe5bfbbf6a924d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image_90.png](http://upload-images.jianshu.io/upload_images/3623720-121c44fb64674f09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
