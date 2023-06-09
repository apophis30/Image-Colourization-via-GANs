# Image-Colourization-via-GANs

![](https://i.imgur.com/jcIQBO3.jpg)

## Motivation
The task of colourizing black and white photographs necessitates a lot of human input and hardcoding. Online tools can be used for image colorization, but they are very time consuming. The goal is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black and white image as input and producing a colourized image as output.

## Solution Approach
- The colorization of grayscale images can be thought of as an image-to-image translation task where we have the corresponding labels for the input grayscale image. A **conditional GAN (CGAN)** conditioned on grayscale images can be used to generate the corresponding colorized images.


- The architecture of the model consists of a conditional generator with grayscale image inputs and a random noise vector and the output of the generator are two image channels a, b in the *LAB image space* to be concatenated with the L channel i.e. the grayscale input image.
### Why LAB colorspace?
When we load a colored image as RGB, we get three channels corresponding to different colors (Red, Blue & Green).
![](https://i.imgur.com/6ME9L79.png)
But in LAB colorspace, the L stands for Lightness, which is the black and white image. *A* and *B* channel tell how much green-red and yellow-blue each pixel is respectively. 
![](https://i.imgur.com/ZmyTggA.png)
When using *Lab*, we can give the L channel to the model (which is the grayscale image) and want it to predict the other two channels (*a, *b) and after its prediction, we concatenate all the channels and we get our colorful image.
But if you use RGB, you have to first convert your image to grayscale, feed the grayscale image to the model and hope it will predict 3 numbers for you which is a way more difficult and unstable task due to the many more possible combinations of 3 numbers compared to two numbers.

### Dataset 
We use the **COCO (Common Object in Context) dataset** for training as it contains many different scenes and locations. So, our model will be trained on a variety of images. We use 8000 images for training the model and 2000 for evaluation.

### Losses
- Consider x as the grayscale image, z as the input noise for the generator, and y as the 2-channel output we want from the generator (it can also represent the 2 color channels of a real image). Also, G is the generator model and D is the discriminator. Then the loss for our conditional GAN will be:
![](https://i.imgur.com/2j2g6tq.png)

- We combine this loss function with L1 Loss of the predicted colors compared with the actual colors:
![](https://i.imgur.com/cnBoNNG.png)

- So, our combined loss function will be:
![](https://i.imgur.com/6bkwrfI.png)
where λ is a coefficient to balance the contribution of the two losses to the final loss. 

### Generator Network
- For the generator network, a **U-Net Architecture** is used which downsamples and then upsamples the input image. In the upsampling block, the corresponding output of the downsampling layer is also concatinated.
- ![](https://i.imgur.com/kqdeNkK.png)
 
### Discriminator Network 
- The generated image is input to the **PatchGAN discriminator** which outputs a score for each patch in an input image based on if the patch is real or not. These are used as learning signals for the Generator to generate better images.
- The discriminator is made by stacking blocks of *Conv-BatchNorm-LeackyReLU* to decide whether the input image is fake or real.

### Finally.. 
- When trained adversarially, the generator should get better at generating realistic colorized images that share a similar structure with the input grayscale images and the discriminator should get better at discriminating between real and fake images.
- The trained generator can then be used for generating colorized images given input grayscale images.

## Results
After 50 epochs, these were the results obtained evaluated on validation dataset:
![](https://i.imgur.com/27RwT4J.png)

Our model did a good job in performing the task. But, we can still see some bad colorization done on the first image of bananas. We can also see some bluish color in the second image, which makes it look fake.

### Plotting losses

![](https://i.imgur.com/ga6x45b.png)

We can see the loss of real and fake images converging towards each other, which implies that the generator is getting better at producing real images.

![](https://i.imgur.com/drlOUVK.png)

The generator loss is quickly decreasing with the number of epochs and will further reduce if trained on more epochs.

![](https://i.imgur.com/AQgtaVM.png)

A comparison between generator and discriminator loss.

### Changing strategy
Initially, neither generator nor discriminator knows anything about the task at the beginning of training. To avoid the problem of **the blind leading the blind** in the GAN game we introduce a bit of pre-training in our generator.

We will use a pretrained ResNet18 as the backbone of the U-Net and to accomplish the second stage of pretraining, we are going to train the U-Net on our training set with only L1 Loss. Then we will move to the combined adversarial and L1 loss, as we did in the previous section.

![](https://i.imgur.com/fe7KCPN.png)

![](https://i.imgur.com/7AROPTn.png)


## Conclusion 
We accomplished the task of colorizing image. We can see that the model is able to produce realistic images. By using some pretraining, we improved the performance of our model.
