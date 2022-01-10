# Vanilla implementation of U-Net  

derived from: [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)


### Details of the original architecture
The architecture looks as follows:
- **encoder block**: In this block the image is convolved and meanwhile slightly reduced in size. The first convolution doubles the number of feature channels
  - Convolution with 3x3 kernel and step of 1
  - ReLU
  - Convolution as above again
  - ReLU
- **down-block**: this block reduces the imagesize in half by using a MaxPooling of size 2x2 with step 2
- **"up convolution"**: Often a Transposed Convolution is used here. Ronneberger et al. describe that the use
  - an upsampling of factor 2 to increase the resolution
  - a "2x2" convolution with onesided padding which amounts to a same size image and is used to half the number of feature maps
- **decoder block**: very similar to the encoder block, except that the cropped output of the mirror image encoder block is being preprended to the signal along the feature dimension, effectively doubling the feature dim again after a up convolution, which is again halved by the first convolution of the decoder block.
- **output convolution**: this reduces the final feature map/image to the number of targets to find in an image as each target has its own layer.

architecture image:
![u-net arch](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)