# Deep learning framework for image segmentation -- Unet, using Keras

## Overview

### Data
The confocal images were collected from the whole-mount tissue blocks by confocal laser-scanning microscopy and the gold standard images are black and white images where white pixels were denoted as ICC networks(target), and black pixels were denoted as background. You can find it in folder data/membrane.

### Data augmentation
The data for training contains 17 100*100 images and data augmentation involved elastic deformation, flip, rotation and crop with determinate parameters.

### U-net Model
This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures. The U-net model used in this task has been modifed from the standard u-net model developed in 2015. I added a pair of layers in the decoding and encoding path. I also applied batch normalisation for my model. The model after modification achieved better performance compared to the standard u-net model.

### Hyperparameter Tuning
The model can be tuned in the validation process to optimise the hyperparameters, including number of filters in the first layer, batch size, learning rate and dropout-rate.  The hyperparameter tuning method is a combination of nested cross validation and bayesian optimisation which is implemented in the Python library Hyperopt.

### Validation
Cross-validation -- cross-validation.py
Different validation metrics -- acc.py 



