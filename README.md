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
o	main_augmentation.py << used to train model (split into training, validation and testing sets)
o	data_1.py << custom DataGenerator for image pre-processing 
o	acc << used to calculate five evaluation metrics based on the testing set.
o	model_1.py << U-net model
o	model_3.py << extended U-net model
o	train_fulldata.py << use all data to train the model (split into training and validation sets).
o	boundary.py << use to overlap one image to its predicted image
o	error_map.py << generate error map between a prediction and the corresponding gold-standard image.
o	Tuning_good_split.py << hyperparameter tuning.
o	main_tile.py << use overlap_tile segmentation method to segment the whole-mount tissue images. The whole-mount tissue images named “6465antrum_transprox_m27_c2.tif” can be could under the folder “raw-dataset”.
o	cross-validation.py << used to generate cross-validation results.
o	test_full_data.hdf5 << final framework (trained model)
o	data << this folders contained inputs and outputs for the model.




