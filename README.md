# MWF_prediction
This repository intends to assess the feasibility of predicting a MWF MRI image from T1 &amp; T2 images

## Directories
T1, T2 and MWF directories intend to contain the images the deep learning algorithm will be feed with.
The shell scripts allow to preprocess the images to standardise their size, orientation and brain tissues alignment.

## display.py
This library describe several functions for displaying samples from the data loaders but also the training loss or examples of the CNN output.

## main.py
This contains the main data loaders and the training function to make the CNN learning.
It allows to train the CNN as long as save it and display some results.

## Prediction_only.py
This file only allows to import a CNN with the same structure. You can then display your results.
