# Pneumonia Detection from Chest X-rays

This project implements a Deep Learning model to detect pneumonia from chest X-rays. The dataset used in this project is the [ChestX-ray8](https://arxiv.org/abs/1705.02315) dataset which contains 112,120 chest X-ray images of 30,805 unique patients. The model architecture is the [CheXNet](https://arxiv.org/abs/1711.05225) model which is implimented using Pytorch.

There are three notebooks.
- `01. Pneumonia Data Preprocessing.ipynb` explores the dataset's characteristics and splits the dataset into the train, test and validation dataset.
- `02. Processing Chest X-ray Images.ipynb` downloads the images in a convenient way and visualizes some of the X-ray images
- `03. Training and Evaluating Model Using Sagemaker.ipynb` trains and evaluates the model on AWS Sagemaker platform.

The model is trained only for 5 epochs which is not enough to get a good performance out of this model.

## Requirment
The `03. Training and Evaluating Model Using Sagemaker.ipynb` notebook needs to be run on AWS Sagemaker platform.