Image classification with PyTorch

This project provides a framework for training a image classification model using PyTorch. 
The code is designed to use a dataset of Mississippi State University building.

Functions

get_data_transforms()
Defines data augmentation and normalization transformations for the training and validation datasets.

load_data(data_dir, batch_size=32)
Loads the dataset from the specified directory, applying transformations and creating data loaders for both training and validation data.

create_model(num_classes)
Loads a pre-trained ResNet18 model and adjusts the final fully connected layer to match the specified number of classes.

train_model(model, dataloaders, criterion, optimizer, num_epochs=10, dataset_sizes=None)
Trains the model for a specified number of epochs, computing and displaying the loss and accuracy for each epoch. The model’s best weights are saved based on validation accuracy.

evaluate_model(model, dataloader, criterion, class_names)
Evaluates the model on the validation set, printing metrics such as accuracy, precision, recall, F1 score, and log loss.

Training 

The model is trained for 10 epochs with data augmentation and normalized inputs. 

Dependencies 

Python3
,PyTorch
,Torchvision
,gdown 
