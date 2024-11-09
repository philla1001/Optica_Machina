Team Name: Chao Machina
Team Member 1 (leader): Phillip Crawford, pjc216@msstate.edu
Team Member 2: Pacey Whitt, jpw464@msstate.edu

APPROACH: The approach was to the resnet18 model with pytorch. Neither of us were familar with python or AI so we both used internet (youtube, python docs, stackoverflow.com, chatGPT) resources to learn how to program this.
Here is the order of what we programmed:
1. image transformations
2. model definition
3. model training
4. model evaluation
5. data loading
6. building identification

USE GUIDE: 1. Make sure all packages are installed. 2. Make sure the dataset is installed. 3. Set data_dir to the file location to where your dataset is. 4. Ensure your train and val folders are set up and make sure the carpenter hall are jpg and not heic. 5. Set test_image_path to the image that you want the AI to identify the building in.

NOTE: This should run without any trained model files as it automatically trains a new model each time anyways.

PROGRAM PROBLEM ACKNOWLEDGEMENT: We understand that the program is not user friendly but when we tried to make it more user friendly for somereason the accuracy of the model tanked. There was not enough time to find out wy the accuracy tanked so the we did not update the program with our planned features to make it more user friendly.

BEST MODEL METRICS: 
Accuracy: 0.9464
Precision: 0.9339
Recall: 0.9464
F1 Score: 0.9320
Log Loss: 0.4572
(after 9 epochs)
