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
Metrics from right after the model is trained:
Accuracy: 0.9464
Precision: 0.9339
Recall: 0.9464
F1 Score: 0.9320
Log Loss: 0.4572
(after 10 epochs)

Metrics from the best model being evaluated by the evaluation script:
Accuracy: 0.3401
Precision: 0.2331
Recall: 0.3401
F1-Score: 0.2609
Log Loss: 15.6997
NOTE: The evaluation script was rushed due to main dev's computer being broken for several days and I faced a lot of errors when trying to use the eval script so it might be inaccurate.
Classification report from eval script:
                                 precision    recall  f1-score   support

Campus Vision Challenge Dataset       0.00      0.00      0.00       158
                          train       0.35      0.27      0.31       168
                            val       0.34      0.73      0.46       168

                       accuracy                           0.34       494
                      macro avg       0.23      0.33      0.26       494
                   weighted avg       0.23      0.34      0.26       494
                   


