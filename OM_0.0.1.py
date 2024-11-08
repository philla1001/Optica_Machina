from cmath import phase

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import device
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import gdown



def get_data_transforms():

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(512),        # Randomly crop the image to 512x512
            transforms.RandomHorizontalFlip(),        # Randomly flip the image horizontally
            transforms.ToTensor(),                    # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),                    # Resize the image to 512x512
            transforms.CenterCrop(512),                # Crop from the center
            transforms.ToTensor(),                     # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
    }
    return data_transforms


def load_data(data_dir, batch_size=32):
    data_transforms = get_data_transforms()


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}


    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("Dataset Sizes:", dataset_sizes)

    return dataloaders, dataset_sizes, class_names

def create_model(num_classes):

    model = models.resnet18(pretrained=True)


    num_ftrs = model.fc.in_features


    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, dataset_sizes=None):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU/CPU

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update the weights

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss


def evaluate_model(model, dataloader, criterion, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)  # Get the class with the highest score
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities

            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())  # Store predicted labels
            all_probs.extend(probs.cpu().numpy())  # Store probabilities

    avg_loss = running_loss / len(dataloader.dataset)  # Calculate average loss


    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logloss = log_loss(all_labels, all_probs)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Log Loss: {logloss:.4f}')

    return accuracy, precision, recall, f1, logloss


if __name__ == "__main__":
    data_dir = r'C:\Users\pjc13\PycharmProjects\OM_0.0.1\Dataset_Pics\Campus Vision Challenge Dataset-20241108T001339Z-001'
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

    print("Validation Set Performance:")
    evaluate_model(model, dataloaders['val'], criterion, class_names)

    # Save the best model
    torch.save(model.state_dict(), "best_model.pth")
