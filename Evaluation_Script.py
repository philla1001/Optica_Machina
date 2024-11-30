
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

def load_trained_model(model_path, num_classes):
    """
    Load the trained ResNet18 model with the specified number of classes.

    Args:
        model_path (str): Path to the saved model weights.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)  # Match the training script
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Adjust the final layer
    try:
        # Load only the state_dict
        state_dict = torch.load(model_path, map_location=device)
        state_dict['fc.weight'] = torch.nn.Parameter(torch.randn(num_classes, num_ftrs))  # Adjust the weight size
        state_dict['fc.bias'] = torch.nn.Parameter(torch.randn(num_classes))  # Adjust the bias size
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture matches the one used during training.")
        raise e
    return model

def evaluate_model_on_test_set(model, dataloader, class_names, output_dir):
    """
    Evaluate the trained model on the test set and calculate metrics.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set.
        class_names (list): List of class names.
        output_dir (str): Directory to save evaluation results.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    logloss = log_loss(y_true, y_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")

    # Save metrics to a file
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Log Loss: {logloss:.4f}\n")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save confusion matrix to file
    confusion_matrix_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {confusion_matrix_file}")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    classification_report_file = os.path.join(output_dir, "classification_report.txt")
    with open(classification_report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved to {classification_report_file}")

if __name__ == "__main__":
    # Define paths and parameters
    test_dir = r'C:\Users\pjc13\PycharmProjects\OM_0.0.1\Dataset_Pics\Campus Vision Challenge Dataset-20241108T001339Z-001'  # Test dataset directory
    model_path = r'C:\Users\pjc13\PycharmProjects\OM_1.0.0\best_model.pth'  # Path to the saved model weights
    output_dir = r'C:\Users\pjc13\PycharmProjects\OM_1.0.0\evaluation_results'  # Directory to save evaluation results

    # Define transformations for the test dataset
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the test dataset
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # Get class names from the dataset
    class_names = test_dataset.classes

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_trained_model(model_path, num_classes=len(class_names))
    model = model.to(device)

    # Evaluate the model
    evaluate_model_on_test_set(model, test_dataloader, class_names, output_dir)
