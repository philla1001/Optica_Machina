import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Use a pre-trained ResNet18 model
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 10 classes (change as needed)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()  # Common loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, torch.zeros(images.size(0), dtype=torch.long).to(
            device))  # Dummy target, replace with real labels
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

# Save the model after training
torch.save(model.state_dict(), 'trained_model.pth')
