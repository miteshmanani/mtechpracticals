import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define a simple CNN with BatchNorm


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7, 10)
        )

    def forward(self, x):
        return self.net(x)


# Load MNIST (train) and MNIST with noise (test)
transform_clean = transforms.Compose([transforms.ToTensor()])
transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.2 * torch.randn_like(x))  # Add noise
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform_clean, download=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform_noisy)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(1):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- Test-time Adaptation ---


def test_with_adaptation(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    # Enable BN layers to update stats (simulating test-time adaptation)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()  # Update running stats on test data

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass (to update BN stats)
            _ = model(images)

            # In actual TTA, you might fine-tune weights here or use entropy minimization
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy after TTA: {100 * correct / total:.2f}%')


test_with_adaptation(model, test_loader)
