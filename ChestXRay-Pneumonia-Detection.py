import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ===============================
# Kaggle Authentication
# ===============================
api = KaggleApi()
api.authenticate()

# ===============================
# Dataset Download & Path
# ===============================
dataset_dir = "chest_xray_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Download & unzip
api.dataset_download_files("paultimothymooney/chest-xray-pneumonia",
                           path=dataset_dir,
                           unzip=True,
                           quiet=False)

# Adjust path if Kaggle adds a subfolder
dataset_dir = os.path.join(dataset_dir, "chest-xray-pneumonia")

# ===============================
# 1. Configuration
# ===============================
batch_size = 32
num_classes = 2
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. Data Preprocessing
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "test"), transform=val_test_transform)

# ===============================
# WeightedRandomSampler for imbalance
# ===============================
train_targets = [sample[1] for sample in train_dataset.samples]
class_counts = torch.bincount(torch.tensor(train_targets))
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[train_targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===============================
# 3. Load Pretrained DenseNet121
# ===============================
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# ===============================
# 4. Loss Function & Optimizer
# ===============================
class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
class_weights_loss = 1.0 / class_counts_tensor
class_weights_loss /= class_weights_loss.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights_loss)  # ⚡ روی CPU باشه

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# ===============================
# 5. Training Function
# ===============================
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Training
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects / len(val_dataset)
        val_losses.append(val_epoch_loss)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}\n")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_densenet121.pth")

    print(f"Best Val Acc: {best_acc:.4f}")
    return train_losses, val_losses

# ===============================
# 6. Train the Model
# ===============================
train_losses, val_losses = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# ===============================
# 7. Test the Model
# ===============================
model.load_state_dict(torch.load("best_densenet121.pth"))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report on Test Data:")
print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, test_preds))

# ===============================
# 8. Plot Loss Curve
# ===============================
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()