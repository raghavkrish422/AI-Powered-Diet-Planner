import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Step 1: Extract Zip File
def extract_zip(zip_path, extract_to):
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted: {zip_path}")
    else:
        print(f"❌ Not a zip file: {zip_path}")

# Step 2: Define a Simple CNN Model (you can customize this later)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Step 3: Training Loop
def train_model(data_dir, epochs=10, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"🔎 Found {len(dataset.classes)} classes.")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = CNNModel(num_classes=len(dataset.classes))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}"):
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# Entry point
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Helps on macOS/Windows

    # ✅ Update your zip file path and target folder here
    zip_path = '/Users/pavankumaretta/Desktop/AML_Project/food-101.zip.zip'          # <-- path to your zip file
    extract_path = './data'            # <-- where to extract it

    extract_zip(zip_path, extract_path)
    train_model(extract_path, epochs=10, batch_size=32)




# import os
# import zipfile
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import shutil

# # ===============================
# # Step 1: Unzip Dataset
# # ===============================
# zip_path = "/Users/pavankumaretta/Desktop/AML_Project/food-101.zip.zip"  # 🔁 Replace with your zip file path
# extract_path = "./food-101"

# if not os.path.exists(extract_path):
#     print("📦 Unzipping dataset...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#     print("✅ Dataset unzipped.")

# # Move the 'images' folder inside food-101/ if needed
# if os.path.exists(os.path.join(extract_path, "food-101", "images")):
#     extract_path = os.path.join(extract_path, "food-101")

# train_dir = os.path.join(extract_path, "images")
# val_dir = train_dir  # Same structure used for both in Food101 (usually split via file lists)

# # ===============================
# # Step 2: Data Preprocessing
# # ===============================
# transform_train = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3),
# ])

# transform_val = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3),
# ])

# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
# val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# num_classes = len(train_dataset.classes)
# print(f"🔎 Found {num_classes} classes.")

# # ===============================
# # Step 3: Define Model from Scratch
# # ===============================
# class Food101CNN(nn.Module):
#     def __init__(self, num_classes):
#         super(Food101CNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x64
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32x32
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# 16x16
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128*16*16, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Food101CNN(num_classes).to(device)

# # ===============================
# # Step 4: Train Model
# # ===============================
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# epochs = 10

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for imgs, labels in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}"):
#         imgs, labels = imgs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     print(f"📊 Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     print(f"✅ Validation Accuracy: {100*correct/total:.2f}%\n")

# # ===============================
# # Step 5: Save Model
# # ===============================
# os.makedirs("models", exist_ok=True)
# torch.save(model.state_dict(), "models/amlprojectmodel.pth")
# print("📁 Model saved as models/amlprojectmodel.pth")
