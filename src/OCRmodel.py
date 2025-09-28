import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

MAX_LEN = 5
CHAR_SET = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
NUM_CLASSES = len(CHAR_SET)
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHAR_SET)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class CaptchaDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.total_imge = [file for file in os.listdir(main_dir) if file.endswith(".png")]
        self.main_dir = main_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.total_imge)

    def __getitem__(self, idx):
        while True:
            file_name = self.total_imge[idx]
            path = os.path.join(self.main_dir, file_name)

            try:
                label_text = os.path.splitext(file_name)[0]
                assert len(label_text) == MAX_LEN
                assert all(char in CHAR_TO_INDEX for char in label_text)

                image = Image.open(path).convert("L")
                image = self.transform(image)

                label = torch.tensor([CHAR_TO_INDEX[char] for char in label_text], dtype=torch.long)
                return image, label

            except (Image.UnidentifiedImageError, AssertionError, OSError) as e:
                print(f"⚠️ Skipping file: {file_name} due to {type(e).__name__}")
                idx = (idx + 1) % len(self.total_imge)

class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, MAX_LEN * NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x.view(-1, MAX_LEN, NUM_CLASSES)

def calculate_character_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.numel()
    return correct / total

def calculate_sequence_accuracy(preds, labels):
    return (preds == labels).all(dim=1).float().mean().item()

if __name__ == "__main__":
    model = OCRModel().cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    dataset = CaptchaDataset("full_dataset/Large_Captcha_Dataset")

    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    Epochs = 100
    train_char_accuracies = []
    test_char_accuracies = []
    train_seq_accuracies = []
    test_seq_accuracies = []

    best_acc = 0


    start_time = time.time()

    for epoch in range(Epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Epochs}]")
        for images, labels in loop:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            loss = 0
            for i in range(MAX_LEN):
                loss += loss_function(outputs[:, i, :], labels[:, i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=2)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            loop.set_postfix(loss=loss.item())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        train_char_acc = calculate_character_accuracy(all_preds, all_labels)
        train_seq_acc = calculate_sequence_accuracy(all_preds, all_labels)
        train_char_accuracies.append(train_char_acc)
        train_seq_accuracies.append(train_seq_acc)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                preds = torch.argmax(outputs, dim=2)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        test_char_acc = calculate_character_accuracy(all_preds, all_labels)
        test_seq_acc = calculate_sequence_accuracy(all_preds, all_labels)
        test_char_accuracies.append(test_char_acc)
        test_seq_accuracies.append(test_seq_acc)

 
        scheduler.step(test_seq_acc)

        if test_seq_acc > best_acc:
            best_acc = test_seq_acc
            torch.save(model.state_dict(), "ocr_best_model.pth")

        elapsed = time.time() - start_time
        remaining = (Epochs - (epoch + 1)) * elapsed / 60
        print(f"[Epoch {epoch+1}/{Epochs}] ETA: {remaining:.1f} min")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Epochs + 1), train_char_accuracies, label="Train Char Accuracy")
    plt.plot(range(1, Epochs + 1), test_char_accuracies, label="Test Char Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Per-Character Accuracy")
    plt.title("OCR Character Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("char_accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Epochs + 1), train_seq_accuracies, label="Train Sequence Accuracy")
    plt.plot(range(1, Epochs + 1), test_seq_accuracies, label="Test Sequence Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Full Sequence Accuracy")
    plt.title("OCR Full Sequence Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sequence_accuracy_curve.png")
    plt.show()
