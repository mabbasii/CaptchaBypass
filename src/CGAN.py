import os
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

CHARACTERS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
CAPTCHA_MAX_LEN = 5
CHAR_MAX_LEN = len(CHARACTERS)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Dataset
class CustomDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.total_imge = [file for file in os.listdir(main_dir) if file.endswith(".png")]
        self.main_dir = main_dir
        self.transform = transform

    def __len__(self):
        return len(self.total_imge)

    def __getitem__(self, idx):
        path = os.path.join(self.main_dir, self.total_imge[idx])
        file_name = self.total_imge[idx]
        try:
            image = Image.open(path).convert("L")
        except:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        label_tensor = self.encode_label(os.path.splitext(file_name)[0])
        return image, label_tensor

    def encode_label(self, captcha):
        label_tensor = torch.zeros(CAPTCHA_MAX_LEN, CHAR_MAX_LEN)
        for i, char in enumerate(captcha):
            if char in CHARACTERS:
                label_tensor[i][CHARACTERS.index(char)] = 1.0
        return label_tensor

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomDataset("data/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Data loaded: ", len(dataset), "images")

os.makedirs("graph_results", exist_ok=True)

class Generator(nn.Module):
    def __init__(self, noise_size=100, conv_dim=64):
        super().__init__()
        self.label_fc = nn.Linear(CAPTCHA_MAX_LEN * CHAR_MAX_LEN, noise_size)
        self.deconv1 = nn.ConvTranspose2d(noise_size * 2, conv_dim * 16, 4, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(conv_dim * 16, conv_dim * 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(conv_dim, 1, 4, 2, 1)
        self.deconv7 = nn.ConvTranspose2d(1, 1, 4, 2, 1)

        self.bn1 = nn.InstanceNorm2d(conv_dim * 16)
        self.bn2 = nn.InstanceNorm2d(conv_dim * 8)
        self.bn3 = nn.InstanceNorm2d(conv_dim * 4)
        self.bn4 = nn.InstanceNorm2d(conv_dim * 2)
        self.bn5 = nn.InstanceNorm2d(conv_dim)

    def forward(self, z, label):
        batch_size = z.size(0)
        label_embedding = self.label_fc(label.view(batch_size, -1))
        combined = torch.cat([z, label_embedding], dim=1).view(batch_size, -1, 1, 1)
        x = F.relu(self.bn1(self.deconv1(combined)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = torch.tanh(self.deconv6(x))
        x = torch.tanh(self.deconv7(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super().__init__()
        self.label_fc = nn.Linear(CAPTCHA_MAX_LEN * CHAR_MAX_LEN, 256 * 256)
        self.conv1 = nn.Conv2d(2, conv_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(conv_dim * 8, conv_dim * 16, 4, 2, 1)
        self.conv6 = nn.Conv2d(conv_dim * 16, conv_dim * 16, 4, 2, 1)

        self.bn1 = nn.InstanceNorm2d(conv_dim)
        self.bn2 = nn.InstanceNorm2d(conv_dim * 2)
        self.bn3 = nn.InstanceNorm2d(conv_dim * 4)
        self.bn4 = nn.InstanceNorm2d(conv_dim * 8)
        self.bn5 = nn.InstanceNorm2d(conv_dim * 16)
        self.bn6 = nn.InstanceNorm2d(conv_dim * 16)

        self.real_or_fake = nn.Conv2d(conv_dim * 16, 1, 4)
        self.character_pred = nn.Linear(conv_dim * 16 * 4 * 4, CAPTCHA_MAX_LEN * CHAR_MAX_LEN)

    def forward(self, x, label):
        batch_size = x.size(0)
        label_embedding = self.label_fc(label.view(batch_size, -1)).view(batch_size, 1, 256, 256)
        x = torch.cat([x, label_embedding], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        real_fake = self.real_or_fake(x).view(batch_size)
        character_pred = self.character_pred(x.view(batch_size, -1)).view(batch_size, CAPTCHA_MAX_LEN, CHAR_MAX_LEN)
        return real_fake, character_pred

adversarial_loss = nn.BCEWithLogitsLoss()
ocr_loss = nn.CrossEntropyLoss()
G, D = Generator().to(device), Discriminator().to(device)
g_optimizer = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))

epochs = 30
g_adv_losses, g_ocr_losses = [], []
d_adv_losses, d_ocr_losses, d_total_losses = [], [], []

for epoch in range(epochs):
    epoch_start = time.time()
    g_running_loss = 0
    d_running_loss = 0
    g_adv_running = 0
    g_ocr_running = 0
    d_adv_running = 0
    d_ocr_running = 0

    for real_images, real_labels in dataloader:
        real_images, real_labels = real_images.to(device), real_labels.to(device)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, 100, device=device)

        fake_images = G(noise, real_labels).detach()
        real_case, real_chars = D(real_images, real_labels)
        fake_case, fake_chars = D(fake_images, real_labels)

        d_real_loss = adversarial_loss(real_case, torch.ones_like(real_case))
        d_fake_loss = adversarial_loss(fake_case, torch.zeros_like(fake_case))
        real_targets = real_labels.argmax(dim=2)
        d_ocr_real = ocr_loss(real_chars.view(-1, CHAR_MAX_LEN), real_targets.view(-1))
        d_ocr_fake = ocr_loss(fake_chars.view(-1, CHAR_MAX_LEN), real_targets.view(-1))

        d_adv_loss = d_real_loss + d_fake_loss
        d_ocr_loss_total = d_ocr_real + d_ocr_fake
        d_loss = d_adv_loss + d_ocr_loss_total

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        fake_images = G(noise, real_labels)
        fake_case, fake_chars = D(fake_images, real_labels)
        g_adv = adversarial_loss(fake_case, torch.ones_like(fake_case))
        g_ocr = ocr_loss(fake_chars.view(-1, CHAR_MAX_LEN), real_targets.view(-1))
        g_loss = g_adv + g_ocr

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_running_loss += d_loss.item()
        d_adv_running += d_adv_loss.item()
        d_ocr_running += d_ocr_loss_total.item()
        g_running_loss += g_loss.item()
        g_adv_running += g_adv.item()
        g_ocr_running += g_ocr.item()

    d_total_losses.append(d_running_loss)
    d_adv_losses.append(d_adv_running)
    d_ocr_losses.append(d_ocr_running)
    g_adv_losses.append(g_adv_running)
    g_ocr_losses.append(g_ocr_running)

    epoch_time = time.time() - epoch_start
    remaining = (epochs - (epoch + 1)) * epoch_time / 60
    print(f"[Epoch {epoch+1}/{epochs}] ETA: {remaining:.1f} min")

    if (epoch + 1) % 5 == 0:
        G.eval()
        with torch.no_grad():
            sample_labels = real_labels[:4]
            sample_noise = torch.randn(4, 100, device=device)
            sample_imgs = G(sample_noise, sample_labels)
            sample_imgs = (sample_imgs + 1) / 2.0
            grid = vutils.make_grid(sample_imgs.cpu(), nrow=4, padding=2)
            plt.figure(figsize=(8, 2))
            plt.axis("off")
            plt.title(f"Generated CAPTCHAs (Epoch {epoch + 1})")
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.savefig(f"graph_results/generated_epoch_{epoch + 1}.png")
            plt.close()
        G.train()

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.plot(d_total_losses, label="D Total")
plt.plot(d_adv_losses, label="D Adv")
plt.plot(d_ocr_losses, label="D OCR")
plt.title("Discriminator Losses")
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(g_adv_losses, label="G Adv")
plt.title("Generator Adversarial Loss")
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(g_ocr_losses, label="G OCR")
plt.title("Generator OCR Loss")
plt.legend()

plt.subplot(1, 4, 4)
plt.plot([sum(x) for x in zip(g_adv_losses, g_ocr_losses)], label="G Total")
plt.title("Generator Total Loss")
plt.legend()

plt.tight_layout()
plt.savefig("graph_results/loss_curves.png")
plt.close()
