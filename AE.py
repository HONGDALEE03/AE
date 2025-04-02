# %%[markdown]
#

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torchvision.utils as vutils

class StressDataset(Dataset):
    def __init__(self, stress_dir, transform=None):
        self.stress_dir = stress_dir
        self.stress_files = [os.path.join(stress_dir, f) for f in os.listdir(stress_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.stress_files)

    def __getitem__(self, idx):
        img_path = self.stress_files[idx]
        stress = Image.open(img_path).convert('RGB')
        if self.transform:
            stress = self.transform(stress)
        return stress

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

traindataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
    transform=transform
)

dataloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0)

class AE(nn.Module):
    def __init__(self, latent_dim=128):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Linear(512 * 8 * 8, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid() 
        )

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc(h)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 8, 8)  
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(latent_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, stress in enumerate(dataloader):
        stress = stress.to(device)
        optimizer.zero_grad()
        recon_stress, _ = model(stress)

        loss = loss_function(recon_stress, stress)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}")
#%%
torch.save(model.state_dict(), 'ae_model_state_dict.pth')

model.eval()
with torch.no_grad():
    z = torch.randn(64, 128).to(device)
    generated_stress = model.decode(z)

# 保存生成的图像
output_dir = "D:\\Microsoft VS Code\\新建文件夹\\GENERATE"
os.makedirs(output_dir, exist_ok=True)
vutils.save_image(generated_stress, os.path.join(output_dir, "ae_generated_stress.png"), nrow=8, normalize=True)