# %%
import torch
import torch.nn.functional 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from PIL import Image
from model import AE_model


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
        stress1 = np.array(stress).astype(np.float32).transpose(2, 0, 1)/255.0
        # stress = np.array(stress)
        # plt.imshow(stress)
        # plt.show()
        
        if self.transform:
            stress = self.transform(stress)
        # stress1=stress.permute([1,2,0])
        # plt.imshow(stress1)
        # plt.show()
        return stress,stress1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

traindataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
    transform=transform
)

dataloader = DataLoader(traindataset, batch_size=8, shuffle=True, num_workers=0)

loss_fn = torch.nn.MSELoss(reduction='mean').cuda()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE_model().to(device)
model.load_state_dict(torch.load('ae_model_state_dict.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
print(f"initial lr: {scheduler.get_last_lr()[0]}")

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = []
    for batch_idx, (stress,stress1) in enumerate(dataloader):
        stress = stress.to(device)
        
        recon_stress, _ = model(stress)

        loss = loss_fn(recon_stress, stress1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()) 
        if batch_idx % 100 == 0:
            print(f" batchidx [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}，lr: {scheduler.get_last_lr()[0]:.8f}")

    print(f"== Epoch [{epoch+1}/{num_epochs}], avrage Loss: {sum(train_loss) / len(train_loss):.4f} ==")
    scheduler.step()
    if sum(train_loss) / len(dataloader.dataset) < 0.001:
        torch.save(model.state_dict(), 'ae_model_state_dict.pth')
