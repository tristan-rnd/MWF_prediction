import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from skimage import io
from sklearn.model_selection import train_test_split
import skimage
from skimage.metrics import structural_similarity

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ---------------------

batch = 30

T1_T2_match = pd.read_csv("index.csv")
_ , test_match = train_test_split(T1_T2_match, test_size=0.1, random_state=25)
# ---------------------

class MWF_pred_Dataset(Dataset):
    def __init__(self, match, img_dir, transform=None, target_transform=None):
        self.img_match = match
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_match)

    def __getitem__(self, idx):

        T1 = io.imread(self.img_dir + "T1/Reoriented/Slices/" + str(self.img_match.iloc[idx, 0]))

        T2 = io.imread(self.img_dir + "T2/Reoriented/Aligned/Slices/" + str(self.img_match.iloc[idx, 1]))

        MWF = io.imread(self.img_dir + "MWF/Reoriented/Aligned/Slices/" + str(self.img_match.iloc[idx, 2]))

        T1_T2 = np.array([T1, T2])
        T1_T2 = np.transpose(T1_T2, (1, 2, 0))

        if self.transform:
            T1_T2_transformed = self.transform(T1_T2)
        else:
            T1_T2_transformed = T1_T2
        if self.target_transform:
            MWF_transformed = self.target_transform(MWF)
        else:
            MWF_transformed = MWF

        return T1_T2_transformed, MWF_transformed


# -----------------------

test_data = MWF_pred_Dataset(
    img_dir="",
    match=test_match,
    transform=transforms.ToTensor(),
    target_transform=transforms.ToTensor()
)

test_loader = DataLoader(test_data, batch_size=27, shuffle=True)

# -------------------------------

T1_T2_imgs = torch.stack([T for T, _ in test_loader], dim=2)

print(T1_T2_imgs.view(2, -1).mean(dim=1))
print(T1_T2_imgs.view(2, -1).std(dim=1))
T1_T2_test_mean = T1_T2_imgs.view(2, -1).mean(dim=1)
T1_T2_test_std = T1_T2_imgs.view(2, -1).std(dim=1)
del T1_T2_imgs

# ------------------------

test_data_normalized = MWF_pred_Dataset(
    img_dir="",
    match=test_match,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(T1_T2_test_mean, T1_T2_test_std)
    ]),
    target_transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_loader_norm = DataLoader(test_data_normalized, batch_size=batch, shuffle=True)

# ------------------------------


display.samples(test_loader_norm, norm=1)


# -------------------------------


class MWF_pred_net(nn.Module):

    def __init__(self):
        super(MWF_pred_net, self).__init__()

        self.conv1 = nn.Conv2d(2, 64, 5, padding=2)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=2)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=2)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 5, padding=2)
        self.norm6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 64, 3)
        self.norm7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 1, 3)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.norm6(self.conv6(x)))
        x = F.relu(self.norm7(self.conv7(x)))

        x = self.conv8(x)

        return x


net = MWF_pred_net().to(device)
net.load_state_dict(torch.load('MWF_pred'))
net.eval()

# -------------------------------

Prediction = display.prediction(net, test_loader_norm, device)

# -------------------------------

T1_T2, MWF = next(iter(test_loader_norm))
fig, axs = plt.subplots(1, 2)
fig.tight_layout()
fig.subplots_adjust(top=0.92)
axs[0].imshow(MWF[5][0], cmap="gray")
axs[1].imshow(net(T1_T2[5].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach(),
              cmap="gray")
plt.savefig('True vs Predicted.pdf')
fig.suptitle("True vs Predicted")
plt.show()
output = net(T1_T2[5].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach().numpy()
target = MWF[5][0].numpy()
io.imsave("MWF_Prediction.tiff", output)
io.imsave("MWF_true.tiff", target.astype(np.float32))

# -------------------------------

moyenne = 0
t = 0
for j in range(10):
    T1_T2, MWF = next(iter(test_loader_norm))
    for i in range(len(test_loader_norm)):
        pred = net(T1_T2[i].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach().numpy()
        target = MWF[i][0].numpy()
        index, SSIM = skimage.metrics.structural_similarity(target, pred, data_range=pred.max() - pred.min(),
                                                            gradient=True)
        if index > 0.25:
            moyenne = moyenne + index
            t = t + 1
print(moyenne / t)

# -------------------------------
fig, ax = plt.subplots(1,1)
fig.suptitle("Similarities")
ax.imshow(SSIM)
ax.text(0.8, 0.2, "Index SSIM: " + "{:.3f}".format(index), horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
plt.show()
plt.savefig('Similarities.pdf')
