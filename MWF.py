import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from skimage import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

batch = 30

T1_T2_match = pd.read_csv("index.csv")
train_match, test_match = train_test_split(T1_T2_match, test_size=0.1, random_state=25)
train_match, validation_match = train_test_split(train_match, test_size=0.2, random_state=25)


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


train_data = MWF_pred_Dataset(
    img_dir="",
    match=train_match,
    transform=transforms.ToTensor(),
    target_transform=transforms.ToTensor()
)
validation_data = MWF_pred_Dataset(
    img_dir="",
    match=validation_match,
    transform=transforms.ToTensor(),
    target_transform=transforms.ToTensor()
)

test_data = MWF_pred_Dataset(
    img_dir="",
    match=test_match,
    transform=transforms.ToTensor(),
    target_transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, batch_size=47, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=27, shuffle=False)
test_loader = DataLoader(test_data, batch_size=27, shuffle=True)

T1_T2, MWF = next(iter(test_loader))
plt.imshow(MWF[0][0], cmap='gray')
print(torch.max(MWF[0][0]).item())

# ------------------------------


display.samples(train_loader, norm=0)

# -------------------------------

T1_T2_imgs = torch.stack([T for T, _ in test_loader], dim=2)

print(T1_T2_imgs.view(2, -1).mean(dim=1))
print(T1_T2_imgs.view(2, -1).std(dim=1))
T1_T2_test_mean = T1_T2_imgs.view(2, -1).mean(dim=1)
T1_T2_test_std = T1_T2_imgs.view(2, -1).std(dim=1)
del T1_T2_imgs

"""
Test loader :
tensor([63.7809, 63.6069], dtype=torch.float64)
tensor([119.4135, 120.0291], dtype=torch.float64)
"""
T1_T2_imgs = torch.stack([T for T, _ in validation_loader], dim=2)

print(T1_T2_imgs.view(2, -1).mean(dim=1))
print(T1_T2_imgs.view(2, -1).std(dim=1))
T1_T2_val_mean = T1_T2_imgs.view(2, -1).mean(dim=1)
T1_T2_val_std = T1_T2_imgs.view(2, -1).std(dim=1)
del T1_T2_imgs
"""
Validation loader :
tensor([61.8711, 62.5530], dtype=torch.float64)
tensor([116.0363, 118.3785], dtype=torch.float64)
"""

T1_T2_imgs = torch.stack([T for T, _ in train_loader], dim=2)

print(T1_T2_imgs.view(2, -1).mean(dim=1))
print(T1_T2_imgs.view(2, -1).std(dim=1))
T1_T2_train_mean = T1_T2_imgs.view(2, -1).mean(dim=1)
T1_T2_train_std = T1_T2_imgs.view(2, -1).std(dim=1)
del T1_T2_imgs

""""
Train loader :
tensor([59.5729, 62.0313], dtype=torch.float64)
tensor([115.3614, 118.4364], dtype=torch.float64)
"""

# ------------------------

train_data_normalized = MWF_pred_Dataset(
    img_dir="",
    match=train_match,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(T1_T2_train_mean, T1_T2_train_std)
    ]),
    target_transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

validation_data_normalized = MWF_pred_Dataset(
    img_dir="",
    match=validation_match,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(T1_T2_val_mean, T1_T2_val_std)
    ]),
    target_transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

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

train_loader_norm = DataLoader(train_data_normalized, batch_size=batch, shuffle=True)
validation_loader_norm = DataLoader(validation_data_normalized, batch_size=batch, shuffle=False)
test_loader_norm = DataLoader(test_data_normalized, batch_size=batch, shuffle=True)

# ------------------------------


display.samples(test_loader_norm, norm=1)


# -------------------------------

def train(model, device, train_loader, validation_loader, epochs, lr=0.001):
    criterion = nn.MSELoss()

    train_loss, validation_loss = [], []

    for epoch in range(epochs):
        if epoch % 3 == 0 and epoch != 0:
            lr = lr / 2
        running_loss = 0.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for data, target in tepoch:

                if device == "cuda":
                    data, target = data.to(device).type(torch.cuda.FloatTensor), target.to(device).to(device).type(
                        torch.cuda.FloatTensor)
                else:
                    data, target = data.to(device).float(), target.to(device).float()

                target = torch.unsqueeze(target, 1)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.reshape(-1), target.reshape(-1))
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(lr=lr, loss=loss.item(), epoch=str(epoch + 1) + "/" + str(epochs))
                time.sleep(0.1)
                running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))

        model.eval()
        running_loss = 0.

        with torch.no_grad():
            for data, target in validation_loader:

                if device == "cuda":
                    data, target = data.to(device).type(torch.cuda.FloatTensor), target.to(device).to(device).type(
                        torch.cuda.FloatTensor)
                else:
                    data, target = data.to(device).float(), target.to(device).float()

                target = torch.unsqueeze(target, 1)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.reshape(-1), target.reshape(-1))

                tepoch.set_postfix(loss=loss.item())
                running_loss += loss.item()
                time.sleep(0.1)

            validation_loss.append(running_loss / len(validation_loader))

    return train_loss, validation_loss


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


epochs = 15
lr = 0.0005

net = MWF_pred_net().to(device)

print("Nombre total de paramtres {:10d}".format(sum(p.numel() for p in net.parameters())))
t = time.time()
train_loss, validation_loss = train(net, device, train_loader_norm, validation_loader_norm, epochs, lr)
print("Temps d'apprentissage : ", int(time.time() - t) // 60)

# -------------------------------



def loss(train_loss, validation_loss, epochs):
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(range(0, epochs), train_loss, label="Train data")
    ax_acc.plot(range(0, epochs), validation_loss, label="validation data")
    ax_acc.set_title("Loss over epochs")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Loss")
    ax_acc.legend()
    ax_acc.text(0.8, 0.2, "Final loss: " + "{:.10f}".format(validation_loss[-1]), horizontalalignment='center',
                verticalalignment='center', transform=ax_acc.transAxes)
    plt.savefig('Loss.pdf')

loss(train_loss, validation_loss, epochs)
# -------------------------------


torch.save(net.state_dict(), "MWF_pred")

net.load_state_dict(torch.load('MWF_pred'))
net.eval()


# -------------------------------

def prediction(net, loader, device="cuda", nb=8):
    T1_T2, MWF = next(iter(loader))
    fig_trained, ax_trained = plt.subplots(nb, 4, figsize=(10, 15))
    fig_trained.suptitle("MWF created from T1 & T2")

    fig_trained.tight_layout()
    fig_trained.subplots_adjust(top=0.92)

    for i in range(nb):
        if device == "cuda":
            output = net(T1_T2[i].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach()
        else:
            output = net(T1_T2[i].to(device).float().unsqueeze(dim=0)).item()

        ax_trained[i][0].imshow(T1_T2[i][0], cmap="gray")
        ax_trained[i][0].set_title("T1")

        ax_trained[i][1].imshow(T1_T2[i][1], cmap="gray")
        ax_trained[i][1].set_title("T2")

        ax_trained[i][2].imshow(MWF[i][0], cmap="gray")
        ax_trained[i][2].set_title("True MWF")

        ax_trained[i][3].imshow(output, cmap="gray")
        ax_trained[i][3].set_title("Predicted MWF")
    plt.savefig('Prediction.pdf')

    return output

Prediction = prediction(net, test_loader_norm, device)
plt.imshow(Prediction)
plt.show()

# -------------------------------


import skimage
from skimage.metrics import structural_similarity

# -------------------------------

T1_T2, MWF = next(iter(test_loader_norm))
fig, axs = plt.subplots(1, 2)
axs[0].imshow(MWF[5][0], cmap="gray")
axs[1].imshow(net(T1_T2[5].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach(), cmap="gray")
plt.savefig('VS.pdf')
plt.show()

# -------------------------------



def comparison(truth, pred):
    comp = np.zeros(np.shape(truth), dtype=np.float64)
    index, ss = skimage.metrics.structural_similarity(truth, pred, data_range=pred.max() - pred.min(), gradient=True)

    return comp, ss, index


# -------------------------------
moyenne = 0
t = 0
for j in range(25):
    T1_T2, MWF = next(iter(test_loader_norm))
    for i in range(len(test_loader_norm)):
        comp, ss, index = comparison(MWF[i][0].numpy(), net(T1_T2[i].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach().numpy())
        if index > 0.25:
            moyenne = moyenne + index
            t = t + 1
print(moyenne/t)

# -------------------------------

plt.imshow(ss, cmap="gray")
plt.show()
plt.savefig('Similarities.pdf')
# -------------------------------


# -------------------------------
