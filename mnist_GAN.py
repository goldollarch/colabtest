import pandas
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0
        img_df = self.data_df.iloc[index, 1:].values
        image_values = torch.FloatTensor(img_df) / 255.0
        return label, image_values, target


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784 + 10, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)


def generate_random_one_hot(size):
    label_tensor = torch.zeros(size)
    random_idx = np.random.randint(0, size)
    label_tensor[random_idx] = 1
    return label_tensor


def generate_random(size):
    random_data = torch.randn(size)
    return random_data


def plot_conditional_images(label):
    label_tensor = torch.zeros(10)
    label_tensor[label] = 1.0
    f, ax_arr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = generator_net(generate_random(100).to(DEVICE),
                                   label_tensor.to(DEVICE))
            img = output.detach().cpu().numpy().reshape(28, 28)
            ax_arr[i, j].imshow(img, interpolation='None', cmap='Blues')
    plt.show()


train_dataset = MnistDataset('mnist_train.csv')

discriminator_net = Discriminator().to(DEVICE)
generator_net = Generator().to(DEVICE)
loss_function = torch.nn.BCELoss()

optimizer_d = torch.optim.Adam(discriminator_net.parameters())
optimizer_g = torch.optim.Adam(generator_net.parameters())

progress_d_real = []
progress_d_fake = []
progress_g = []
counter = 0
real_label = torch.FloatTensor([1.0]).to(DEVICE)
fake_label = torch.FloatTensor([0.0]).to(DEVICE)

for i in range(10):
    for label, real_data, target in train_dataset:
        discriminator_net.zero_grad()
        output = discriminator_net(real_data.to(DEVICE), target.to(DEVICE))
        loss_d_real = loss_function(output, real_label)
        random_label = generate_random_one_hot(10).to(DEVICE)
        gen_img = generator_net(generate_random(100).to(DEVICE),
                                random_label)
        output = discriminator_net(gen_img.detach(), random_label)
        loss_d_fake = loss_function(output, fake_label)
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        generator_net.zero_grad()
        gen_img = generator_net(generate_random(100).to(DEVICE),
                                random_label)
        output = discriminator_net(gen_img,
                                   random_label)
        loss_g = loss_function(output, real_label)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        counter += 1
        if counter % 500 == 0:
            progress_d_real.append(loss_d_real.item())
            progress_d_fake.append(loss_d_fake.item())
            progress_g.append(loss_g.item())
        if counter % 10000 == 0:
            print(f'epoch = {i + 1}, counter = {counter}')

    plot_conditional_images(9)
