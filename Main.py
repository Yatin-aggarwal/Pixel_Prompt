import torch
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data import Dataset
import torch.utils.data.dataset
from Discriminator import Discriminator
from Generator import Generator
from weights import intialize_weights
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from Save import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
batch_size = 64
image_size = 32
channels_img = 3
z_dim = 32
num_epoch = 70
feature_Disc = 64
feature_Gen = 64
real_label = 1
lstm_image_output = 32

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

data_set = Dataset(transforms)
data = DataLoader(data_set, batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, feature_Gen,lstm_image_output).to(device)
disc = Discriminator(channels_img, feature_Disc,lstm_image_output).to(device)

intialize_weights(gen)
intialize_weights(disc)

optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/Anime/real")
writer_fake = SummaryWriter(f"runs/Anime/fake")
step = 0


print("training begins")
progressbar_epoch = tqdm(total=num_epoch, desc="Epoch")
progressbar = tqdm(total=6400/batch_size, desc="Batch Index")
for epoch in range(num_epoch):
    if (epoch % 2 == 0):
        state = {
            "Generator": gen.state_dict(),
            "Discriminator": disc.state_dict(),
            "optimizer_gen": optimizer_gen.state_dict(),
            "optimizer_disc": optimizer_disc.state_dict(),
            "epoch": epoch,
        }
        save_model(state)
    for batch_idx, (image, prompt) in enumerate(data):
        disc.zero_grad()
        real = image.to(device)
        prompt = prompt.to(device).type(torch.float32)
        noise = torch.randn(image.shape[0], 32, 1, 1).to(device)

        real_pred = disc(real, prompt)
        real_loss = criterion(real_pred, torch.ones_like(real_pred))
        real_loss.backward()

        fake_pred = gen(noise, prompt)
        fake = disc(fake_pred.detach(), prompt)
        fake_loss = criterion(fake, torch.zeros_like(fake))
        fake_loss.backward(retain_graph=True)
        optimizer_disc.step()

        noise = torch.randn(image.shape[0], 32, 1, 1).to(device)
        fake = gen(noise, prompt)
        fake_pred = disc(fake, prompt)
        fake_loss = criterion(fake_pred, torch.ones_like(fake_pred))
        gen.zero_grad()
        fake_loss.backward()
        optimizer_gen.step()

        progressbar.update(1)

        if batch_idx % 10 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise, prompt.type(torch.float32))
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1
    progressbar_epoch.update(1)