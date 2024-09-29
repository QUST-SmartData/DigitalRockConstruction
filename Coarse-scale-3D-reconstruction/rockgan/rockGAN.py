import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from architecture import Generator
from architecture import Discriminator
from architecture import gradient_penalty
import random
from rockgan.utils import (
    MyLoader,
    seed_everything,
    porosity,
    save_checkpoint,
)

# Uploading 1024 samples extracted from the original sample
# Replace 'YOUR_DATASET_PATH' with the actual path to your dataset
DATASET_PATH = "./data"
DATASET = torch.from_numpy(np.load(os.path.join(DATASET_PATH, '2.npy')))
print(DATASET.shape)
plt.imshow(DATASET[0, 0, :, :], cmap='gray')
plt.colorbar()
image_path = './first_image.png'

# Save the image
plt.savefig(image_path)
plt.close()
# Set CUDA device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


DEVICE = torch.device('cuda')

# Specifying folder location to save models per epoch
CHECKPOINT_GEN = "./checkpoints0/generator/"
CHECKPOINT_CRITIC = "./checkpoints0/critic/"

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
Z_DIM = 16
NUM_EPOCHS = 151
CRITIC_ITERATIONS = 4
LAMBDA_GP = 25

# Initialize data loader
class MyLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


loader = DataLoader(MyLoader(DATASET), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


seed_everything(seed=11)
# Initialize generator and critic
#gen = Generator(in_channels=1, out_channels=1).to(DEVICE)
gen = Generator().to(DEVICE)
gen.train()

#critic = Discriminator(in_channels=1, out_channels=1).to(DEVICE)
critic = Discriminator().to(DEVICE)
critic.train()

# Initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, 4 * NUM_EPOCHS)

opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, 4 * NUM_EPOCHS * CRITIC_ITERATIONS)

# Fixed noise for display
fixed_noise = torch.randn(BATCH_SIZE, 1, Z_DIM, Z_DIM, Z_DIM).to(DEVICE)

# Criterion for measuring porosity difference
criterion = torch.nn.MSELoss()

# Training
losses_gen = []
losses_critic = []
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)


for epoch in range(NUM_EPOCHS):
    batches = tqdm(loader)
    mean_loss_gen = 0
    mean_loss_critic = 0
    for batch_idx, real in enumerate(batches):
        real = real.float().unsqueeze(1).to(DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # Equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, 1, Z_DIM, Z_DIM, Z_DIM).to(DEVICE)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            scheduler_critic.step()

            # Mean critic loss
            mean_loss_critic += loss_critic.item()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        scheduler_gen.step()

        # Mean generator loss
        mean_loss_gen += loss_gen.item()

        batches.set_postfix(
            epoch=epoch,
            gen_loss=loss_gen.item(),
            critic_loss=loss_critic.item(),
        )

    # Show an example of the samples generated and the losses
    if epoch % 10 == 0:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        # Samples
        gen_sample = gen(fixed_noise)[0, 0, 0, :, :].detach().cpu().numpy()
        ax[0].imshow(gen(fixed_noise)[0, 0, 0, :, :].detach().cpu().numpy(), cmap='gray')
        # Losses (generator and critic)
        ax[1].plot(losses_gen, label='generator', linewidth=2)
        ax[1].plot(losses_critic, label='critic', linewidth=2)
        ax[1].legend()
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        plt.show()
        
        output_path = os.path.join(output_dir, f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.savefig(output_path)
        plt.close(fig)

    # Save losses at each epoch
    losses_gen.append(mean_loss_gen / batch_idx)
    losses_critic.append(mean_loss_critic / (batch_idx * CRITIC_ITERATIONS))
    
    final_output_path = os.path.join(output_dir, 'final_samples.png')
    plt.imshow(gen(fixed_noise)[0, 0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.savefig(final_output_path)
    plt.close()
    
    #Save checkpoints (Uncomment the following to save checkpoints while training)
    if (epoch + 1) % 10 == 0:
        save_checkpoint(gen, opt_gen, path=os.path.join(CHECKPOINT_GEN, f"generator_RockGAN_{epoch}.pt"))
        save_checkpoint(critic, opt_critic, path=os.path.join(CHECKPOINT_CRITIC, f"critic_RockGAN_{epoch}.pt"))
