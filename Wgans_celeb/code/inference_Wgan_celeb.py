import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import os


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameters
NOISE_DIM = 100  # Size of the noise vector
batch_size = 32  # Number of images to generate
device = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES_GEN = 64
CHANNELS_IMG = 3  # Assuming you're working with RGB CelebA dataset

# Generate fixed noise for consistent images
fixed_noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)


generator = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    # Use DataParallel for multi-GPU inference
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)

    # Load the corresponding saved model
generator_file = f"/scratch/ep23btech11012.phy.iith/Wgans_celeb/save_gen/generator_epoch_1.pth"
generator.load_state_dict(torch.load(generator_file, map_location=device))
generator.eval()  # Set to evaluation mode

    # Generate fake images using the fixed noise
with torch.no_grad():
    fake_images = generator(fixed_noise).reshape(-1, CHANNELS_IMG, 64, 64)  # Reshape to 64x64 image format

    # Normalize and save generated images as a grid
output_dir = "/scratch/ep23btech11012.phy.iith/Wgans_celeb/results/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"generated_image_grid_1.png")
    
    # Save the image grid (8 rows, 4 columns)
save_image(fake_images, output_file, normalize=True, nrow=8)

print(f"Generated images saved at {output_file}")

# Load saved generator model and generate images for each saved epoch
for i in range(20, 501, 20):

    # Initialize the generator
    generator = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    # Use DataParallel for multi-GPU inference
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)

    # Load the corresponding saved model
    generator_file = f"/scratch/ep23btech11012.phy.iith/Wgans_celeb/save_gen/generator_epoch_{i}.pth"
    generator.load_state_dict(torch.load(generator_file, map_location=device))
    generator.eval()  # Set to evaluation mode

    # Generate fake images using the fixed noise
    with torch.no_grad():
        fake_images = generator(fixed_noise).reshape(-1, CHANNELS_IMG, 64, 64)  # Reshape to 64x64 image format

    # Normalize and save generated images as a grid
    output_dir = "/scratch/ep23btech11012.phy.iith/Wgans_celeb/results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generated_image_grid_{i}.png")
    
    # Save the image grid (8 rows, 4 columns)
    save_image(fake_images, output_file, normalize=True, nrow=8)

    print(f"Generated images saved at {output_file}")
