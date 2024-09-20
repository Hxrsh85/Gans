import torch
import torchvision
import os
import torch.nn as nn

# Define the Generator class (unchanged)
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim, device=device)

# Set device (change to 'cuda' if using GPU)
device = 'cuda'

# Load the generator model
z_dim = 64  # Your z_dim value
num_images = 25  # Number of images to generate

# Loop over every 10th epoch from 1 to 490
for epoch in range(0, 491, 10):
    print(f"Generating images for epoch {epoch}...")

    # Initialize the generator and load the corresponding model weights
    gen = Generator(z_dim=z_dim).to(device)
    model_path = f'/home/ep23btech11012.phy.iith/gans/Wgans/save_epochs/gen/generator_epoch_{epoch}.pth'
    gen.load_state_dict(torch.load(model_path, map_location=device))

    # Set the generator to evaluation mode
    gen.eval()

    # Generate noise and create fake images
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise)

    # Define the output path for the current epoch
    output_path = f'/home/ep23btech11012.phy.iith/gans/Wgans/results/epoch_{epoch}/'
    os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the generated images
    output_file = os.path.join(output_path, f'generated_images_epoch_{epoch}.png')
    torchvision.utils.save_image((fake_images + 1) / 2, output_file, nrow=5)

    print(f"Generated images for epoch {epoch} saved to {output_file}")
