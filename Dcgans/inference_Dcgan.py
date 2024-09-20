import torch
import torchvision
import os
import torch.nn as nn

# Define the Generator class as provided
class Generator(nn.Module):
    def __init__(self, z_dim=64, im_chan=1, hidden_dim=64):
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
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

# Define the get_noise function
def get_noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim, device=device)

# Set device (change to 'cuda' if using GPU)
device = 'cuda'

# Load the generator model
z_dim = 64  # Your z_dim value
gen = Generator(z_dim=z_dim).to(device)
model_path = '/home/ep23btech11012.phy.iith/gans/Dcgans/save_epochs/gen/generator_epoch_10.pth'
gen.load_state_dict(torch.load(model_path, map_location=device))

# Set the generator to evaluation mode
gen.eval()

# Generate images
num_images = 25  # Number of images to generate
noise = get_noise(num_images, z_dim, device=device)
fake_images = gen(noise)

# Define the output path
output_path = '/home/ep23btech11012.phy.iith/gans/Dcgans/results/epoch10'
os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the generated images
output_file = os.path.join(output_path, 'generated_images.png')
torchvision.utils.save_image((fake_images + 1) / 2, output_file, nrow=5)

print(f"Generated images saved to {output_file}")
