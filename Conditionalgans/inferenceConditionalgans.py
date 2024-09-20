import os
import torch
from torch import nn
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
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
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def combine_vectors(x, y):
    return torch.cat((x.float(), y.float()), 1)

def generate_images_for_all_classes(generator, z_dim, n_classes, output_file):
    images = []
    
    for class_label in range(n_classes):
        noise = torch.randn(1, z_dim)
        one_hot_label = torch.zeros(1, n_classes)
        one_hot_label[0][class_label] = 1
        noise_and_label = combine_vectors(noise, one_hot_label)

        with torch.no_grad():
            fake_image = generator(noise_and_label)
            images.append(fake_image)

    # Stack all images together and create a grid of images (nrow=5 creates a 5x2 grid for 10 images)
    images_tensor = torch.cat(images)
    save_image((images_tensor + 1) / 2, output_file, nrow=5)  # Normalize to [0,1] and save

if __name__ == "__main__":
    z_dim = 64
    n_classes = 10
    checkpoint_dir = '/home/ep23btech11012.phy.iith/gans/Conditionalgans/save_epochs/gen'  # Folder where the checkpoints are stored
    results_dir = '/home/ep23btech11012.phy.iith/gans/Conditionalgans/result'         # Folder to save the generated images

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Loop through the checkpoints, e.g., 1, 11, 21,..., 2341
    for epoch in range(1, 2342, 10):
        checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')

        if os.path.exists(checkpoint_path):
            # Load the generator model for the current epoch
            generator = Generator(input_dim=z_dim + n_classes)
            generator.load_state_dict(torch.load(checkpoint_path))
            generator.eval()

            # Create a result folder for this epoch
            epoch_results_dir = os.path.join(results_dir, f'epoch_{epoch}_result')
            if not os.path.exists(epoch_results_dir):
                os.makedirs(epoch_results_dir)

            # Generate and save one image with all numbers (0 to 9) in a grid
            output_file = os.path.join(epoch_results_dir, f'epoch_{epoch}_all_classes.png')
            generate_images_for_all_classes(generator, z_dim, n_classes, output_file=output_file)

            print(f"Generated image for epoch {epoch} saved in {output_file}")
        else:
            print(f"Checkpoint {checkpoint_path} not found.")
