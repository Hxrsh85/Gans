import numpy as np
import matplotlib.pyplot as plt

# Load the saved losses from the .npz file
data = np.load("/home/ep23btech11012.phy.iith/gans/Wgans/save_epochs/loses/Wgans.npz")
train_losses = data['generator_losses']
val_losses = data['critic_losses']

# Plot the losses
plt.plot(train_losses, label='generator Loss')
plt.plot(val_losses, label='critic Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
