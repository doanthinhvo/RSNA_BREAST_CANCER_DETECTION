import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

transform = torchvision.transforms.RandomResizedCrop((128, 128), scale=(1, 1), ratio=(1, 1))

# Load an example image
img = Image.open("data/archive/5/640805896.png")
# show image 
plt.imshow(img)
plt.show()


# Apply the transform to the image
img_transformed = transform(img)

# Convert the transformed image to a tensor
img_tensor = torchvision.transforms.ToTensor()(img_transformed)

img_tensor = img_tensor.permute(1, 2, 0)
plt.imshow(img_tensor.numpy())
plt.show()

# Display the transformed image
# torchvision.utils.save_image(img_tensor, "example_transformed_image.jpg")
