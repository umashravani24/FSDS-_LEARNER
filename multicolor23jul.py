import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Load Cow image
cow_url = "https://www.shutterstock.com/image-photo/white-cow-calf-lying-pasture-260nw-2365649603.jpg"
cow = load_image_from_url(cow_url).convert("RGB")
cow_np = np.array(cow)

# Split RGB channels
R, G, B = cow_np[:, :, 0], cow_np[:, :, 1], cow_np[:, :, 2]

# Create images emphasizing each channel
red_img = np.zeros_like(cow_np)
green_img = np.zeros_like(cow_np)
blue_img = np.zeros_like(cow_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display original and RGB color-emphasized images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cow_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(red_img)
plt.title("Red Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(green_img)
plt.title("Green Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blue_img)
plt.title("Blue Channel Emphasis")
plt.axis("off")

plt.tight_layout()
plt.show()

# Optional: Apply a colormap to grayscale
cow_gray = cow.convert("L")
cow_gray_np = np.array(cow_gray)

plt.figure(figsize=(6, 5))
plt.imshow(cow_gray_np, cmap="viridis")  # Change cmap to 'hot', 'cool', etc.
plt.title("Colormapped Grayscale")
plt.axis("off")
plt.colorbar()
plt.show()
