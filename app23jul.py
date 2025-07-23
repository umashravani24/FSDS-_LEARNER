import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Helper function to load image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# cow image URL
cow_url = "https://www.shutterstock.com/image-photo/white-cow-calf-lying-pasture-260nw-2365649603.jpg"

# Load  cow image
cow = load_image_from_url(cow_url)

# Display original image
plt.figure(figsize=(6, 4))
plt.imshow(cow)
plt.title("Cow Calf")
plt.axis("off")
plt.show()

# Convert to NumPy array and print shape
cow_np = np.array()
print("Cow Calf image shape:", cow_np.shape)

# Convert to grayscale
cow_gray = cow.convert("L")

# Display grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(cow_gray, cmap="gray")
plt.title("Cow Calf (Grayscale)")
plt.axis("off")
plt.show()
