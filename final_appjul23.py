import streamlit as st 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="National Emblem Image Processor", layout="wide")

# Title
st.title("National Emblem Image - Multi-Color Channel Visualizer")

# Load image from URL
@st.cache_data 
def load_image():
    url = "https://d2jnu6hkti1tqv.cloudfront.net/upload/0c4f9cb7-8ffc-469f-ba7d-13249897008f.jpg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Load and display image
emb = load_image()
st.image(emb, caption="Original National Emblem Image", use_container_width=True)

# Convert to NumPy array
emb_np = np.array(emb)
R, G, B = emb_np[:, :, 0], emb_np[:, :, 1], emb_np[:, :, 2]

# Create channel images
red_img = np.zeros_like(emb_np)
green_img = np.zeros_like(emb_np)
blue_img = np.zeros_like(emb_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display RGB channels
st.subheader("RGB Channel Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with col3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

# Grayscale + Colormap
st.subheader("Colormapped Grayscale Image")

colormap = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

emb_gray = emb.convert("L")
emb_gray_np = np.array(emb_gray)

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(emb_gray_np, cmap=colormap)
plt.axis("off")

# DO NOT USE: plt.show()
# USE THIS INSTEAD:
st.pyplot(fig)
