import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="The Queen of kakathiya Empire Image Processor", layout="wide")

# Title
st.title("Rani Rudrama Devi Image - Multi-Color Channel Visualizer")

# Load image from local path
@st.cache_data
def load_image():
    path = r"C:\Users\sss\Downloads\rudrama.jpg"
    return Image.open(path).convert("RGB")

# Load and display image
ran = load_image()
st.image(ran, caption="Original my Image", use_container_width=True)

# Convert to NumPy array
ran_np = np.array(ran)
R, G, B = ran_np[:, :, 0], ran_np[:, :, 1], ran_np[:, :, 2]

# Create channel images
red_img = np.zeros_like(ran_np)
green_img = np.zeros_like(ran_np)
blue_img = np.zeros_like(ran_np)

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

ran_gray = ran.convert("L")
ran_gray_np = np.array(ran_gray)

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(ran_gray_np, cmap=colormap)
ax.axis("off")
st.pyplot(fig)
