import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Load the pre-trained model ---
model = deeplabv3_resnet101(pretrained=True).eval()

# --- Load the image ---
img_path = "sample.jpg"
img = Image.open(img_path).convert("RGB")

# --- Preprocess the image ---
preprocess = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0)

# --- Pass the image through the model ---
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0).byte().cpu().numpy()

# --- Generate an overlay with segmentation result ---
image_np = np.array(img.resize((512, 512)))
overlay = image_np.copy()

# Use a color map for visualizing the segmentation
cmap = plt.get_cmap('tab20')  # Choose any colormap

# Assign colors based on the segmentation output
segmentation_colored = np.zeros_like(image_np)

# Loop through each unique class in the output and apply a color
for class_id in np.unique(output_predictions):
    if class_id == 0:  # Skip background
        continue
    mask = np.uint8(output_predictions == class_id)
    
    # Get the color for this class using the colormap
    color = np.array(cmap(class_id / 20)[:3]) * 255  # Normalize and scale to 0-255
    color = color.astype(np.uint8)
    
    # Apply the color to the segmentation area
    segmentation_colored[mask == 1] = color

# --- Display the result ---
plt.figure(figsize=(10, 6))
plt.title("Semantic Segmentation")
plt.imshow(segmentation_colored)
plt.axis('off')
plt.show()
