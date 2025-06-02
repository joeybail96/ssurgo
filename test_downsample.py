import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt

# Parameters
Resolution_Factor = 0.5

# Define the image
Image = np.array([[0, 10, 20, 30],
                  [40, 50, 60, 70]])

# Ensure the image is in the proper uint8 format
Image = np.uint8(Image)

# Get the image dimensions
Image_Height, Image_Width = Image.shape

# Resize the image using the resolution factor
Resized_Image = np.array(PILImage.fromarray(Image).resize(
    (round(Image_Width * Resolution_Factor), round(Image_Height * Resolution_Factor)),
    PILImage.NEAREST))

# Adjust the resolution by repeating elements
Resolution_Adjusted_Image = np.kron(Resized_Image, np.ones((round(1 / Resolution_Factor), round(1 / Resolution_Factor))))

# Plot the original and pseudo-resolution adjusted images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(Image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(Resolution_Adjusted_Image, cmap='gray')
axes[1].set_title("Pseudo-Resolution Adjusted Image (kept pixel quality)")
axes[1].axis('off')

plt.show()

# Print the sizes of the images
print("The size of the original image is:", Image.shape)
print("The size of the pseudo-resolution adjusted image is:", Resolution_Adjusted_Image.shape)
