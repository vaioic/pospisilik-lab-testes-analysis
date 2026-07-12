import numpy as np
import skimage
from matplotlib import pyplot as plt
from openslide import OpenSlide
from scipy.ndimage import binary_fill_holes

from shared import core_logic

file = "../data/271491.svs"

slide = OpenSlide(file)

print(f"Total levels available: {slide.level_count}")
print(f"Dimensions at Level 0 (highest res): {slide.dimensions}")
print(f"Dimensions across all levels: {slide.level_dimensions}")
print(f"Downsample factors per level: {slide.level_downsamples}")

ROI = [(178, 2000), (1604, 3718)]

downsampled_patch = slide.read_region((0, 0), 2, slide.level_dimensions[2])

downsampled_patch = np.array(downsampled_patch.convert("RGB"))

print(f"Shape: {downsampled_patch.shape}")
# Convert the image to grayscale

image_gray = skimage.color.rgb2gray(downsampled_patch)

image_gray_filt = skimage.filters.gaussian(image_gray)

mask = image_gray_filt < 0.83

mask = skimage.morphology.opening(mask, skimage.morphology.disk(2))

# Try to identify the whole tissue
tissue_mask = skimage.morphology.closing(mask, skimage.morphology.disk(10))
tissue_mask = binary_fill_holes(tissue_mask)

tissue_mask = skimage.morphology.opening(tissue_mask, skimage.morphology.disk(2))
tissue_mask = skimage.morphology.remove_small_objects(tissue_mask, max_size=10000)

# Label the tissue
tissue_label = skimage.measure.label(tissue_mask)

# Loop through each tissue to identify individual cells
curr_tissue_id = 3

curr_tissue_mask = mask
curr_tissue_mask[tissue_label != curr_tissue_id] = False

# Try inverting to find centers for thresholding
cell_centers = ~curr_tissue_mask
cell_centers[tissue_label != curr_tissue_id] = False


# (178, 2000, 1604 3718)
# plt.imshow(image_gray_filt, cmap="gray")
plt.imshow(cell_centers)
plt.show()


# downsampled_patch.show()


def main():
    print(core_logic.main())
