# Running code for the "brown" image

# Try reading in the brown image and see if the same (or similar) threshold would work
# for it
import numpy as np
import openslide

file = "../data/271490.svs"

slide = openslide.OpenSlide(file)
ds_level = 2

# image_ds = slide.read_region((0, 0), ds_level, slide.level_dimensions[ds_level])
# image_ds_rgb = np.array(image_ds.convert("RGB"))

# plt.imshow(image_ds_rgb)
# plt.show()
# exit()

ROI = [642, 3620, 1050, 3894]


curr_ds_factor = int(slide.level_downsamples[ds_level])

x = ROI[0] * curr_ds_factor
y = ROI[1] * curr_ds_factor
width = (ROI[2] - ROI[0]) * curr_ds_factor
height = (ROI[3] - ROI[1]) * curr_ds_factor
# width = 4096
# height = 4096

image = slide.read_region((x, y), 0, (width, height))
image_rgb = np.array(image.convert("L"))
# plt.imshow(image_rgb)
# plt.show()
# exit()

# image_rgb = np.array(image_ds.convert("RGB"))

# curr_img_hed = skimage.color.rgb2hed(image_rgb)
# img_cells = curr_img_hed[..., 2]


# plt.imshow(img_cells)
# plt.show()
# exit()
core.process_image(r"../data/271490.svs", "../processed/20260713/271490", is_brown=True)

core.process_image(r"../data/271492.svs", "../processed/20260713/271492", is_brown=True)


# mask_stemcell = image_rgb < 40
# # print(mask_stemcell.shape)
# mask_stemcell = skimage.morphology.opening(mask_stemcell, skimage.morphology.disk(2))

# mask_stemcell = skimage.morphology.remove_small_holes(mask_stemcell, max_size=50)
# mask_stemcell = skimage.morphology.remove_small_objects(mask_stemcell, max_size=400)

# ov = skimage.segmentation.mark_boundaries(image_rgb, mask_stemcell)

# plt.imshow(ov)
# plt.show()
