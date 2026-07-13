import skimage
from matplotlib import pyplot as plt

image = skimage.io.imread(
    r"D:\Projects\pospisilik-lab-testes-analysis\processed\20260712 Dev\Tubules_Tissue\\image_roi00.tif"
)

print(image.shape)
print(image.dtype)

image = image[::2, ::2, :]

image_hed = skimage.color.rgb2hed(image)

# plt.imshow(image_hed[..., 2])
# plt.show()
# plt.close()
# exit()
# # Try to match by color

# mask = oic_toolkit.segment.match_color(image, (71, 39, 26), radius=10)

# mask = skimage.morphology.remove_small_holes(mask, max_size=30)

# mask = skimage.morphology.opening(mask, skimage.morphology.disk(2))

mask = image_hed[..., 2] > 0.05
mask = skimage.morphology.opening(mask, skimage.morphology.disk(2))

mask = skimage.morphology.remove_small_holes(mask, max_size=50)
mask = skimage.morphology.remove_small_objects(mask, max_size=5)

ov = skimage.segmentation.mark_boundaries(image, mask)

plt.imshow(ov)
plt.show()
