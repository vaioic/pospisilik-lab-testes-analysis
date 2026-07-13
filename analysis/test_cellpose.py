from pathlib import Path

import numpy as np
import oic_toolkit
import openslide
import pandas as pd
import skimage
import tifffile
from cellpose import models

file = Path(r"../data/271491.svs")

output_dir = Path(
    r"D:\Projects\pospisilik-lab-testes-analysis\processed\20260712 Dev\Tubules_Tissue"
)

output_dir.mkdir(parents=True, exist_ok=True)

slide = openslide.OpenSlide(file)
print(f"Total levels available: {slide.level_count}")
print(f"Dimensions at Level 0 (highest res): {slide.dimensions}")
print(f"Dimensions across all levels: {slide.level_dimensions}")
print(f"Downsample factors per level: {slide.level_downsamples}")

mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))

segment = False

ds_level = 2

curr_ds_factor = int(slide.level_downsamples[ds_level])

downsampled_patch = slide.read_region(
    (0, 0), ds_level, slide.level_dimensions[ds_level]
)
downsampled_patch = np.array(downsampled_patch.convert("RGB"))

print(downsampled_patch.dtype)
exit()

# print(downsampled_patch.shape)
# print(f"Downsample: {slide.level_downsamples[ds_level]}")

# Get patches
model = models.CellposeModel(gpu=True)

# Get patches
roi_list = oic_toolkit.display.get_ROI(downsampled_patch, downsample_factor=None)

imgs = []
imgs_ds = []

for roi in roi_list:
    x = roi["xmin"] * curr_ds_factor
    y = roi["ymin"] * curr_ds_factor
    width = (roi["xmax"] - roi["xmin"]) * curr_ds_factor
    height = (roi["ymax"] - roi["ymin"]) * curr_ds_factor

    # Read the full size image
    image = slide.read_region((x, y), 0, (width, height))
    image = np.array(image.convert("RGB"))

    imgs.append(image)

    # Convert the image to grayscale and downsample
    image_ds = image[::curr_ds_factor, ::curr_ds_factor, :]
    image_gray = skimage.color.rgb2gray(image_ds)
    image_gray_filt = skimage.filters.gaussian(image_gray, sigma=2)

    imgs_ds.append(image_gray_filt)

masks, _, _ = model.eval(imgs_ds, flow_threshold=0.4, cellprob_threshold=0.0)

all_data = []

for idx, mask in enumerate(masks):
    curr_img = imgs[idx]

    # Resize the image
    mask_full = skimage.transform.resize(
        mask,
        curr_img.shape[:2],
        preserve_range=True,
        order=0,
        anti_aliasing=False,
    )

    # Convert to HED
    curr_img_hed = skimage.color.rgb2hed(curr_img)

    ## TODO: Count number of brown cells

    # plt.imshow(curr_img)
    # plt.show()
    # plt.close()

    # plt.imshow(curr_img_hed[:, :, 2])
    # plt.show()
    # plt.close()

    # Write data
    tifffile.imwrite(output_dir / f"image_roi{idx:02d}.tif", curr_img, compress="lzw")
    tifffile.imwrite(output_dir / f"mask_roi{idx:02d}.tif", mask_full, compress="lzw")

    print(curr_img.shape)
    print(mask_full.shape)

    # Export the overlay
    ov = skimage.segmentation.mark_boundaries(
        curr_img.astype(np.uint8), mask_full.astype(np.uint8)
    )

    skimage.io.imsave(
        output_dir / f"overlay_roi{idx:02d}.jpg", skimage.util.img_as_ubyte(ov)
    )

    # Measure properties
    props = skimage.measure.regionprops_table(
        mask_full, properties=("label", "eccentricity", "area")
    )

    roi = roi_list[idx]

    x = roi["xmin"] * curr_ds_factor
    y = roi["ymin"] * curr_ds_factor
    width = (roi["xmax"] - roi["xmin"]) * curr_ds_factor
    height = (roi["ymax"] - roi["ymin"]) * curr_ds_factor

    df_current = pd.DataFrame(props)
    df_current["Image"] = file.name
    df_current["ROI"] = idx
    df_current["ROI_xmin"] = x
    df_current["ROI_ymin"] = y
    df_current["ROI_width"] = width
    df_current["ROI_height"] = height

    df_current["area_micron"] = df_current["area"] * mpp

    all_data.append(df_current)

combined_df = pd.concat(all_data, ignore_index=True)

front_cols = ["Image", "ROI", "ROI_xmin", "ROI_ymin", "ROI_width", "ROI_height"]
other_cols = [col for col in combined_df.columns if col not in front_cols]

new_column_order = front_cols + other_cols

combined_df = combined_df[new_column_order]

combined_df.to_csv(output_dir / "data.csv", index=False)


# TODO: Convert units, Export data

# plt.imshow(image_gray)
# plt.show()
# exit()

# - Area
#     - Eccentricity
#     - Total area of tissue
#     - Number of dark brown cells along the edge


# if segment:
#     print(f"Total levels available: {slide.level_count}")
#     print(f"Dimensions at Level 0 (highest res): {slide.dimensions}")
#     print(f"Dimensions across all levels: {slide.level_dimensions}")
#     print(f"Downsample factors per level: {slide.level_downsamples}")

#     ROI = [(178, 2000), (1604, 3718)]

#     image_gray_filt = skimage.filters.gaussian(image_gray)

#     model = models.CellposeModel(gpu=True)
#     masks, flows, styles = model.eval(
#         image_gray_filt, flow_threshold=0.4, cellprob_threshold=0.0
#     )

# else:
#     print("Read mask")
#     masks = skimage.io.imread("masks.tif")

# # skimage.io.imsave("masks.tif", masks)

# # print(np.max(masks))
# # print(masks.dtype)

# # Expand to full image
# # masks_full = skimage.transform.resize(
# #     masks,
# #     (slide.level_dimensions[0][1], slide.level_dimensions[0][0]),
# #     preserve_range=True,
# #     order=0,
# #     anti_aliasing=False,
# # )

# # skimage.io.imsave("masks_full.tif", masks_full.astype(np.uint8))

# mask_full = skimage.io.imread("masks_full.tif")

# print(mask_full.shape)

# image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
# image = np.array(image.convert("RGB"))

# print(image.shape)

# # # Generate an overlay
# ov = skimage.segmentation.mark_boundaries(image, mask_full)

# skimage.io.imsave("overlay.jpg", ov)

# # fig = plt.figure(figsize=(12, 4))
# # plt.imshow(ov)
# # plt.tight_layout()
# # plt.show()
