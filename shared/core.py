from pathlib import Path

import numpy as np
import oic_toolkit
import openslide
import pandas as pd
import skimage
import tifffile
from cellpose import models


def process_image(file, output_path, ds_level=2):

    if not isinstance(file, Path):
        file = Path(file)

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # Create a Slide object
    slide = openslide.OpenSlide(file)

    # Get pixel size
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))

    # Read in a downsampled region for ROI selection
    image_ds = slide.read_region((0, 0), ds_level, slide.level_dimensions[ds_level])
    curr_ds_factor = int(slide.level_downsamples[ds_level])

    # Allow the user to grab ROIs
    roi_list = oic_toolkit.display.get_ROI(image_ds, downsample_factor=None)

    imgs = []
    imgs_ds = []

    bbox = []

    for roi in roi_list:
        x = roi["xmin"] * curr_ds_factor
        y = roi["ymin"] * curr_ds_factor
        width = (roi["xmax"] - roi["xmin"]) * curr_ds_factor
        height = (roi["ymax"] - roi["ymin"]) * curr_ds_factor

        bbox.append((x, y, width, height))

        # Read the full size image
        image = slide.read_region((x, y), 0, (width, height))
        image_rgb = np.array(image.convert("RGB"))

        imgs.append(image_rgb)

        image_gray = np.array(image.convert("L"))
        image_gray = image_gray[::curr_ds_factor, ::curr_ds_factor]

        # Filter the image for segmentation later
        image_gray_filt = skimage.filters.gaussian(image_gray, sigma=2)

        imgs_ds.append(image_gray_filt)

    # Use Cellpose to segment the tubules
    model = models.CellposeModel(gpu=True)
    masks, _, _ = model.eval(imgs_ds, flow_threshold=0.4, cellprob_threshold=0.0)

    # Analyze data
    all_data = []

    for idx, mask in enumerate(masks):
        curr_img = imgs[idx]

        # Resize the mask back to the full size
        mask_full = skimage.transform.resize(
            mask,
            curr_img.shape[:2],
            preserve_range=True,
            order=0,
            anti_aliasing=False,
        )

        # Export images and mask
        tifffile.imwrite(
            output_path / f"image_roi{idx:02d}.tif",
            curr_img.astype(np.uint8),
            compression="lzw",
        )
        tifffile.imwrite(
            output_path / f"mask_roi{idx:02d}.tif",
            mask_full.astype(np.uint8),
            compression="lzw",
        )

        # Convert to HED
        curr_img_hed = skimage.color.rgb2hed(curr_img[::4, ::4, :])
        img_cells = curr_img_hed[..., 2]

        mask_stemcell = img_cells > 0.05
        # print(mask_stemcell.shape)
        mask_stemcell = skimage.morphology.opening(
            mask_stemcell, skimage.morphology.disk(2)
        )

        mask_stemcell = skimage.morphology.remove_small_holes(
            mask_stemcell, max_size=50
        )
        mask_stemcell = skimage.morphology.remove_small_objects(
            mask_stemcell, max_size=5
        )

        mask_stemcell = skimage.transform.resize(
            mask_stemcell,
            mask_full.shape,
            preserve_range=True,
            order=0,
            anti_aliasing=False,
        )

        mask_stemcell = np.logical_and(mask_stemcell, mask_full)

        labels_stemcell = skimage.measure.label(mask_stemcell)

        # Measure properties
        props = skimage.measure.regionprops_table(
            mask_full, properties=("label", "eccentricity", "area")
        )

        stem_cell_counts = []

        for p in range(len(props["label"])):
            #
            curr_tubule_mask = mask_full == props["label"][p]

            curr_stem_cell_labels = labels_stemcell[curr_tubule_mask]

            unique_labels = np.unique(curr_stem_cell_labels)
            num_cells = len(unique_labels[unique_labels > 0])

            stem_cell_counts.append(num_cells)

        # Convert data to DataFrame and add other values

        df_current = pd.DataFrame(props)
        df_current["Image"] = file.name
        df_current["ROI"] = idx
        df_current["ROI_xmin"] = bbox[idx][0]
        df_current["ROI_ymin"] = bbox[idx][1]
        df_current["ROI_width"] = bbox[idx][2]
        df_current["ROI_height"] = bbox[idx][3]

        df_current["Num_cells"] = stem_cell_counts

        df_current["area_micron"] = df_current["area"] * (mpp**2)

        # df_current.to_csv(f"roi_{idx:02d}.csv")

        all_data.append(df_current)

        # Export the overlay
        ov = skimage.segmentation.mark_boundaries(
            curr_img.astype(np.uint8), mask_full.astype(np.uint8)
        )
        ov = skimage.segmentation.mark_boundaries(ov, labels_stemcell.astype(np.uint8))

        skimage.io.imsave(
            output_path / f"overlay_roi{idx:02d}.jpg", skimage.util.img_as_ubyte(ov)
        )

    # Combine all data and export to CSV
    combined_df = pd.concat(all_data, ignore_index=True)

    front_cols = ["Image", "ROI", "ROI_xmin", "ROI_ymin", "ROI_width", "ROI_height"]
    other_cols = [col for col in combined_df.columns if col not in front_cols]

    new_column_order = front_cols + other_cols

    combined_df = combined_df[new_column_order]

    combined_df.to_csv(output_path / "data.csv", index=False)
