from openslide import OpenSlide

file = "../data/271491.svs"

slide = OpenSlide(file)

with open("tmp_metadat.txt", "w", encoding="utf-8") as f:
    f.write(f"Metadata for slide: {file}\n")
    f.write("=" * 50 + "\n\n")

    # 3. Loop through all metadata key-value pairs and write them to the file
    for key, value in slide.properties.items():
        f.write(f"{key}: {value}\n")

# print(f"Original image bit depth: {slide.properties.get('tiff.BitsPerSample')}")

# downsampled_patch = slide.read_region((0, 0), 2, slide.level_dimensions[2])


# downsampled_patch = np.array(downsampled_patch.convert("RGB"))

# print(f"After conversion: {downsampled_patch.dtype}")
