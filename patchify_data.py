#!/usr/bin/env python3
# %%
import glob
import os
from patchify import patchify
import tifffile as tiff

PATCH_SIZE = 512
DATA_DIR = "../../data/watershed/"

# Unpatchified directories
IMG_IN = os.path.join(DATA_DIR, f"rgbi/strict/simple_imp")

# Patchified directories
IMG_OUT = IMG_IN


def patchify_data(img_in, img_out, patch_size=512):
    imgs = glob.glob(os.path.join(img_in, "*.tif"))
    imgs.sort()

    # Append patch_size directory to output directory
    img_out = os.path.join(img_out, str(patch_size))

    # Make output dirs if they don't exist
    if not os.path.exists(img_out):
        os.makedirs(img_out)

    for img in imgs:
        img_name = img.split("/")[-1].split(".tif")[0]
        img = tiff.imread(img)

        if len(img.shape) == 3:
            patch_dims = (patch_size, patch_size, img.shape[-1])
        else:
            patch_dims = (patch_size, patch_size)

        patches = patchify(
            img,
            patch_dims,
            step=patch_size,
        )

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                if len(patch_dims) == 3:
                    tiff.imwrite(
                        os.path.join(img_out, f"{img_name}_{i}_{j}.tif"),
                        patches[i, j, 0, :, :, :],
                    )
                else:
                    tiff.imwrite(
                        os.path.join(img_out, f"{img_name}_{i}_{j}.tif"),
                        patches[i, j, :, :],
                    )


# %%
patchify_data(
    IMG_IN,
    IMG_OUT,
    PATCH_SIZE,
)
