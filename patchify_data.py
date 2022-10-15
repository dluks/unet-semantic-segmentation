#!/usr/bin/env python3
# %%
import glob
import os
from patchify import patchify
import tifffile as tiff

PATCH_SIZE = 512
DATA_DIR = "../../data/watershed/"

# Unpatchified directories
RGBI_IN = os.path.join(DATA_DIR, "rgbi/loose/")
LABELS_IN = os.path.join(DATA_DIR, "labels/loose")

# Patchified directories
RGBI_OUT = os.path.join(DATA_DIR, "rgbi/loose")
LABELS_OUT = os.path.join(DATA_DIR, "labels/loose")


def patchify_data(rgb_in, labels_in, rgb_out, labels_out, patch_size=512):
    rgbs = glob.glob(os.path.join(rgb_in, "*.tif"))
    labels = glob.glob(os.path.join(labels_in, "*.tif"))
    rgbs.sort()
    labels.sort()

    # Append patch_size directory to output directory
    rgb_out = os.path.join(rgb_out, str(patch_size))
    labels_out = os.path.join(labels_out, str(patch_size))

    # Make output dirs if they don't exist
    if not os.path.exists(rgb_out):
        os.makedirs(rgb_out)
    if not os.path.exists(labels_out):
        os.makedirs(labels_out)

    for rgb, label in zip(rgbs, labels):
        rgb_name = rgb.split("/")[-1].split(".tif")[0]
        label_name = label.split("/")[-1].split(".tif")[0]
        rgb = tiff.imread(rgb)
        label = tiff.imread(label)

        patches_train = patchify(
            rgb,
            (patch_size, patch_size, rgb.shape[-1]),
            step=patch_size,
        )

        patches_label = patchify(
            label,
            (patch_size, patch_size),
            step=patch_size,
        )

        for i in range(patches_train.shape[0]):
            for j in range(patches_train.shape[1]):
                tiff.imwrite(
                    os.path.join(rgb_out, f"{rgb_name}_{i}_{j}.tif"),
                    patches_train[i, j, 0, :, :, :],
                )
                tiff.imwrite(
                    os.path.join(labels_out, f"{label_name}_{i}_{j}.tif"),
                    patches_label[i, j, :, :],
                )


# %%
patchify_data(
    RGBI_IN,
    LABELS_IN,
    RGBI_OUT,
    LABELS_OUT,
    PATCH_SIZE,
)
