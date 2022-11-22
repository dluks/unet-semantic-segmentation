import os
import glob
import numpy as np
import tifffile as tiff
from sklearn.impute import SimpleImputer, KNNImputer


def impute_mask(MODE, rgb_in, loose_in, strict_in, out_dir):
    """
    Takes unimputed RGB(I) tifs and returns new tifs in which "bad" label locations
    have been filled with imputed value.

    Args:
        MODE (str): Imputation type
        rgb_in (str): Location of RGB(I) tifs that will be imputed
        loose_in (str): Location of "loose" label tifs. These will be compared with
            strict_in labels to determine the "bad" masks
        strict_in (str): Location of "strict" label tifs. These will correspond to
            the final labels to be used in training.
        out_dir (str): Directory in which the imputed images will be saved.
    """
    rgb_in = glob.glob(os.path.join(rgb_in, "*.tif"))
    loose_in = glob.glob(os.path.join(loose_in, "*.tif"))
    strict_in = glob.glob(os.path.join(strict_in, "*.tif"))
    out_dir = os.path.join(out_dir, MODE)

    rgb_in.sort()
    loose_in.sort()
    strict_in.sort()

    assert MODE in ["mean", "simple_imp", "knn_imp"], "Mode is invalid."
    for rgb, loose, strict in zip(rgb_in, loose_in, strict_in):
        # Get the filename
        fn = rgb.split("/")[-1].split(".tif")[0]
        # Load rgb and loose and strict labels
        rgb = tiff.imread(rgb).astype(np.float32)
        loose = tiff.imread(loose)
        strict = tiff.imread(strict)
        # Find where loose labels exist but not strict
        idx = np.where((loose > 0) & (strict == 0))
        # Impute values
        if MODE == "mean":
            # Get the mean values of each channel
            rgb_mns = [rgb[..., j].mean() for j in range(rgb.shape[2])]
            # Replace values of "bad" labels with the channel mean
            for k, m in enumerate(rgb_mns):
                rgb[idx[0], idx[1], k] = m
        elif MODE == "simple_imp":
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
            for i in range(rgb.shape[-1]):
                # First replace all "masked" pixels with nans
                rgb[idx[0], idx[1], i] = np.nan
                rgb[..., i] = imputer.fit_transform(rgb[..., i])
            # Convert datatype back to int
            rgb = rgb.astype(np.uint8)
        elif MODE == "knn_imp":
            imputer = KNNImputer(missing_values=np.nan)
            for i in range(rgb.shape[-1]):
                # First replace all "masked" pixels with nans
                rgb[idx[0], idx[1], i] = np.nan
                rgb[..., i] = imputer.fit_transform(rgb[..., i])
            # Convert datatype back to int
            rgb = rgb.astype(np.uint8)

        # Save rgb
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        tiff.imwrite(os.path.join(out_dir, f"{fn}_masked_{MODE}.tif"), rgb)


DATA_DIR = "../../data/watershed/"
MODE = "simple_imp"
rgb_in = os.path.join(DATA_DIR, "rgbi/loose")
loose_in = os.path.join(DATA_DIR, "labels/loose")
strict_in = os.path.join(DATA_DIR, "labels/strict")
out_dir = os.path.join(DATA_DIR, "rgbi/strict")

impute_mask(MODE, rgb_in, loose_in, strict_in, out_dir)
