import os
import glob
import numpy as np
import tifffile as tiff
from sklearn.impute import SimpleImputer, KNNImputer

DATA_DIR = "../../data/watershed/"
MODE = "simple_imp"

rgb_in = glob.glob(os.path.join(DATA_DIR, "rgbi/loose/*.tif"))
loose_in = glob.glob(os.path.join(DATA_DIR, "labels/loose/*.tif"))
strict_in = glob.glob(os.path.join(DATA_DIR, "labels/strict/*.tif"))
out_dir = os.path.join(DATA_DIR, f"rgbi/strict/{MODE}")

rgb_in.sort()
loose_in.sort()
strict_in.sort()

assert MODE in ["mean", "simple_imp", "knn_imp"], "Mode is invalid."
for rgb, loose, strict in zip(rgb_in, loose_in, strict_in):
    # Get the filename
    fn = rgb.split("/")[-1].split(".tif")[0]
    # Load rgb and loose and strict labels
    rgb = tiff.imread(rgb)
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
    elif MODE == "simple_impute":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        for i in range(rgb.shape[-1]):
            # First replace all "masked" pixels with nans
            rgb[idx[0], idx[1], i] = np.nan
            rgb[..., i] = imputer.fit_transform(rgb[..., i])
        # Convert datatype back to int
        rgb = rgb.astype(int)
    elif MODE == "knn_impute":
        imputer = KNNImputer(missing_values=np.nan)
        for i in range(rgb.shape[-1]):
            # First replace all "masked" pixels with nans
            rgb[idx[0], idx[1], i] = np.nan
            rgb[..., i] = imputer.fit_transform(rgb[..., i])
        # Convert datatype back to int
        rgb = rgb.astype(int)

    # Save rgb
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tiff.imwrite(os.path.join(out_dir, f"{fn}_masked_{MODE}.tif"), rgb)
