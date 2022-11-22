import os
import glob
import numpy as np
import tifffile as tiff
from patchify import patchify
from sklearn.model_selection import train_test_split
import scipy.ndimage as ndimage
from skimage.segmentation import watershed
from skimage.measure import regionprops
import laspy as lp
from scipy import interpolate
from scipy.spatial import cKDTree as kdtree
import geopandas as gpd


def get_by_ext(src, ext=".tif"):
    if not ext.startswith("."):
        ext = "." + ext
    return sorted(glob.glob(os.path.join(src, f"*{ext}")))


def patch_train_label(
    raster,
    labels,
    img_size,
    channels=False,
    merge_channel=False,
    mask_class=False,
    mask_val=-1,
):
    """
    Patchifies images and labels into square patches

    Args:
        raster (str): Location of rgb(i) images
        labels (str): Location of label images
        img_size (int): Patch size (single side only)
        channels (int, optional): Number of channels in the output patches. Defaults to
            False to use all available channels.
        merge_channel (str, optional): Location of images to be merged with raster as
            extra channel. Defaults to False.
        mask_class (bool, optional): Set to True to replace messed up "ignore" mask
            value with mask_val (this should fixed in actual image masking and removed). Defaults to False.
        mask_val (int, optional): Value of the desired "ignore" mask. Defaults to -1.

    Returns:
        data_train (ndarray): Array of observations
        data_label (ndarray): Array of binarized labels
        data_label_inst (ndarray): Array of labels with classes intact
    """

    raster = get_by_ext(raster)
    labels = get_by_ext(labels)

    assert len(raster) > 0, "Raster list is empty."
    samp_rast = tiff.imread(raster[0])
    img_base_size = samp_rast.shape[0]
    n = len(raster)
    m = (img_base_size // img_size) ** 2

    if not channels:
        channels = samp_rast.shape[-1]

    if merge_channel:
        merge_channel = get_by_ext(merge_channel)
        channels += tiff.imread(merge_channel[0]).shape[-1]

    data_train = np.zeros((n * m, img_size, img_size, channels))
    data_label = np.zeros((n * m, img_size, img_size))

    for k in range(n):
        if merge_channel:
            r = np.concatenate(
                (tiff.imread(raster[k]), tiff.imread(merge_channel[k])), axis=-1
            )
        else:
            r = tiff.imread(raster[k])[..., :channels]

        # Only read in the specified number of channels from input raster
        patches_train = patchify(
            r,
            (img_size, img_size, channels),
            step=img_size,
        )
        patches_label = patchify(
            tiff.imread(labels[k]), (img_size, img_size), step=img_size
        )
        data_train[k * m : (k + 1) * m, :, :, :] = patches_train.reshape(
            -1, img_size, img_size, channels
        )
        data_label[k * m : (k + 1) * m, :, :] = patches_label.reshape(
            -1, img_size, img_size
        )

    if mask_class:
        print("Old mask class:", data_label.max())
        data_label = ((data_label > 0) & (data_label < data_label.max())).astype(
            np.uint8
        )
        data_label[np.where(data_label == data_label.max())] = mask_val
        print("New mask class:", data_label.min())
    else:
        data_label_inst = data_label
        data_label = (data_label > 0).astype("int")
    data_label = np.expand_dims(data_label, axis=-1)
    data_label_inst = np.expand_dims(data_label_inst, axis=-1)
    data_train = data_train.astype("float") / 255

    print(
        f"\nPatched data sizes:\ndata_train: {data_train.shape}\ndata_label: {data_label.shape}"
    )

    return data_train, data_label, data_label_inst


def prep_data(data_dir, hand_dir_name, ws_rgb, ws_label, seed=157, channels=3):
    # DATASET
    hand_dir = os.path.join(data_dir, hand_dir_name)
    hand_rgb_dir = os.path.join(hand_dir, "rgb")
    hand_label_dir = os.path.join(hand_dir, "label")

    if channels == 4:
        hand_nir_dir = os.path.join(hand_dir, "nir")
        data_train, data_label, data_label_inst = patch_train_label(
            hand_rgb_dir, hand_label_dir, 128, merge_channel=hand_nir_dir
        )
    else:
        data_train, data_label, data_label_inst = patch_train_label(
            hand_rgb_dir, hand_label_dir, 128
        )

    X_train, X_test, y_train, y_test, inst_train, inst_test = train_test_split(
        data_train,
        data_label,
        data_label_inst,
        test_size=0.33,
        shuffle=True,
        random_state=seed,
    )

    print(
        f"\nSizes with only hand-labeled data:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    # Patchify watershed data (pre-patchified)
    WS_DIR = os.path.join(data_dir, "watershed")
    WS_RGBI_DIR = os.path.join(WS_DIR, f"rgbi/{ws_rgb}/512")
    WS_LABEL_DIR = os.path.join(WS_DIR, f"labels/{ws_label}/512")

    data_train, data_label, data_label_inst = patch_train_label(
        WS_RGBI_DIR, WS_LABEL_DIR, 128, channels=channels, mask_class=False
    )

    # Always use the hand-labeled test split as final test (outside KF CV) because
    # we know it is higher quality
    X_train = np.concatenate((X_train, data_train), axis=0)
    y_train = np.concatenate((y_train, data_label), axis=0)
    inst_train = np.concatenate((inst_train, data_label_inst))

    pct_bg = (np.count_nonzero(y_train == 0) / len(y_train.ravel())) * 100
    pct_trees = (np.count_nonzero(y_train == 1) / len(y_train.ravel())) * 100
    pct_masked = (np.count_nonzero(y_train == -1) / len(y_train.ravel())) * 100
    print("\nWatershed percents")
    print("--------------------")
    print(f"% BG: {pct_bg:.2f}%")
    print(f"% Trees: {pct_trees:.2f}%")
    print(f"% Masked: {pct_masked:.2f}%")

    print(
        f"\nSizes after adding watershed data:\n\
    X_train: {X_train.shape}\n\
    y_train: {y_train.shape}\n\
    X_test: {X_test.shape}\n\
    y_test: {y_test.shape}"
    )

    return X_train, y_train, X_test, y_test, inst_train, inst_test


def filter_labels(labels, img, band, area, ecc, ar, abr, intensity):
    """Takes a set of labels and returns a filtered set based on regionprops parameters.

    Args:
        labels (ndarray): Watershed labels
        img (ndarray): Image to pair with labels for regionprops extraction
        band (int): The band to use for intensity in regionprops
        area (float): Minimum area of filtered regions
        ecc (float): Maximum eccentricity of filtered regions
        ar (float): Minimum ratio of minor and major axes (1=square, 0.5=rectangle)
        abr (float): Minimum ratio of area of non-zero pixels compared to bounding box area
        intensity (float): Minimum band intensity of corresponding image

    Returns:
        filtered_labels (ndarray): Array of the filtered labels
        bbox (list): List of the coordinates of each label's bounding box
    """
    if band:
        regions = regionprops(labels, img[..., band])
    else:
        regions = regionprops(labels, img)
    filtered_labels = np.zeros((labels.shape[0], labels.shape[1]), dtype=int)
    bbox = []
    for region in regions:
        if (
            region.area >= area
            and (region.axis_minor_length / region.axis_major_length >= ar)
            and (region.eccentricity <= ecc)
            and (region.area / region.area_bbox >= abr)
            and (region.intensity_mean >= intensity)
        ):
            filtered_labels[region.coords[:, 0], region.coords[:, 1]] = region.label
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            bbox.append([bx, by])

    return filtered_labels, bbox


def watershed_labels(img, neighborhood_size, threshold, min_height):
    p_smooth = ndimage.gaussian_filter(img, threshold)
    p_max = ndimage.maximum_filter(p_smooth, neighborhood_size)
    local_maxima = p_smooth == p_max
    local_maxima[img == 0] = 0
    labeled, num_objects = ndimage.label(local_maxima)
    xy = np.array(
        ndimage.center_of_mass(
            input=img, labels=labeled, index=range(1, num_objects + 1)
        )
    )
    binary_mask = np.where(img >= min_height, 1, 0)
    binary_mask = ndimage.binary_fill_holes(binary_mask).astype(int)

    labels = watershed(-img, labeled, mask=binary_mask)
    return labels, xy


def unpickle(pickle):
    a = np.zeros((len(pickle), len(pickle[0])))
    for i in range(len(pickle)):
        a[i, :] = pickle[i]
    return a


def las2chm(las_file):
    las = lp.read(las_file)
    points = las.xyz.copy()
    return_num = las.return_number.copy()
    num_of_returns = las.number_of_returns.copy()
    classification = las.classification.copy()
    select = classification != 5
    select += (return_num == 1) * (num_of_returns == 1)
    select += (return_num == 2) * (num_of_returns == 2)
    select += (return_num == 3) * (num_of_returns == 3)
    select += (return_num == 4) * (num_of_returns == 4)
    select += (return_num == 5) * (num_of_returns == 5)
    points = points[~select]
    tr = kdtree(points)
    distances, indices = tr.query(points, k=25, workers=-1)
    distances = distances[:, -1]
    thr = 2.0
    select = distances > thr
    points = points[~select]
    orginal_points = las.xyz.copy()
    tr = kdtree(orginal_points)
    distances, indices = tr.query(points, k=10, workers=-1)
    distances = distances[:, -1]
    indices = np.unique(indices[distances < 0.5])
    points = np.vstack((points, orginal_points[indices]))
    slice_position = np.mean(points[:, 1])
    width = 5
    slice_org = np.sqrt((orginal_points[:, 1] - slice_position) ** 2) <= width
    slice = np.sqrt((points[:, 1] - slice_position) ** 2) <= width
    gridsize = 1.0  # [m]
    ground_points = las.xyz[las.classification == 2]
    grid_x = ((ground_points[:, 0] - ground_points[:, 0].min()) / gridsize).astype(
        "int"
    )
    grid_y = ((ground_points[:, 1] - ground_points[:, 1].min()) / gridsize).astype(
        "int"
    )
    grid_index = grid_x + grid_y * grid_x.max()
    df = gpd.GeoDataFrame(
        {"gi": grid_index, "gx": grid_x, "gy": grid_y, "height": ground_points[:, 2]}
    )
    df2 = df.sort_values(["gx", "gy", "height"], ascending=[True, True, True])
    df3 = df2.groupby("gi")[["gx", "gy", "height"]].last()
    grid_x = np.array(df3["gx"])
    grid_y = np.array(df3["gy"])
    max_height = np.array(df3["height"])
    DTM = np.ones((grid_x.max() + 1, grid_y.max() + 1)) * np.nan
    DTM[grid_x, grid_y] = max_height
    mask = np.isnan(DTM)
    xx, yy = np.meshgrid(np.arange(DTM.shape[0]), np.arange(DTM.shape[1]))
    valid_x = xx[~mask]
    valid_y = yy[~mask]
    newarr = DTM[~mask]
    DTM_interp = interpolate.griddata(
        (valid_x, valid_y), newarr.ravel(), (xx, yy), method="linear"
    )
    gridsize = 1.0  # [m]
    filt_points = points
    grid_x = ((filt_points[:, 0] - filt_points[:, 0].min()) / gridsize).astype("int")
    grid_y = ((filt_points[:, 1] - filt_points[:, 1].min()) / gridsize).astype("int")
    grid_index = grid_x + grid_y * grid_x.max()
    df = gpd.GeoDataFrame(
        {"gi": grid_index, "gx": grid_x, "gy": grid_y, "height": filt_points[:, 2]}
    )
    df2 = df.sort_values(["gx", "gy", "height"], ascending=[True, True, True])
    df3 = df2.groupby("gi")[["gx", "gy", "height"]].last()
    grid_x = np.array(df3["gx"])
    grid_y = np.array(df3["gy"])
    max_height = np.array(df3["height"])
    DSM = np.ones((grid_x.max() + 1, grid_y.max() + 1)) * np.nan
    DSM[grid_x, grid_y] = max_height
    CHM = DSM - DTM_interp
    CHM[np.isnan(CHM)] = 0
    return CHM
