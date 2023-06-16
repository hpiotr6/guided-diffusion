import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import re

def get_data_loaders(
    *,
    data_dir,
    image_size,
    batch_size,
    shuffle=False,
    length=-1,
    margin=0
    ):

    all_files_ = _list_image_files_recursively(data_dir)
    regex = r"([A-Za-z0-9]+_?[A-Za-z0-9]+).*-([C])-.*\.png"
    without = {}
    all_files = []
    for path in all_files_:
        filename = bf.basename(path)
        match = re.match(regex, filename)
        if not match:
            all_files.append(path)
        else:
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")
            without[filename[:-10]] = pil_image

    loaders = []
    for path in all_files:
        filename = bf.basename(path)

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        tiles, width, height, grid = divide_into_tiles(pil_image, image_size, margin)

        dataset = MyImageDataset(
            image_size,
            tiles,
            classes=None,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size()
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )

        clean = without[filename[:-10]]

        loaders.append((loader, width, height, grid, clean))
        if len(loaders) == length:
            break

    if shuffle:
        loaders = random.shuffle(loaders)
    return loaders


def divide_into_tiles(image, image_size, margin = 0):
    height = image.height
    width = image.width

    tiles = []
    x = 0
    grid = [1, 1]
    while x + image_size < width:
        y = 0
        while y + image_size < height:
            tiles.append(image.crop((x, y, x+image_size, y+image_size)))
            y += image_size - margin
            grid[0] += 1
        tiles.append(image.crop(
            (x, max(0, height-image_size), x+image_size, height)))
        x += image_size - margin
        grid[1] += 1
    y = 0
    while y + image_size < height:
        tiles.append(image.crop(
            (max(0, width-image_size), y, width, y+image_size)))
        y += image_size - margin
    tiles.append(image.crop((max(0, width-image_size),
                    max(0, height-image_size), width, height)))

    return tiles, width, height, grid

def join_from_tiles(tiles, width, height, grid, margin = 0):
    image = np.zeros((height, width, 3))
    image_size = tiles[0].shape[0]

    i = 0
    x = 0
    while x + image_size < width:
        y = 0
        while y + image_size < height:
            image[y:y+image_size,x: x+image_size, :] += apply_merging_margin(tiles[i], grid, i, margin)
            i+= 1
            y += image_size - margin

        image[y:height, x:x+image_size, :] += apply_merging_margin(tiles[i][image_size-height+y:,:,:], grid, i, margin)
        i+=1
        x += image_size - margin
    y = 0
    while y + image_size < height:
        image[y:y+image_size, x:width, :] += apply_merging_margin(tiles[i][:,image_size-width+x:,:], grid, i, margin)
        i+=1
        y += image_size - margin
    image[y:height, x:width, :] += apply_merging_margin(tiles[i][image_size-height+y:,image_size-width+x:,:], grid, i, margin)

    return image

def apply_merging_margin(tile, grid, index, margin):
    top = True
    bottom = True
    left = True
    right = True
    if index < grid[0]:
        left = False
    if index >= grid[0]*(grid[1] -1):
        right = False
    if index % grid[0] == 0:
        top = False
    if (index + 1) % grid[0] == 0:
        bottom = False

    height, width, _ = tile.shape

    t = tile.dtype
    tile = tile.astype(np.float64)
    if top:
        for i in range(margin):
            tile[i,:,:] *= (i+1)/(margin+1)
    if bottom:
        for i in range(margin):
            tile[height-i-1,:,:] *= (i+1)/(margin+1)
    if left:
        for i in range(margin):
            tile[:,i,:] *= (i+1)/(margin+1)
    if right:
        for i in range(margin):
            tile[:,width-i-1,:] *= (i+1)/(margin+1)

    return tile.astype(t)


class MyImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        images,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = images[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        pil_image = self.local_images[idx]

        arr = center_crop_arr(pil_image, self.resolution)

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    weighted_samplng=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # # Assume classes are the first part of the filename,
        # # before an underscore.
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        regex = r"([A-Za-z]+_?[A-Za-z]+).*-([CR])-.*\.png"
        class_names=[]
        for path in all_files:
            filename = bf.basename(path)
            match = re.match(regex, filename)
            if match:
                city = match.group(1)
                element = match.group(2)
                class_names.append(element)
                # print("City:", city)
                # print("Element:", element)
            else:
                raise ValueError("No match found for filename:", filename)

        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    else:
        # take only rainy classes
        regex = r"([A-Za-z]+_?[A-Za-z]+).*-([C])-.*\.png"
        for path in all_files:
            filename = bf.basename(path)
            match = re.match(regex, filename)
            if match:
                all_files.remove(path)
                # city = match.group(1)
                # element = match.group(2)
                # print("City:", city)
                # print("Element:", element)



    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if weighted_samplng:
        class_count = np.unique(classes, return_counts=True)[1]
        weight = 1.0 / class_count
        samples_weight = weight[classes]
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=not deterministic)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1, drop_last=True, sampler=sampler
        )
    elif deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
