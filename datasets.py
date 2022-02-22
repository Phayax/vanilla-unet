import logging
import math
from pathlib import Path
from typing import Tuple, List

import imageio
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects
from torch.utils.data import Dataset

from utils import PathLike


class KvasirSEGDataset(Dataset):
    """
    This Dataset provides the data from the KvasirSEG Data as image, mask pairs
    Image loading is done using imageio and wrapping this in np.asarray (https://stackoverflow.com/a/50280844)
    """
    def __init__(self, images_path: PathLike, masks_path: PathLike, target_size: Tuple[int, int], keep_aspect_ratio: bool, preload: bool = True):
        super(KvasirSEGDataset, self).__init__()
        logging.info("Creating new KvasirSegDataset instance...")
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)

        # get a list of the images.
        self.image_list = [x for x in sorted(list(Path(images_path).glob("*.jpg")))]
        # check that every image has a mask
        # after this we only need to store the mask path and image list.
        for image in self.image_list:
            if not (Path(masks_path) / image.name).exists():
                raise FileNotFoundError(f"Could not find mask for image: {image.name}. Aborting!")

        self.target_size = target_size
        self.keep_aspect_ratio = keep_aspect_ratio

        self.images = []  # type: List[np.ndarray]
        self.masks = []  # type: List[np.ndarray]
        self.preload = preload
        if self.preload:
            self._preload()
        logging.info("Done!")

    def __resize_and_pad(self, resize_im: np.ndarray) -> np.ndarray:
        """
        This function takes an image and a target size and resizes the image to fit this size.
        After resizing while keeping the aspect ratio, it is padded with black space in each necessary dimension.
        We are using PIL.Image.crop for this. It is not documented well but can also pad the picture if coordinates
        are larger than image or negative.
        Reminder: PIL uses cartesian coordinates so width is x or size[0], height is y or size[1]
        Note:
        If we need to pad an uneven number of pixels the right and bottom will get the additional pixel
        :param resize_im: PIL Image object to be resized
        :param target_size: The target size as a Tuple of int
        :return: the resized and padded image
        """
        resize_im = Image.fromarray(resize_im)
        resize_im.thumbnail(self.target_size, Image.LANCZOS)

        pad_left = math.floor((self.target_size[0] - resize_im.size[0]) / 2.0)
        pad_right = math.ceil((self.target_size[0] - resize_im.size[0]) / 2.0)
        pad_top = math.floor((self.target_size[1] - resize_im.size[1]) / 2.0)
        pad_bottom = math.ceil((self.target_size[1] - resize_im.size[1]) / 2.0)
        # join them into a flat tuple
        crop_box = 0 - pad_left, 0 - pad_top, resize_im.size[0] + pad_right, resize_im.size[1] + pad_bottom
        # print(crop_box)
        resize_im = resize_im.crop(crop_box)
        return np.asarray(resize_im)

    def _preprocess_image(self, im: np.ndarray) -> np.ndarray:
        """
        Resize the image to target dimensions and normalize
        :param im:
        :return:
        """
        if self.keep_aspect_ratio:
            im = self.__resize_and_pad(im)
        else:
            im = np.asarray(Image.fromarray(im).resize(size=self.target_size, resample=Image.LANCZOS))
        im = im.astype(np.float32) / 255.0

        im = im.transpose((2, 0, 1))
        return im

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        KvasirSEG Masks are RGB Jpeg images.
        1. We convert to grayscale with Pillow
        2. Remove jpeg artifacts with thresholding and small object removal
        3. Resize the resulting image to target size
        4. Convert to proper Segmentation mask labels.
        :param mask:
        :return:
        """
        # convert to grayscale
        pil_mask = Image.fromarray(mask)
        pil_mask = pil_mask.convert("L")

        # the copy is important otherwise we're getting an immutable view into the numpy values of the PIL.Image
        np_mask = np.asarray(pil_mask).copy()

        # binarise image
        np_mask[np_mask < 240] = 0
        np_mask[np_mask >= 240] = 255

        # removing elements smaller than 10 pixels
        # in this step we are also changing the image from 255 to 1 as segmentation target
        np_mask = remove_small_objects(np_mask.astype(bool), min_size=10, connectivity=2).astype(np.uint8)

        if self.keep_aspect_ratio:
            np_mask = self.__resize_and_pad(np_mask)
        else:
            np_mask = np.asarray(Image.fromarray(np_mask).resize(size=self.target_size, resample=Image.LANCZOS))

        np_mask = np_mask[np.newaxis, :, :]

        # Cast to float for loss calculation
        np_mask = np_mask.astype(np.float32)

        return np_mask

    def _preload(self):
        """
        Preloads all images into memory and does preprocessing.
        Images are stored as a list of numpy arrays.
        Kvasir-SEG isn't huge so this should be no problem in most cases.
        :return:
        """
        logging.info(f"Preloading {self.__len__()} image pairs")
        for image in self.image_list:
            im = np.asarray(imageio.imread(image))
            im = self._preprocess_image(im)
            mask = np.asarray(imageio.imread(self.masks_path / image.name))
            mask = self._preprocess_mask(mask)
            self.images.append(im)
            self.masks.append(mask)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        if self.preload:
            im = self.images[item]
            mask = self.masks[item]
        else:
            im = np.asarray(imageio.imread(self.image_list[item]))
            im = self._preprocess_image(im)

            mask = np.asarray(imageio.imread(self.masks_path / self.image_list[item].name))
            mask = self._preprocess_mask(mask)

        # TODO: add data augmentation possibilities here
        # im, mask = self._augment(im, mask)

        return im, mask
