import cv2
import random
import numbers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
# import albumentations


class ConvertFromInts(object):
    def __call__(self, image, masks=None):
        return image.astype(np.float16), masks.astype(np.float16)


class Resize(object):
    """
    resize image with specific interpolate method

    Args:
         image_size :  image size.
         interpolation : interpolate method one of
            [cv2.INTER_LINEAR, cv2.INTER_AREA , cv2.INTER_CUBIC].
    """

    def __init__(self, image_size, interpolation=cv2.INTER_LINEAR):
        if isinstance(image_size, numbers.Number):
            self.image_size = (image_size, numbers)
        else:
            self.image_size = image_size
        self.interpolation = interpolation

    def __call__(self, image, masks=None):
        image_shape = image.shape
        image = cv2.resize(image, self.image_size,
                           interpolation=self.interpolation)
        if len(image.shape) < len(image_shape):
            image = np.expand_dims(image, -1)
        if masks is not None:
            masks_shape = masks.shape
            masks = cv2.resize(masks, self.image_size,
                               interpolation=self.interpolation)
            # for idx, mask in enumerate(masks[...,:]):
            #     masks[idx] = cv2.resize(mask, self.image_size,
            #                             interpolation=self.interpolation)
            if len(masks.shape) < len(masks_shape):
                masks = np.expand_dims(masks, -1)
        return image, masks


class RandomCrop(object):
    """Crop the given Image at a random location.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode='reflect'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img : Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, masks=None):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = np.pad(img, self.padding, self.padding_mode,
                         constant_values=self.fill)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            padding_width = (self.size[1] - img.shape[1]
                             )//2 + (self.size[1] - img.shape[1]) % 2
            padding_size = ((0, 0), (padding_width, padding_width), (0, 0))
            if self.padding_mode == 'constant':
                img = np.pad(img, padding_size,
                             self.padding_mode, constant_values=self.fill)
                if masks is not None:
                    masks = np.pad(masks, padding_size,
                                   self.padding_mode, constant_values=self.fill)
            else:
                img = np.pad(
                    img, padding_size, self.padding_mode)
                if masks is not None:
                    masks = np.pad(
                        masks, padding_size, self.padding_mode)
            # for idx in range(len(masks)):
            #     if self.padding_mode == 'constant':
            #         masks[idx] = np.pad(masks[idx], padding_size,
            #                             self.padding_mode, constant_values=self.fill)
            #     else:
            #         masks[idx] = np.pad(
            #             masks[idx], padding_size, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            padding_height = (
                self.size[0] - img.shape[0])//2+(self.size[0] - img.shape[0]) % 2
            padding_size = ((padding_height, padding_height), (0, 0), (0, 0))
            if self.padding_mode == 'constant':
                img = np.pad(img, padding_size, self.padding_mode,
                             constant_values=self.fill)
                if masks is not None:
                    masks = np.pad(masks, padding_size,
                                   self.padding_mode, constant_values=self.fill)
            else:
                img = np.pad(img, padding_size, self.padding_mode)
                if masks is not None:
                    masks = np.pad(
                        masks, padding_size, self.padding_mode)
            # for idx in range(len(masks)):
            #     if self.padding_mode == 'constant':
            #         masks[idx] = np.pad(masks[idx], padding_size,
            #                             self.padding_mode, constant_values=self.fill)
            #     else:
            #         masks[idx] = np.pad(
            #             masks[idx], padding_size, self.padding_mode)
        i, j, h, w = self.get_params(img, self.size)
        img = img[i:i+h, j:j+w]
        if masks is not None:
            masks = masks[i:i+h, j:j+w]

        return img, masks


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        scale (int): Isotropic scale factor.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, scale=1, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.scale = scale
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, masks=None):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        image_size = img.shape
        angle = self.get_params(self.degrees)
        M = cv2.getRotationMatrix2D(
            (image_size[1]//2, image_size[0]//2), angle,  self.scale)
        for i in range(img.shape[2]):
            img[..., i] = cv2.warpAffine(
                img[..., i], M, (image_size[1], image_size[0]))
        if masks is not None:
            for i in range(masks.shape[2]):
                masks[..., i] = cv2.warpAffine(
                    masks[..., i], M, (image_size[1], image_size[0]))
        return img, masks


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, masks=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        img = ((img/255.-self.mean)/self.std).astype(np.float32)
        return img, masks


class rgb2gray(object):
    def __call__(self, img, masks=None):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), masks


class RandomFlip(object):
    """
        Args:
            orientation: vertical flip 0, horizontal flip 1
            p: probability for filp
        Returns:
            PIL Image: Randomly flipped image.
        """

    def __init__(self, orientation=0, p=0.5):
        assert orientation in [0,1]
        self.orientation = orientation
        self.p = p

    def __call__(self, img, masks=None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img= cv2.flip(img, self.orientation)
            masks=cv2.flip(masks, self.orientation)
        return img, masks


class ToTensor(object):
    """ Convert image, masks  to torch.FloatTensor """

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img, masks=None):
        img = self.totensor(img)
        if masks is not None:
            masks = torch.as_tensor(
                masks).permute(2, 0, 1)

        return img, masks


class ImageTransform():
    def __init__(self, image_size, mean, std):
        self.augment = [
            # RandomCrop(image_size),
            # RandomRotation(180),
            # rgb2gray(),
            RandomFlip(),
            RandomFlip(orientation=1),
            Resize(image_size, cv2.INTER_LINEAR),
            Normalize(mean, std),
            ToTensor()
        ]

    def __call__(self, image, masks=None):
        for transfer in self.augment:
            image, masks = transfer(image, masks)
        return image, masks


class MaskTransform():
    def __init__(self, image_size=(227, 227)):
        self.augment = [
            Resize(image_size),
            ToTensor()
        ]

    def __call__(self, image):
        for transfer in self.augment:
            image, _ = transfer(image)
        return image
