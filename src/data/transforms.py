"""Data augmentation transforms using albumentations."""
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    height: int = 480,
    width: int = 2176,
    p_hflip: float = 0.5,
    p_vflip: float = 0.0,
    p_shift: float = 0.3,
    p_scale: float = 0.3,
    p_rotate: float = 0.0,
    p_brightness: float = 0.2,
    p_contrast: float = 0.2,
    p_noise: float = 0.1,
) -> A.Compose:
    """Get training data augmentation transforms.

    Args:
        height: Target height for resize.
        width: Target width for resize.
        p_hflip: Probability of horizontal flip.
        p_vflip: Probability of vertical flip.
        p_shift: Probability of shift.
        p_scale: Probability of scale.
        p_rotate: Probability of rotation.
        p_brightness: Probability of brightness adjustment.
        p_contrast: Probability of contrast adjustment.
        p_noise: Probability of Gaussian noise.

    Returns:
        Albumentations Compose object.
    """
    transforms = [
        A.HorizontalFlip(p=p_hflip),
        A.VerticalFlip(p=p_vflip),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=5 if p_rotate > 0 else 0,
            border_mode=0,
            p=max(p_shift, p_scale, p_rotate),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=max(p_brightness, p_contrast),
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=p_noise),
    ]

    return A.Compose(transforms)


def get_val_transforms(
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> A.Compose:
    """Get validation data transforms (no augmentation).

    Args:
        height: Target height for resize (optional).
        width: Target width for resize (optional).

    Returns:
        Albumentations Compose object.
    """
    transforms = []

    if height is not None and width is not None:
        transforms.append(A.Resize(height, width))

    return A.Compose(transforms)


def apply_transform(image, mask, transform):
    """Apply albumentations transform to image and mask.

    Args:
        image: Image array (H, W, C) or (H, W).
        mask: Mask array (H, W) or (H, W, C).
        transform: Albumentations transform.

    Returns:
        Transformed (image, mask) tuple.
    """
    if transform is None:
        return image, mask

    # Handle multi-channel masks
    if mask.ndim == 3 and mask.shape[0] < mask.shape[-1]:
        # (C, H, W) -> (H, W, C)
        mask = mask.transpose(1, 2, 0)
        transpose_back = True
    else:
        transpose_back = False

    result = transform(image=image, mask=mask)
    transformed_image = result["image"]
    transformed_mask = result["mask"]

    if transpose_back:
        transformed_mask = transformed_mask.transpose(2, 0, 1)

    return transformed_image, transformed_mask
