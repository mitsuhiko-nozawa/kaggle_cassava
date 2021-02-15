from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

def get_resize_transforms():
    return Compose([
            Resize(512, 512),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])



def get_transforms(phase, params, transforms=None):    
    if phase == 'train':
        return Compose([ get_aug(aug, params) for aug in transforms])

    elif phase in ['valid', 'test']:
        
        return Compose([
            #CenterCrop(params["size"], params["size"], p=1.), #
            Resize(params["size"], params["size"]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def get_aug(aug, params):
    if aug in ["Resize", "RandomResizedCrop"]:
        return eval(aug)(params["size"], params["size"])

    elif aug in ["Transpose", "HorizontalFlip", "VerticalFlip", "ShiftScaleRotate"]:
        return eval(aug)(p=0.5)

    elif aug in ["HueSaturationValue"]:
        return eval(aug)(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5)

    elif aug in ["RandomBrightnessContrast"]:
        return eval(aug)(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5)

    elif aug in ["Normalize"]:
        return eval(aug)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif aug in ["CoarseDropout", "Cutout"]:
        return eval(aug)(p=0.5)

    elif aug in ["ToTensorV2"]:
        return eval(aug)()