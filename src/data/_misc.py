from torch import Tensor

from torchvision.tv_tensors import (
    BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)

_boxes_keys = ['format', 'canvas_size']


def convert_to_tv_tensor(tensor: Tensor, box_format='xyxy', spatial_size=None) -> Tensor:
    box_format = getattr(BoundingBoxFormat, box_format.upper())
    _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
    return BoundingBoxes(tensor, **_kwargs)