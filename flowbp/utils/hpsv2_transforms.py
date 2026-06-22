import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Compose

class BatchResizeMaxSize(nn.Module):
    def __init__(self, resize_max_size_ins):
        super().__init__()
        self.max_size = resize_max_size_ins.max_size
        self.interpolation = resize_max_size_ins.interpolation
        self.fn = resize_max_size_ins.fn
        self.fill = resize_max_size_ins.fill

    def forward(self, img):
        # img: (B, C, H, W)
        height, width = img.shape[2:]
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img

class BatchMaskAwareNormalize(nn.Module):
    def __init__(self, mask_aware_normalize_ins):
        super().__init__()
        self.normalize = mask_aware_normalize_ins.normalize

    def forward(self, tensor):
        if tensor.shape[1] == 4:
            return torch.cat(
                [self.normalize(tensor[:, :3]), tensor[:, 3:]], dim=1,
            )
        else:
            return self.normalize(tensor)

class HPSV2TransformsWithGrad(Compose):
    def __init__(self, hpsv2_preprocessor: Compose):
        transforms = [
            BatchResizeMaxSize(hpsv2_preprocessor.transforms[2]),
            BatchMaskAwareNormalize(hpsv2_preprocessor.transforms[3]),
        ]
        super().__init__(transforms)