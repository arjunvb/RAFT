import os, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from utils import flow_viz

TAG_FLOAT = 202021.25


def __apply_motion_blur(img):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 100

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    return horizonal_mb


def load_image(imfile, apply_blur=False):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if apply_blur:
        img = __apply_motion_blur(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def viz(flo, name):
    # map flow to rgb image
    flo_v = flow_viz.flow_to_image(flo, convert_to_bgr=True)
    cv2.imwrite(name, flo_v)


def flow_write(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, "wb")
    # write the header
    np.array(TAG_FLOAT).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
