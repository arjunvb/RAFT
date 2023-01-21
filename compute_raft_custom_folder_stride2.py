import os, sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from raft import RAFT
from utils.utils import InputPadder
from .custom_utils import *

DEVICE = "cuda"


def infer_optical_flows_stride2(model, image_dir, output_dir, skip_exists=False):
    flow_save_path = os.path.join(output_dir, "flow_f2")
    if not os.path.exists(flow_save_path):
        os.makedirs(flow_save_path)
    flow_bk_save_path = os.path.join(output_dir, "flow_b2")
    if not os.path.exists(flow_bk_save_path):
        os.makedirs(flow_bk_save_path)
    flow_img_save_path = os.path.join(output_dir, "flow_imgs2")
    if not os.path.exists(flow_img_save_path):
        os.makedirs(flow_img_save_path)

    with torch.no_grad():
        # images = glob.glob(os.path.join(image_dir, '*.png')) + \
        # glob.glob(os.path.join(image_dir, '*.jpg'))
        # images = glob.glob(image_dir + "*.png") + glob.glob(image_dir + "*.jpg")
        images = image_dir
        images = sorted(images)
        n_images = len(images)

        for img_id in tqdm(range(n_images - 2)):
            # output name
            imfile1, imfile2 = images[img_id], images[img_id + 2]
            flow_save_name = os.path.join(
                flow_save_path,
                imfile1.split("/")[-1].replace(".png", ".flo").replace(".jpg", ".flo"),
            )
            flow_bk_save_name = os.path.join(
                flow_bk_save_path,
                imfile1.split("/")[-1].replace(".png", ".flo").replace(".jpg", ".flo"),
            ).replace(".flo", "-bk.flo")
            img_save_name = os.path.join(flow_img_save_path, imfile1.split("/")[-1])
            if skip_exists:
                if (
                    os.path.exists(flow_save_name)
                    and os.path.exists(flow_bk_save_name)
                    and os.path.exists(img_save_name)
                ):
                    continue

            # compute forward and backward flow
            image1 = load_image(imfile1).to(DEVICE)
            image2 = load_image(imfile2).to(DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=32, test_mode=True)
            flow_low_bk, flow_up_bk = model(image2, image1, iters=32, test_mode=True)
            flow_forward, flow_backward = padder.unpad(flow_up), padder.unpad(
                flow_up_bk
            )
            flow_forward = flow_forward[0].permute(1, 2, 0).cpu().numpy()
            flow_backward = flow_backward[0].permute(1, 2, 0).cpu().numpy()

            # Visualize and save the results for point trajectory
            viz(flow_forward, img_save_name)
            flow_write(flow_save_name, flow_forward)
            flow_write(flow_bk_save_name, flow_backward)


def compute_raft_custom_folder_stride2(
    image_dir, output_dir, args=None, skip_exists=False
):
    """
    Inputs:
    - image_dir: str - The folder containing input images
    - ouptut_dir: str - The workspace directory
    """
    # use default args
    if args is None:
        args = argparse.Namespace()
        args.small = False
        args.alternate_corr = False
        args.mixed_precision = False
        args.model = "models/raft-things.pth"

    # initiate RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    curpath = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curpath, args.model)
    print("Loading weights:  {0}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # inference
    infer_optical_flows_stride2(model, image_dir, output_dir, skip_exists=skip_exists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="The folder containing input images")
    parser.add_argument("--output_dir", help="Path to the output workspace")
    parser.add_argument(
        "--model", default="./models/raft-things.pth", help="restore checkpoint"
    )
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    parser.add_argument(
        "--skip_exists", action="store_true", help="whether to skip exists"
    )
    args = parser.parse_args()

    compute_raft_custom_folder_stride2(
        args.image_dir, args.output_dir, args=args, skip_exists=args.skip_exists
    )
