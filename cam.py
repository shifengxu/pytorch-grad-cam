import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad
}

log_fn = print


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[2])
    parser.add_argument('--image-path', type=str, default='./examples/both.png')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'],
                        help='gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    device = f"cuda:{args.gpu_ids[0]}" if args.gpu_ids and torch.cuda.is_available() else 'cpu'
    args.device = device
    log_fn(f"args: {args}")
    log_fn(f"device: {device}")
    return args


def gen_attr_map(args):
    model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4[-1]]

    log_fn(f"Model: {type(model).__name__}")
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        log_fn(f"  model = torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
    model.to(args.device)
    log_fn(f"  model.to({args.device})")

    # https://docs.python.org/release/2.3.5/whatsnew/section-slices.html
    # The slicing syntax supports an optional 3rd "step" argument.
    # Negative values also work to make a copy of the same list in reverse order
    # >>> L = range(10)
    # >>> L[::2]  # ==> [0, 2, 4, 6, 8]
    # >>> L::-1]  # ==> [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    image_path = args.image_path
    log_fn(f"Load file: {image_path}")
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    # cv2.imread(image_path, 1) ==> ndarray, shape: [224, 224, 3]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # input_tensor shape: [1, 3, 224, 224]

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    method = args.method
    log_fn(f"cam method: {method}")
    cam_algorithm = methods[method]

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=args.device) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
        log_fn(f"args.aug_smooth  : {args.aug_smooth}")
        log_fn(f"args.eigen_smooth: {args.eigen_smooth}")
        # grayscale_cam: ndarray, shape [1, 224, 224]

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        # grayscale_cam: ndarray, shape [224, 224]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    # with cam_algorithm

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.device)
    gb = gb_model(input_tensor, target_category=target_category)
    # gb shape: [224, 224, 3]

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    stem, ext = os.path.splitext(image_path)    # ./path1/path2/filename .jpg
    path1 = f'{stem}_{method}_cam{ext}'
    path2 = f'{stem}_{method}_cam_gb{ext}'
    path3 = f'{stem}_{method}_gb{ext}'
    cv2.imwrite(path1, cam_image)
    cv2.imwrite(path2, cam_gb)
    cv2.imwrite(path3, gb)
    log_fn(f"image_path: {image_path}")
    log_fn(f"image new : {path1}")
    log_fn(f"image new : {path2}")
    log_fn(f"image new : {path3}")


def main():
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    args = get_args()
    gen_attr_map(args)

if __name__ == '__main__':
    main()
