import argparse
import logging
import os
import glob
import tifffile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time


from utils.data_loading import BasicDatase
from src.MAFNet import MAFNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                rgb_img,
                dsm_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    # 预处理RGB图像
    rgb_img = torch.from_numpy(BasicDataset.preprocess(None, rgb_img, scale_factor, is_mask=False))
    rgb_img = rgb_img.unsqueeze(0).to(device=device, dtype=torch.float32)

    # 预处理DSM图像
    dsm_img = torch.from_numpy(BasicDataset.preprocess(None, dsm_img, scale_factor, is_mask=False, normalize_dsm=True))
    dsm_img = dsm_img.unsqueeze(0).to(device=device, dtype=torch.float32)
                  
    with torch.no_grad():
        start_time = time.time()
        # 前向传播得到预测结果
        output = net(rgb_img, dsm_img).cpu()  # 双分支传入RGB和DSM
        end_time = time.time()
        # 对输出进行插值处理
        output = F.interpolate(output, (rgb_img.size(2), rgb_img.size(3)), mode='bilinear')
        
        inference_time = end_time - start_time

        # 根据类别数来决定是多分类还是二分类
        if net.num_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy(), inference_time


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    # 模型文件参数
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    # 输入图像文件夹
    parser.add_argument('--input-dir', '-i', metavar='INPUT_DIR', required=True,
                        help='Directory containing input RGB images')  # RGB 图像文件夹

    # DSM 图像文件夹
    parser.add_argument('--dsm-dir', '-d', metavar='DSM_DIR', required=True,
                        help='Directory containing input DSM images')  # DSM 图像文件夹

    # 输出掩码文件夹
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR', required=True,
                        help='Directory to save output masks')

    # 可视化参数
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')

    # 不保存掩码
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')

    # 设置掩码的阈值
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    # 设置缩放因子
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')

    # 设置是否使用双线性上采样
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')

    # 类别数
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(rgb_dir, dsm_dir, output_dir):
    # 获取输入目录中的所有图像文件
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*')))  # 假设RGB文件为.tif格式
    dsm_files = sorted(glob.glob(os.path.join(dsm_dir, '*.tif')))  # 假设DSM文件为.tif格式

    print(len(rgb_files))
    print(len(dsm_files))
    # 确保RGB和DSM文件数量一致
    assert len(rgb_files) == len(dsm_files), "Number of RGB and DSM files do not match!"

    # 为每个输入文件生成输出文件名
    out_files = [os.path.join(output_dir, f'{os.path.splitext(os.path.basename(fn))[0]}_OUT.png') for fn in rgb_files]

    return rgb_files, dsm_files, out_files


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    rgb_dir = args.input_dir  # RGB图像所在的目录
    dsm_dir = args.dsm_dir   # DSM图像所在的目录
    output_dir = args.output_dir  # 输出掩码保存的目录

    # 获取文件夹中的所有图像文件及对应的输出文件名
    rgb_files, dsm_files, out_files = get_output_filenames(rgb_dir, dsm_dir, output_dir)

    net = MAFNet(n_channels_rgb=3, n_channels_dsm=1, num_classes=args.classes, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, (rgb_filename, dsm_filename) in enumerate(zip(rgb_files, dsm_files)):
        logging.info(f'Predicting image {rgb_filename} and {dsm_filename} ...')

        # 加载RGB图像和DSM图像
        rgb_img = Image.open(rgb_filename)
        dsm_img = Image.open(dsm_filename)

        # 调用预测函数
        mask, infer_time = predict_img(net=net,
                           rgb_img=rgb_img,
                           dsm_img=dsm_img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 保存结果
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        # 可视化
        if args.viz:
            logging.info(f'Visualizing results for image {rgb_filename}, close to continue...')
            plot_img_and_mask(rgb_img, mask)
