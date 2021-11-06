import os
from PIL import Image
import argparse
import glob
import time
import re
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor
from skimage import color, io


parser = argparse.ArgumentParser()
parser.add_argument(
    "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
)
parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
parser.add_argument("--cuda", default=True, action="store_false")
parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
parser.add_argument("--image", type=str, default="./images/01.jpg", help="path of input clips")
parser.add_argument("--ref", type=str, default="./ref/01.jpg", help="path of refernce images")
parser.add_argument("--output", type=str, default="./output/01.png", help="path of output clips")
opt = parser.parse_args()


def colorize_image(image_path, ref_path, output_file, nonlocal_net, colornet, vggnet, sigma_color=4, lambda_value=500):
    wls_filter_on = True
    lambda_value = lambda_value
    sigma_color = sigma_color

    print("processing the file:", image_path)

    # NOTE: resize frames to 216*384
    transform = transforms.Compose(
        [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    )

    print("reference name:", ref_path)
    frame_ref = Image.open(ref_path)

    total_time = 0
    I_last_lab_predict = None

    frame1 = Image.open(image_path)

    IA_lab_large = transform(frame1).unsqueeze(0).cuda()
    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")

    IA_l = IA_lab[:, 0:1, :, :]
    IA_ab = IA_lab[:, 1:3, :, :]
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]
    if I_last_lab_predict is None:
        if opt.frame_propagate:
            I_last_lab_predict = IB_lab
        else:
            I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

    # start the frame colorization
    with torch.no_grad():
        I_current_lab = IA_lab
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))

        # t_start = time.clock()
        torch.cuda.synchronize()

        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
            I_current_lab,
            I_reference_lab,
            I_last_lab_predict,
            features_B,
            vggnet,
            nonlocal_net,
            colornet,
            feature_noise=0,
            temperature=1e-10,
        )
        # I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

    # update timing
    torch.cuda.synchronize()

    # upsampling
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
    )

    # filtering
    if wls_filter_on:
        guide_image = uncenter_l(curr_bs_l) * 255 / 100
        wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
        )
        curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
        curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
        curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
        curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
        curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
        IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
    else:
        IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

    # save the frames
    final_image = IA_predict_rgb
    if final_image is not None:
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    io.imsave(output_file, final_image)
    # save_frames(IA_predict_rgb, output_path, index)


def set_args(message):
    opt.image = message['user_img']  # 黑白图片
    opt.ref = message['ref_img']  # 指导图片
    opt.output = re.sub("\.jpg|\.png|\.jpeg","",message['user_img']+"_"+message['ref_img']) + ".png" # 输出图片



user_image = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/colorImage/user_imgs'  # 黑白图片文件夹
user_ref = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/colorImage/ref_imgs'  # 指导图片文件夹
result_path = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/colorImage/res_imgs'  # 输出文件夹
message_json = '/workspace/go_proj/src/Ai_WebServer/algorithm_utils/colorImage/message.json'


if __name__ == '__main__':
    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()
    last_msg = {}
    import json, time
    while True:
        try:
            with open(message_json, "r", encoding="utf-8") as f:
                message = json.load(f)

            if message == last_msg:
                print('waiting...')
                time.sleep(1)
                continue
            else:
                set_args(message)

            image_path = os.path.join(user_image, opt.image)
            ref_path = os.path.join(user_ref, opt.ref)
            output_path = os.path.join(result_path, opt.output)
            if os.path.exists(output_path):
                print("debvc exist...")
                continue
            print('image:', image_path, ', ref:', ref_path, ', output:', output_path)
            colorize_image(image_path, ref_path, output_path, nonlocal_net, colornet, vggnet)
            print('colorized success')
            last_msg = message
            time.sleep(1)
        except Exception as e:
            print(e)
            continue