import torch
import torchvision.transforms as transforms
import torchvision.utils as u

import os

import data_loader as data_loader
import time
import net as net
import numpy as np
from collections import OrderedDict

# local dev
local = False

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prefix = '/Users/Johannes/Desktop/mom_al_deep/' if local else '/home/zenn/bachelor_thesis/pytorch_models/'
suffix = ''

moment_alignment_model_path_list = [
    -1,
    prefix + 'moment_alignment_model_0_1_3_0' + suffix,
    prefix + 'moment_alignment_model_0_1_4_0' + suffix,
    prefix + 'moment_alignment_model_0_1_5_0' + suffix,
]

prefix = '/Users/Johannes/Desktop/ada_in_module_loss/' if local else prefix
suffix = ''
model_path_list = [
    prefix + 'ada_in_model_2_2_10' + suffix,
    prefix + 'ada_in_model_3_2_10' + suffix,
    prefix + 'ada_in_model_4_2_10' + suffix,
    prefix + 'ada_in_model_5_2_10' + suffix,
]

use_moment_alignment_model_list = [False, True, True, True]

number_moments_mom_al_list = [-1, 3, 4, 5]

number_moments_list = [2, 3, 4, 5]

ada_in_module_list = [2, -1, -1, -1]

prefix = '../pytorch_models/' if local else prefix
suffix = '.pth'

model_configuration = {
    'torch_lua_vgg_model_path': prefix + 'vgg_r41' + suffix,
    'use_moment_alignment_model_list': use_moment_alignment_model_list,
    'moment_alignment_model_path_list': moment_alignment_model_path_list,
    'number_moments_list': number_moments_list,
    'ada_in_module_list': ada_in_module_list,
    'number_moments_mom_al_list': number_moments_mom_al_list,
}

model_list = [None for _ in range(len(model_path_list))]
for i in range(len(model_list)):
    ada_in_model = net.get_ada_in_model(model_configuration, use_list=True, list_index=i)
    checkpoint = torch.load(model_path_list[i], map_location=device)
    print('loaded model at {}'.format(model_path_list[i]))

    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        # if 'adaIN_layer' not in k:
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    ada_in_model.load_state_dict(new_state_dict)

    model_list[i] = ada_in_model.to(device)

prefix = '/Users/Johannes/Desktop/encoder_decoder_exp/' if local else prefix
suffix = '.pth'
encoder_decoder_model_paths = {
    'encoder_model_path': prefix + 'encoder_1_25_6_state_dict' + suffix,
    'decoder_model_path': prefix + 'decoder_1_25_6_state_dict' + suffix,
}

prefix = '../' if local else '/home/zenn/data/'
style_images_path = prefix + 'testset_style'
content_images_path = prefix + 'testset_content'

number_content_images = len(os.listdir(content_images_path))
number_style_images = len(os.listdir(style_images_path))

content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                       for i in range(number_content_images)]
style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                     for i in range(number_style_images)]

loader = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(256),
     transforms.ToTensor()])

content_images = [data_loader.image_loader(content_image_files[i], loader).to(device) for i in range(number_content_images)]
style_images = [data_loader.image_loader(style_image_files[i], loader).to(device) for i in range(number_style_images)]

prefix = './time_measurement_images/' if local else '/home/zenn/time_measurements/img_ada_in/'
def measure(model, content_image, style_image, img_saving_number):
    # synchronize gpu time and measure fp

    # vvvvvvvvvvvvvvvvvvvvvvv comment here if no cuda available
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y_pred, _, _ = model(style_image, content_image, compute_loss=False)

    # vvvvvvvvvvvvvvvvvvvvvvv comment here if no cuda available
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    u.save_image(y_pred, prefix + 'img_{}.jpg'.format(img_saving_number))

    return elapsed_fp


def benchmark(model):
    # DRY RUNS
    for i in range(5):
        _ = measure(model, content_images[i], style_images[i], -i)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []

    img_saving_number = 0
    for i in range(number_content_images):
        for j in range(number_style_images):
            img_saving_number += 1
            t_fp = measure(model, content_images[i], style_images[j], img_saving_number)
            t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def benchmark_models():
    for i in range(len(model_list)):
        print('BENCHMARKING MODEL {}'.format(model_path_list[i]))
        t = benchmark(model_list[i])
        print('FORWARD PASS: ', np.mean(np.asarray(t) * 1e3), '+/-', np.std(np.asarray(t) * 1e3))
        print('now getting the next model to evaluate')


if __name__ == '__main__':
    print('running main benchmarking loop')
    benchmark_models()
