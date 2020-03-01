import torch
import torchvision.utils as utils

import os

import data_loader
import net

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(configuration):
    pretrained_model_path = configuration['pretrained_model_path']
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    image_saving_path = configuration['image_saving_path']

    print('loading the model from {}'.format(pretrained_model_path))
    ada_in_model = net.get_ada_in_model(configuration)

    print('loading the model from {}'.format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    ada_in_model.load_state_dict(checkpoint['model_state_dict'])

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))
    content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                         for i in range(number_style_images)]

    for i in range(number_style_images):
        print("test_image {} at {}".format(i + 1, style_image_files[i]))

    for i in range(number_content_images):
        print("test_image {} at {}".format(i + 1, content_image_files[i]))

    for i in range(number_style_images):
        for j in range(number_content_images):
            print('at image {}'.format(i))
            with torch.no_grad():
                content_image = data_loader.image_loader(content_image_files[j], loader)
                style_image = data_loader.image_loader(style_image_files[i], loader)
                noise_image = torch.rand(1, 3, 256, 256)

                transfer_decoded_ada_in, _, _ = ada_in_model(style_image, content_image)
                content_decoded_ada_in, _, _ = ada_in_model(content_image, content_image)
                style_decoded_ada_in, _, _ = ada_in_model(style_image, style_image)
                noise_decoded_ada_in, _, _ = ada_in_model(style_image, noise_image)

                utils.save_image([data_loader.imnorm(content_image, None),
                                  data_loader.imnorm(style_image, None),
                                  data_loader.imnorm(transfer_decoded_ada_in, None)],
                                 '{}/ada_in_test_single_image_A_{}_{}.jpeg'.format(image_saving_path, i, j),
                                 pad_value=1)

                utils.save_image([data_loader.imnorm(content_image, None),
                                  data_loader.imnorm(content_image, None),
                                  data_loader.imnorm(content_decoded_ada_in, None)],
                                 '{}/ada_in_test_single_image_B_{}_{}.jpeg'.format(image_saving_path, i, j),
                                 pad_value=1)

                utils.save_image([data_loader.imnorm(style_image, None),
                                  data_loader.imnorm(style_image, None),
                                  data_loader.imnorm(style_decoded_ada_in, None)],
                                 '{}/ada_in_test_single_image_C_{}_{}.jpeg'.format(image_saving_path, i, j),
                                 pad_value=1)

                utils.save_image([data_loader.imnorm(noise_image, None),
                                  data_loader.imnorm(style_image, None),
                                  data_loader.imnorm(noise_decoded_ada_in, None)],
                                 '{}/ada_in_test_single_image_D_{}_{}.jpeg'.format(image_saving_path, i, j),
                                 pad_value=1)
