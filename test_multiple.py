import torch
import torchvision.utils as utils

import os

import data_loader
import net

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(configuration):
    model_path_list = configuration['model_path_list']
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    image_test_saving_path_list = configuration['image_saving_path_list']

    model_list = model_path_list
    for i in range(len(model_path_list)):
        ada_in_model = net.get_ada_in_model(configuration, use_list=True, list_index=i)
        checkpoint = torch.load(model_path_list[i], map_location=device)
        ada_in_model.load_state_dict(checkpoint['model_state_dict'])
        model_list[i] = ada_in_model

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))
    content_image_files = ['{}/{}'.format(content_images_path, sorted(os.listdir(content_images_path))[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, sorted(os.listdir(style_images_path))[i])
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
                result_images = [0 for _ in range(len(model_list))]

                for k in range(len(model_list)):
                    decoded_ada_in, content_loss, style_loss = model_list[k](style_image, content_image)
                    result_images[k] = data_loader.imnorm(decoded_ada_in, None)
                    utils.save_image([data_loader.imnorm(content_image, None),
                                      data_loader.imnorm(style_image, None),
                                      data_loader.imnorm(decoded_ada_in, None)],
                                     '{}/ada_in_test_image_{}_{}.jpeg'.format(image_test_saving_path_list[k], i, j),
                                     normalize=True, scale_each=True, pad_value=1)

                utils.save_image([data_loader.imnorm(content_image, None),
                                  data_loader.imnorm(style_image, None)] + result_images,
                                 '{}/ada_in_test_image_big_A_{}_{}.jpeg'.format(image_test_saving_path_list[-1], i, j),
                                 normalize=True, scale_each=True, pad_value=1)

                utils.save_image([data_loader.imnorm(content_image, None)] + result_images,
                                 '{}/ada_in_test_image_big_B_{}_{}.jpeg'.format(image_test_saving_path_list[-1], i, j),
                                 normalize=True, scale_each=True, pad_value=1)

                utils.save_image([data_loader.imnorm(style_image, None)] + result_images,
                                 '{}/ada_in_test_image_big_C_{}_{}.jpeg'.format(image_test_saving_path_list[-1], i, j),
                                 normalize=True, scale_each=True, pad_value=1)

                utils.save_image([data_loader.imnorm(content_image, None),
                                  data_loader.imnorm(style_image, None)]
                                 + [torch.ones(3, 256, 256) for _ in range(len(result_images)-2)]
                                 + result_images,
                                 '{}/ada_in_test_image_big_D_{}_{}.jpeg'.format(image_test_saving_path_list[-1], i, j),
                                 normalize=True, scale_each=False, pad_value=1, nrow=len(result_images))

                utils.save_image(result_images,
                                 '{}/ada_in_test_image_big_E_{}_{}.jpeg'.format(image_test_saving_path_list[-1], i, j),
                                 normalize=True, scale_each=True, pad_value=1)
