import torch

import torchvision.transforms as transforms

import sys
import yaml
import pprint

import train
import test_multiple
import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################################################
# configuration loading
########################################################################

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


configuration = get_config(sys.argv[1])
action = configuration['action']
print('the configuration used is:')
pprint.pprint(configuration, indent=4)


########################################################################
# image loaders and unloaders
########################################################################

# image size
imsize = configuration['imsize']

# loaders
loaders = {
    'std':      transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'no_norm':  transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor()])
}

# unloaders
unloaders = {
    'std':      transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                     transforms.ToPILImage()]),
    'saving':   transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]),
    'no_norm':  transforms.Compose(
                    [transforms.ToPILImage()]),
    'none':     None
}

configuration['loader'] = loaders[configuration['loader']]
configuration['unloader'] = unloaders[configuration['unloader']]

########################################################################
# main method
########################################################################

if __name__ == '__main__':
    if action == 'train':
        print('AdaINExt with moment alignment for > 2 moments')
        print('starting main training loop with specified configuration')
        train.train(configuration)
        sys.exit()
    if action == 'test':
        print('starting main test loop with specified configuration')
        test.test(configuration)
    if action == 'test_multiple':
        print('starting main next-to test loop with specified configuration')
        test_multiple.test(configuration)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # parameters
# action = sys.argv[1]
# data_path = sys.argv[2]
# working_directory = sys.argv[3]
# lambda_1 = float(sys.argv[4])
#
# number_moments = int(sys.argv[5])
# use_moment_alignment_model = 'True' == sys.argv[6]
# load_model = False
#
# try:
#     moment_alignment_model_number = int(sys.argv[7])
# except:
#     moment_alignment_model_number = 4
#
# try:
#     ada_in_module = int(sys.argv[8])
# except:
#     ada_in_module = number_moments
#
# print('moment alignment model number: {}'.format(moment_alignment_model_number))
#
# # mscoco dataset
# coco_data_path = '{}/cocodataset/train2017'.format(data_path)
# coco_data_path_train = '{}/cocodataset/train_80000'.format(data_path)
# coco_data_path_val = '{}/cocodataset/val_10000'.format(data_path)
#
# # painter-by-numbers dataset
# painter_by_numbers_data_path = '{}/painter-by-numbers/train'.format(data_path)
# painter_by_numbers_data_path_train = '{}/painter-by-numbers/train_80000'.format(data_path)
# painter_by_numbers_data_path_val = '{}/painter-by-numbers/val_10000'.format(data_path)
#
# # model saving
# model_saving_path = '{}/ada_in_ext_models'.format(working_directory)
# image_saving_path = '{}/ada_in_ext_images'.format(working_directory)
#
# # MA model
# moment_alignment_model_path = '{}/../bachelor_thesis/pytorch_models/mom_al_model_{}'.format(data_path, moment_alignment_model_number)
# moment_alignment_model_path_test = '../pytorch_models/mom_al_model_{}'.format(moment_alignment_model_number)
# # moment_alignment_model_path = '{}/../../pytorch_models/mom_al_model_4'.format(data_path)
#
# # tensorboardX log saving
# tensorboardX_path = '{}/ada_in_ext_run'.format(working_directory)
#
# torch_lua_vgg_model_path = '{}/../bachelor_thesis/pytorch_models/vgg_r41.pth'.format(data_path)
#
# # image size
# imsize = 512 if device == 'cuda' else 256  # size of images
#
# # transformations on input image
# loader = transforms.Compose(
#     [transforms.Resize(imsize),
#      transforms.RandomResizedCrop(256),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# # test loader
# test_loader = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(256),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# # transform image back to PIL image (and undo normalization)
# plotting_unloader = transforms.Compose(
#     [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
#      transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
#      transforms.ToPILImage()])
#
# # transform image back to PIL image (and undo normalization)
# saving_unloader = transforms.Compose(
#     [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
#      transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
#
# lst_loader = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(256),
#      transforms.ToTensor()])
#
# lst_unloader = None
#
# # for the actual training
# train_configuration = {
#     'epochs': 50,
#     'epoch_saving_interval': 1,
#     'step_printing_interval': 1000,
#     'image_saving_interval': 1000,
#     'validation_interval': 20000,
#     'coco_data_path': coco_data_path,
#     'coco_data_path_train': coco_data_path_train,
#     'coco_data_path_val': coco_data_path_val,
#     'painter_by_numbers_data_path': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_train': painter_by_numbers_data_path_train,
#     'painter_by_numbers_data_path_val': painter_by_numbers_data_path_val,
#     'model_saving_path': model_saving_path,
#     'image_saving_path': image_saving_path,
#     'batch_size': 8,
#     'lambda_1': lambda_1,
#     'load_model': load_model,
#     'number_moments': number_moments,
#     'ada_in_module': ada_in_module,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'use_moment_alignment_model': use_moment_alignment_model,
#     'moment_alignment_model_path': moment_alignment_model_path,
#     'lr': 0.0001,
#     'model_dir': '{}/torchvision_models/'.format(data_path),
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'tensorboardX_path': tensorboardX_path,
#     'torch_lua_vgg_model_path': torch_lua_vgg_model_path,
#     'number_moments_mom_al': moment_alignment_model_number,
# }
#
# # testing the training locally
# train_configuration_test = {
#     'epochs': 50,
#     'epoch_saving_interval': 1,
#     'step_printing_interval': 1000,
#     'image_saving_interval': 1000,
#     'validation_interval': 20000,
#     'coco_data_path': coco_data_path,
#     'coco_data_path_train': coco_data_path,
#     'coco_data_path_val': coco_data_path,
#     'painter_by_numbers_data_path': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_train': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_val': painter_by_numbers_data_path,
#     'model_saving_path': model_saving_path,
#     'image_saving_path': image_saving_path,
#     'batch_size': 8,
#     'lambda_1': lambda_1,
#     'load_model': load_model,
#     'number_moments': number_moments,
#     'ada_in_module': ada_in_module,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'use_moment_alignment_model': use_moment_alignment_model,
#     'moment_alignment_model_path': moment_alignment_model_path_test,
#     'lr': 0.0001,
#     'model_dir': '{}/torchvision_models/'.format(data_path),
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'tensorboardX_path': tensorboardX_path,
#     'torch_lua_vgg_model_path': '../pytorch_models/vgg_r41.pth',
#     'number_moments_mom_al': moment_alignment_model_number,
# }
#
# style_images_path = '../testset_style'
# content_images_path = '../testset_content'
# # content_images_path = '/Users/Johannes/Desktop/small_content_test'
# # style_images_path = '/Users/Johannes/Desktop/small_style_test'
# #model_path = '../pytorch_models/ada_in_model_66__22_05__14_03_46'
# model_path = '/Users/Johannes/Desktop/ada_in_exp/ada_in_model_2_2_10'
# image_test_saving_path_ada_in = './result_images_ada_in_single'
# moment_alignment_model_number = 2
#
# # # ---- result images AdaIN + Loss Extend ---- #
# #
# # image_saving_path_list = [
# #     './result_images_ada_in_2_1_10',
# #     './result_images_ada_in_2_2_10',
# #     './result_images_ada_in_2_3_10',
# #     './result_images_ada_in_2_4_10',
# #     './result_images_AdaIN+loss_big'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_2_1_10' + suffix,
# #     prefix + 'ada_in_model_2_2_10' + suffix,
# #     prefix + 'ada_in_model_2_3_10_n' + suffix,
# #     prefix + 'ada_in_model_2_4_10_n' + suffix,
# # ]
# #
# # number_moments_list = [1, 2, 3, 4]
# # ada_in_module_list = [2, 2, 2, 2]
# #
# # use_moment_alignment_model_list = [False, False, False, False]
# # moment_alignment_model_path_list = ['' for _ in range(4)]
# # moment_alignment_model_number_list = [-1, -1, -1, -1]
#
# # ---- result images MA + Loss Extend ---- #
#
# image_saving_path_list = [
#     './result_images_ada_in_1_1_10',
#     './result_images_ada_in_2_2_10',
#     './result_images_ada_in_ma_3_3_10',
#     './result_images_ada_in_ma_4_4_10',
#     './result_images_ma+loss_big'
# ]
#
# prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# suffix = ''
# model_path_list = [
#     prefix + 'ada_in_model_1_1_10' + suffix,
#     prefix + 'ada_in_model_2_2_10' + suffix,
#     prefix + 'ada_in_model_3_3_10_n' + suffix,
#     prefix + 'ada_in_model_4_4_10_n' + suffix,
# ]
#
# number_moments_list = [1, 2, 3, 4]
# ada_in_module_list = [1, 2, 2, 2]
#
# moment_alignment_model_number_list = [-1, -1, 3, 4]
#
# use_moment_alignment_model_list = [False, False, True, True]
#
# moment_alignment_model_path_list = [
#     -1, -1,
#     '../pytorch_models/mom_al_model_3',
#     '../pytorch_models/mom_al_model_4'
# ]
#
# # comparison of ada in models trained with factors 100 and 100000 vs. 1000 and 1000000
#
# # image_saving_path_list = [
# #     './result_images_ada_in_ma_3_3_10_o',
# #     './result_images_ada_in_ma_3_3_10',
# #     './result_images_ada_in_ma_4_4_10_o',
# #     './result_images_ada_in_ma_4_4_10',
# #     './result_images_ma_o_n'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_3_3_10_o' + suffix,
# #     prefix + 'ada_in_model_3_3_10_n' + suffix,
# #     prefix + 'ada_in_model_4_4_10_o' + suffix,
# #     prefix + 'ada_in_model_4_4_10_n' + suffix,
# # ]
# #
# # number_moments_list = [3, 3, 4, 4]
# # ada_in_module_list = [3, 3, 4, 4]
# #
# # moment_alignment_model_number_list = [3, 3, 4, 4]
# #
# # use_moment_alignment_model_list = [True, True, True, True]
# #
# # moment_alignment_model_path_list = [
# #     '../pytorch_models/mom_al_model_3',
# #     '../pytorch_models/mom_al_model_3',
# #     '../pytorch_models/mom_al_model_4',
# #     '../pytorch_models/mom_al_model_4'
# # ]
#
# # ---- result images MA + AdaIN Loss ---- #
#
# # image_saving_path_list = [
# #     './result_images_ada_in_1_2_10',
# #     './result_images_ada_in_2_2_10',
# #     './result_images_ada_in_ma_3_2_10',
# #     './result_images_ada_in_ma_4_2_10',
# #     './result_images_ada_in_ma_5_2_10',
# #     './result_images_ma-loss_big'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_1_2_10' + suffix,
# #     prefix + 'ada_in_model_2_2_10' + suffix,
# #     prefix + 'ada_in_model_3_2_10' + suffix,
# #     prefix + 'ada_in_model_4_2_10' + suffix,
# #     prefix + 'ada_in_model_5_2_10' + suffix,
# # ]
# #
# # number_moments_list = [1, 2, 3, 4, 5]
# #
# # ada_in_module_list = [1, 2, 2, 2, 2]
# #
# # moment_alignment_model_number_list = [-1, -1, 3, 4, 5]
# #
# # use_moment_alignment_model_list = [False, False, True, True, True]
# #
# # moment_alignment_model_path_list = [
# #     -1, -1,
# #     '../pytorch_models/mom_al_model_3',
# #     '../pytorch_models/mom_al_model_4',
# #     '../pytorch_models/mom_al_model_5'
# # ]
#
#
# # # ---- result images MA + AdaIN Ext Loss ---- #
# #
# # image_saving_path_list = [
# #     './result_images_ada_in_2_3_10',
# #     './result_images_ada_in_ma_3_3_10',
# #     './result_images_ada_in_2_4_10',
# #     './result_images_ada_in_ma_4_4_10',
# #     './result_images_ma+loss_vs_AdaIN+loss_big'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_2_3_10_n' + suffix,
# #     prefix + 'ada_in_model_3_3_10_n' + suffix,
# #     prefix + 'ada_in_model_2_4_10_n' + suffix,
# #     prefix + 'ada_in_model_4_4_10_n' + suffix,
# # ]
# #
# # number_moments_list = [3, 3, 4, 4]
# #
# # ada_in_module_list = [2, -1, 2, -1]
# #
# # moment_alignment_model_number_list = [-1, 3, -1, 4]
# #
# # use_moment_alignment_model_list = [False, True, False, True]
# #
# # moment_alignment_model_path_list = [
# #     -1,
# #     '../pytorch_models/mom_al_model_3',
# #     -1,
# #     '../pytorch_models/mom_al_model_4',
# # ]
#
# # # ---- result images AdaIN different lambda balancing factors ---- #
# #
# # image_saving_path_list = [
# #     './result_images_ada_in_2_2_05_t',
# #     './result_images_ada_in_2_2_1_t',
# #     './result_images_ada_in_2_2_10_t',
# #     './result_images_ada_in_2_2_100_t',
# #     './result_images_ba_factor_comp'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ba_ada_in_factor_comp/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_2_2_05' + suffix,
# #     prefix + 'ada_in_model_2_2_1' + suffix,
# #     prefix + 'ada_in_model_2_2_10' + suffix,
# #     prefix + 'ada_in_model_2_2_100' + suffix,
# # ]
# #
# # number_moments_list = [2, 2, 2, 2]
# #
# # ada_in_module_list = [2, 2, 2, 2]
# #
# # moment_alignment_model_number_list = [-1, -1, -1, -1]
# #
# # use_moment_alignment_model_list = [False, False, False, False]
# #
# # moment_alignment_model_path_list = [
# #     -1, -1, -1, -1
# # ]
#
# # # ---- result images AdaIN 1-2, 2-2 ---- #
# #
# # image_saving_path_list = [
# #     './result_images_ada_in_1_2_10',
# #     './result_images_ada_in_2_2_10',
# #     './result_images_ba_12_22'
# # ]
# #
# # prefix = '/Users/Johannes/Desktop/ada_in_module_loss/'
# # suffix = ''
# # model_path_list = [
# #     prefix + 'ada_in_model_1_2_10' + suffix,
# #     prefix + 'ada_in_model_2_2_10' + suffix,
# # ]
# #
# # number_moments_list = [2, 2]
# #
# # ada_in_module_list = [1, 2]
# #
# # moment_alignment_model_number_list = [-1, -1]
# #
# # use_moment_alignment_model_list = [False, False]
# #
# # moment_alignment_model_path_list = [
# #     -1, -1
# # ]
#
# prefix = '/Users/Johannes/Desktop/clean/ada_in_module_loss/'
# suffix = ''
#
# model_path = prefix + 'ada_in_model_3_3_10_n' + suffix
# image_test_saving_path_ada_in = prefix + '../../ba_praes_results'
# style_images_path = '/Users/Johannes/Desktop/testset_style'
# content_images_path = '/Users/Johannes/Desktop/content_image'
# use_moment_alignment_model_list = False
# moment_alignment_model_number = 3
#
# # the test configuration
# test_configuration_ada_in = {
#     'style_images_path': style_images_path,
#     'content_images_path': content_images_path,
#     'image_saving_path': image_test_saving_path_ada_in,
#     'image_saving_path_list': image_saving_path_list,
#     'model_path': model_path,
#     'model_path_list': model_path_list,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'saving_unloader': lst_unloader,
#     'number_moments': 4,
#     'number_moments_list': number_moments_list,
#     'ada_in_module_list': ada_in_module_list,
#     'ada_in_module': 4,
#     'lr': 0.001,
#     'model_dir': '/Users/Johannes',
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'use_moment_alignment_model': True,
#     'use_moment_alignment_model_list': use_moment_alignment_model_list,
#     'moment_alignment_model_path': '{}/../../pytorch_models/mom_al_model_3'.format(data_path),
#     'moment_alignment_model_path_list': moment_alignment_model_path_list,
#     'torch_lua_vgg_model_path': '../pytorch_models/vgg_r41.pth',
#     'number_moments_mom_al': moment_alignment_model_number,
#     'number_moments_mom_al_list': moment_alignment_model_number_list,
# }
#
# if __name__ == '__main__':
#     if action == 'train':
#         print('AdaINExt with moment alignment for > 2 moments')
#         print('starting main training loop with specified configuration')
#         train.train(train_configuration)
#         sys.exit()
#     if action == 'train_test':
#         print('AdaINExt with moment alignment for > 2 moments')
#         print('starting main training loop with specified configuration')
#         train.train(train_configuration_test)
#         sys.exit()
#     if action == 'test':
#         print('starting main test loop with specified configuration')
#         test.test(test_configuration_ada_in)
#     if action == 'test_single_images':
#         print('starting main test loop with specified configuration')
#         test_single_images.test(test_configuration_ada_in)
#     if action == 'test_next_to':
#         print('starting main next-to test loop with specified configuration')
#         test_next_to_each_other.test(test_configuration_ada_in)
#
