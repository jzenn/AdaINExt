import torch
import torchvision.utils as utils

import glob
import datetime
import pytz
import os

# the date-time format
fmt = '%d_%m__%H_%M_%S'


def decay_learning_rate(configuration, optimizer, sum_iterations):
    lr = configuration['lr'] / (1.0 + 1e-5 * sum_iterations)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_moments_1d_batches(input, last_moment=6):
    out = []
    for i in range(1, last_moment + 1):
        out += [compute_n_th_moment_batches(input, i)]

    return out


def compute_n_th_moment_batches(x, i):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    mean = torch.mean(x, dim=2).view(n, c, 1, 1)
    eps = 1e-5
    var = torch.var(x, dim=2).view(n, c, 1, 1) + eps
    std = torch.sqrt(var)
    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        sol = ((x.view(n, c, h, w) - mean.expand(n, c, h, w)) / std).pow(i)
        sol = torch.mean(sol.view(n, c, -1), dim=2).view(n, c, 1, 1)
        return sol


def compute_moments(input):
    out = []
    for i in range(1, 5):
        out += [compute_i_th_moment(input, i)]

    return out


def compute_i_th_moment(input, i):
    # get the input size
    input_size = input.size()

    # (n, c, h, w)
    n = input_size[0]
    c = input_size[1]

    mean = torch.mean(input.view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)

    eps = 1e-5
    var = torch.var(input.view(n, c, -1), dim=2, keepdim=True) + eps
    std = torch.sqrt(var).view(n, c, 1, 1)

    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        return torch.mean((((input - mean) / std).pow(i)).view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)


def calc_mean_and_std(input):
    """
    calculates mean and std channelwise (R,G,B)
    :param input:
    :return: mean an std (channelwise)
    """
    assert (len(input.size()) == 4), "the size of the feature map should not be {}".format(input.size())

    input_size = input.size()
    n = input_size[0]
    c = input_size[1]

    mean = torch.mean(input.view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)

    # prevent division by zero producing nan
    eps = 1e-5
    var = torch.var(input.view(n, c, -1), dim=2) + eps
    std = torch.sqrt(var).view(n, c, 1, 1)

    return mean, std


def save_current_model(epoch, model_state_dict, optimizer_state_dict, content_loss, style_loss, model_saving_path):
    """
    saves the given model
    :param epoch: the epoch
    :param model_state_dict: the model state dict
    :param optimizer_state_dict: the optimizer state dict
    :param content_loss: the content loss
    :param style_loss: the style loss
    :param model_saving_path: the model saving path
    :return: no return
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'content_loss': content_loss,
            'style_loss': style_loss
            },
        '{}/ada_in_model_{}__{}'.format(model_saving_path, epoch,
                                    datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt)))


def save_current_best_model(epoch, model_state_dict, optimizer_state_dict, validation_loss, model_saving_path):
    """
    saves the given model
    :param epoch: the epoch
    :param model_state_dict: the model state dict
    :param optimizer_state_dict: the optimizer state dict
    :param validation_loss: the validation loss
    :param model_saving_path: the model saving path
    :return: no return
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'validation_loss': validation_loss
            },
        '{}/ada_in_model'.format(model_saving_path))


def save_images(epoch, step, number_moments, ground_truth_c, ground_truth_s, result_image, transformation,
                image_saving_path, batch_size):
    """
    saves the images to the specified path
    :param epoch: the epoch
    :param step: the current step in the epoch
    :param ground_truth_c: the ground truth content image batch
    :param result_image_c: the content decoded image batch
    :param ground_truth_s: the ground truth style image batch
    :param result_image_s: the style decoded image batch
    :param transformation: the transformation applied before saving
    :param image_saving_path: the path the image is saved to
    :param batch_size: the batch size used
    :return: no return
    """

    rand = int(torch.rand(1).item() * batch_size)

    if transformation is not None:
        ground_truth_c = transformation(ground_truth_c[rand].cpu().clone().squeeze(0))
        ground_truth_s = transformation(ground_truth_s[rand].cpu().clone().squeeze(0))
    else:
        ground_truth_c = ground_truth_c[rand].cpu().clone().squeeze(0)
        ground_truth_s = ground_truth_s[rand].cpu().clone().squeeze(0)
    result_image = result_image[rand].cpu().clone().squeeze(0)

    utils.save_image([ground_truth_s, ground_truth_c, result_image],
                     '{}/ada_in_image_{}_{}_{}__{}.jpeg'.format(image_saving_path, epoch, step, number_moments,
                                                                     datetime.datetime.now(pytz.utc).astimezone(
                                                                         pytz.timezone('Europe/Berlin')).strftime(fmt)))

def get_latest_model(model_path):
    model_path = model_path + '/ada_in_model*'
    latest_file = max(glob.iglob(model_path), key=os.path.getctime)
    print('the latest model is obviously {}'.format(latest_file))
    return latest_file