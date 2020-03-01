import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

import utils
import data_loader
import net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(configuration):
    """
    this is the main training loop
    :param configuration:
    :return:
    """
    epochs = configuration['epochs']
    batch_size = configuration['batch_size']
    print('going to train for {} epochs with a batch size of {}'.format(epochs, batch_size))

    step_printing_interval = configuration['step_printing_interval']
    print('writing to console every {} steps'.format(step_printing_interval))

    epoch_saving_interval = configuration['epoch_saving_interval']
    print('saving the model every {} epochs'.format(epoch_saving_interval))

    image_saving_interval = configuration['image_saving_interval']
    print('saving the images every {} steps'.format(image_saving_interval))

    validation_interval = configuration['validation_interval']
    print('validating the model every {} steps'.format(validation_interval))

    epoch = 0
    print('starting in epoch {}'.format(epoch))

    number_moments = configuration['number_moments']
    print('use {} moments for loss'.format(number_moments))

    loader = configuration['loader']

    coco_data_path_train = configuration['coco_data_path_train']
    painter_by_numbers_data_path_train = configuration['painter_by_numbers_data_path_train']
    print('using {} and {} for training'.format(coco_data_path_train, painter_by_numbers_data_path_train))

    coco_data_path_val = configuration['coco_data_path_val']
    painter_by_numbers_data_path_val = configuration['painter_by_numbers_data_path_val']
    print('using {} and {} for validation'.format(coco_data_path_val, painter_by_numbers_data_path_val))

    train_dataloader = data_loader.get_concat_dataloader(
        coco_data_path_train, painter_by_numbers_data_path_train, batch_size, loader)
    print('got train dataloader')

    val_dataloader = data_loader.get_concat_dataloader(
        coco_data_path_val, painter_by_numbers_data_path_val, batch_size, loader)
    print('got val dataloader')

    tensorboardX_path = configuration['tensorboardX_path']
    writer = SummaryWriter(logdir='{}/runs'.format(tensorboardX_path))
    print('saving tensorboardX logs to {}'.format(tensorboardX_path))

    # ada_in_model = models.net.get_ada_in_model(configuration)
    ada_in_model = net.get_ada_in_model(configuration)
    print('got model')

    if configuration['use_pretrained_model']:
        model_path = configuration['model_saving_path']
        print('loading a AdaIN model from {}'.format(model_path))
        model_path = utils.get_latest_model(model_path)
        checkpoint = torch.load(model_path, map_location=device)
        print('loaded checkpoint')
        ada_in_model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded state dict')
        epoch += checkpoint['epoch']
        epochs += epoch

        optimizer = optim.Adam(ada_in_model.parameters())
        print('got optimizer')

        print('loading optimizer state dict')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loaded optimizer state dict')
    else:
        optimizer = optim.Adam(ada_in_model.parameters(), lr=configuration['lr'])
        print('got optimizer')

    lambda_1 = configuration['lambda_1']
    print('using style lambda of {}'.format(lambda_1))

    number_of_validation = 0
    all_training_iteration = 0
    current_validation_loss = float('inf')

    while epoch < epochs:
        epoch += 1
        print('epoch: {}'.format(epoch))

        print('enumerating data loader')
        train_data_loader_enumerated = enumerate(train_dataloader)
        print('data_loader enumerated')

        while True:

            if all_training_iteration % validation_interval == 0:
                print('enumerating validation data loader')
                val_data_loader_enumerated = enumerate(val_dataloader)
                print('validation data_loader enumerated')

                validation_loss = validate(number_of_validation, val_data_loader_enumerated, ada_in_model, lambda_1, writer)

                number_of_validation += 1

                if validation_loss < current_validation_loss:
                    utils.save_current_best_model(epoch, ada_in_model.state_dict(), optimizer.state_dict(),
                                                  validation_loss, configuration['model_saving_path'])
                    print('got a better model')
                    current_validation_loss = validation_loss
                    print('set the new validation loss to the current one')
                else:
                    print('this model is actually worse than the best one')
                    print('returning and exiting ...')

            try:
                i, data = train_data_loader_enumerated.__next__()
            except StopIteration:
                break
            except:
                print('something went wrong with the dataloader')
                continue

            # count up number of iterations
            all_training_iteration += 1

            # set all the gradients to zero
            optimizer.zero_grad()

            # get the content_image batch
            content_image = data.get('coco').get('image')
            content_image = content_image.to(device)

            # get the style_image batch
            style_image = data.get('painter_by_numbers').get('image')
            style_image = style_image.to(device)

            result_image, content_loss, style_loss = ada_in_model(style_image, content_image)

            # convert losses to scalars (from each GPU we get an extra loss)
            content_loss = content_loss.mean()
            style_loss = style_loss.mean()

            # loss is sum of content and style loss
            loss = content_loss + lambda_1 * style_loss

            # backprop
            loss.backward()

            # make one step
            optimizer.step()

            # print every step_printing_interval the loss
            if i % step_printing_interval == 0:
                print('epoch {}, step {}'.format(epoch, i))
                print('style_loss: {:4f}, content_loss: {:4f}'.format(style_loss.item(), content_loss.item()))

                writer.add_scalar('data/training_loss', loss.item(), all_training_iteration)
                writer.add_scalar('data/training_content_loss', content_loss.item(), all_training_iteration)
                writer.add_scalar('data/training_style_loss', style_loss.item(), all_training_iteration)

            # save every epoch_saving_interval the current model
            if i % image_saving_interval == 0 and epoch % epoch_saving_interval == 0:
                utils.save_current_model(epoch, ada_in_model.state_dict(), optimizer.state_dict(),
                                         content_loss, style_loss, configuration['model_saving_path'])

            # save every image_saving_interval the processed images
            if i % image_saving_interval == 0:
                utils.save_images(epoch, i, configuration['number_moments'], content_image, style_image, result_image,
                                  configuration['unloader'], configuration['image_saving_path'], configuration['batch_size'])

            utils.decay_learning_rate(configuration, optimizer, all_training_iteration)


def validate(number_of_validation, validation_data_loader_enumerated, ada_in_model, lambda_1, writer):
    # accumulate loss to get the mean
    accumulated_loss = 0

    while True:
        try:
            i, data = validation_data_loader_enumerated.__next__()
        except StopIteration:
            break
        except:
            print('something went wrong with the dataloader')
            continue

        # no gradients required for validation
        with torch.no_grad():
            # get the content_image batch
            content_image = data.get('coco').get('image')
            content_image = content_image.to(device)

            # get the style_image batch
            style_image = data.get('painter_by_numbers').get('image')
            style_image = style_image.to(device)

            result_image, content_loss, style_loss = ada_in_model(style_image, content_image)

            # convert losses to scalars (from each GPU we get an extra loss)
            content_loss = content_loss.mean()
            style_loss = style_loss.mean()

            # loss is sum of content and style loss
            loss = content_loss + lambda_1 * style_loss
            # print()
            # print('cont: {}'.format(content_loss.item()))
            # print('style: {}'.format(style_loss.item()))
            # print('loss: {}'.format(loss.item()))
            # print()

            # accumulate loss further
            accumulated_loss += loss

    # get mean
    accumulated_loss /= 10000

    # write some logs
    writer.add_scalar('data/validation_loss_mean', accumulated_loss.item(), number_of_validation)
    print('#validation {}, loss {}'.format(number_of_validation, accumulated_loss))

    return accumulated_loss
