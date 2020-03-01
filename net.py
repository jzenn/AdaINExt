import torch
import torch.nn as nn
import torchvision.models as models

import copy

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaINTorchModel(nn.Module):
    def __init__(self, encoder, decoder, ada_in_layer, number_moments=2,
                 use_moment_alignment_model=False, number_moments_mom_al=False):
        super(AdaINTorchModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adaIN_layer = ada_in_layer
        self.mse_loss = nn.MSELoss()
        self.use_moment_alignment_model = use_moment_alignment_model
        self.number_moments_mom_al = number_moments_mom_al
        self.number_moments = number_moments

        print('using {} moments for loss'.format(self.number_moments))

        # no gradients in encoder required
        for param in encoder.parameters():
            param.requires_grad = False

    def forward(self, style_input, content_input, compute_loss=True):
        # encode style and get style features
        style_input_encoded, style_input_features = self.get_intermediate_features(style_input)

        # encode content and get style features
        content_input_encoded, content_input_features = self.get_intermediate_features(content_input)

        if self.use_moment_alignment_model:
            n, c, h, w = content_input_encoded.size()
            style_input_moments = utils.compute_moments_1d_batches(
                style_input_encoded.view(n * c, 1, h, w), last_moment=self.number_moments_mom_al)

            content_input_moments = utils.compute_moments_1d_batches(
                content_input_encoded.view(n * c, 1, h, w), last_moment=self.number_moments_mom_al)

            ada_in_output = self.adaIN_layer(content_input_encoded.view(n * c, 1, h, w),
                                             content_input_moments, style_input_moments)
            ada_in_output = ada_in_output.view(n, c, h, w)
        else:
            # normalize
            ada_in_output = self.adaIN_layer(content_input_encoded, style_input_encoded)

        # decode output of AdaIN layer
        decoded_ada_in = self.decoder(ada_in_output)

        if not compute_loss:
            return decoded_ada_in.to(device), 0, 0

        # encode decoded output of AdaIN layer for loss calculation
        encoded_decoded_ada_in_output, encoded_decoded_ada_in_features = \
            self.get_intermediate_features(decoded_ada_in)

        # content loss
        content_loss = self.calculate_content_loss(encoded_decoded_ada_in_output, ada_in_output)

        # style loss
        style_loss = 0
        for i in range(len(encoded_decoded_ada_in_features)):
            layer_loss = self.calculate_style_loss(encoded_decoded_ada_in_features[i], style_input_features[i])
            style_loss += 1/4 * layer_loss

        return decoded_ada_in.to(device), content_loss.to(device), style_loss.to(device)

    def get_intermediate_features(self, input):
        encoded_input = self.encoder(input)
        feat_1 = encoded_input['r11']
        feat_2 = encoded_input['r21']
        feat_3 = encoded_input['r31']
        feat_4 = encoded_input['r41']

        return feat_4, [feat_1, feat_2, feat_3, feat_4]

    def calculate_content_loss(self, input, target):
        assert (input.size() == target.size()), \
            'There is something wrong with the sizes {} and {}'.format(input.size(), target.size())
        loss = self.mse_loss(input, target)
        return loss

    def calculate_style_loss(self, input, target):
        assert (input.size() == target.size()), \
            'There is something wrong with the sizes {} and {}'.format(input.size(), target.size())

        input_mean, input_std, input_skew, input_kurtosis = utils.compute_moments(input)
        target_mean, target_std, target_skew, target_kurtosis = utils.compute_moments(target)

        # mean norm factor
        target_mean_norm_factor = target_mean.clone().to(device)
        target_mean_norm_factor[abs(target_mean_norm_factor) < 1] = 1
        target_mean_norm_factor = 1 / torch.mean(target_mean_norm_factor)

        # std norm factor
        target_std_norm_factor = target_std.clone().to(device)
        target_std_norm_factor[abs(target_std_norm_factor) < 1] = 1
        target_std_norm_factor = 1 / torch.mean(target_std_norm_factor)

        # loss
        loss = target_mean_norm_factor * self.mse_loss(input_mean, target_mean)\

        if self.number_moments > 1:
           loss += target_std_norm_factor * self.mse_loss(input_std, target_std)

        if self.number_moments > 2:
            target_skew_norm_factor = target_skew.clone().to(device)
            target_skew_norm_factor[abs(target_skew_norm_factor) < 1] = 1
            target_skew_norm_factor = 1 / torch.mean(target_skew_norm_factor)
            skew_add_fac = 1 / 1000
            loss += skew_add_fac * target_skew_norm_factor * self.mse_loss(input_skew, target_skew)

        if self.number_moments > 3:
            target_kurtosis_norm_factor = target_kurtosis.clone().to(device)
            target_kurtosis_norm_factor[abs(target_kurtosis_norm_factor) < 1] = 1
            target_kurtosis_norm_factor = 1 / torch.mean(target_kurtosis_norm_factor)
            kurt_add_fac = 1 / 1000000
            loss += kurt_add_fac * target_kurtosis_norm_factor * self.mse_loss(input_kurtosis, target_kurtosis)

        return loss


class AdaptiveInstanceNormalizationMean(nn.Module):
    """
    aligns the mean of two tensors channel-wise
    """
    def __init__(self):
        super(AdaptiveInstanceNormalizationMean, self).__init__()

    def forward(self, input_content_features, input_style_features):
        assert (len(input_content_features.size()) == len(input_style_features.size())), \
            'the sizes of the content and style feature maps should be equal'

        feature_size = input_content_features.size()

        mean_content, _ = utils.calc_mean_and_std(input_content_features)
        mean_style, _ = utils.calc_mean_and_std(input_style_features)

        return input_content_features - mean_content.expand(feature_size) + mean_style.expand(feature_size)


class AdaptiveInstanceNormalization(nn.Module):
    """
    aligns the std and mean of two tensors channel-wise
    """
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, input_content_features, input_style_features):
        assert (len(input_content_features.size()) == len(input_style_features.size())), \
            'the sizes of the content and style feature maps should be equal'

        feature_size = input_content_features.size()

        mean_content, std_content = utils.calc_mean_and_std(input_content_features)
        mean_style, std_style = utils.calc_mean_and_std(input_style_features)

        return std_style.expand(feature_size) \
               * ((input_content_features - mean_content.expand(feature_size)) / std_content.expand(feature_size)) \
               + mean_style.expand(feature_size)


class MomentAlignment(nn.Module):
    def __init__(self, mode=1, in_channels=1):
        super(MomentAlignment, self).__init__()
        self.mode = mode

        net = [
            nn.Conv2d(in_channels, 15, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(15, 10, 1, 1, 0),
            nn.ReLU()
        ]

        for _ in range(self.mode-1):
            net.append(nn.Conv2d(10, 10, 1, 1, 0))
            net.append(nn.ReLU())

        net.append(nn.Conv2d(10, 1, 1, 1, 0))

        self.net = nn.Sequential(*net)

    def forward(self, feature_map, feature_map_moments, new_moments):
        # (n, c, h, w) content feature map size
        n, c, h, w = feature_map.size()
        feature_map = feature_map.to(device)

        # produce input to net
        for i in range(self.mode):
            target_moment_layer = new_moments[i].expand(n, c, h, w).to(device)
            input_moment_layer = feature_map_moments[i].expand(n, c, h, w).to(device)
            feature_map = torch.cat([feature_map, target_moment_layer, input_moment_layer], 1)

        # forward pass
        out = self.net(feature_map)

        return out.to(device)


def get_encoder(cnn_enc):
    """
    constructs the encoder based on the given cnn_enc
    :param cnn_enc:
    :return: an encoder Sequential object
    """
    cnn_encoder = copy.deepcopy(cnn_enc)
    model = nn.Sequential()
    # feature_maps = []

    i = 1
    j = 1
    # construct the encoder
    for layer in cnn_encoder:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}_{}'.format(i, j)
            module = layer
            layer.padding = (0, 0)
            model.add_module(name='reflection_padding_{}_{}'.format(i, j), module=nn.ReflectionPad2d((1, 1, 1, 1)))
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            module = layer
            j = 1
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}_{}'.format(i, j)
            module = layer
            j += 1
        else:
            raise RuntimeError('unrecognized layer')
        model.add_module(name, module)
        if name == "relu_4_1":
            break
    model.add_module(name=name, module=module)

    return model


def get_decoder():
    model = nn.Sequential()
    model.add_module(name='reflection_padding_1_1', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_1', module=nn.Conv2d(512, 256, 3, 1, 0))
    model.add_module(name='relu_1_1', module=nn.ReLU(inplace=True))

    model.add_module(name='upsampling_1', module=nn.UpsamplingNearest2d(scale_factor=2))

    model.add_module(name='reflection_padding_1_2', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_2', module=nn.Conv2d(256, 256, 3, 1, 0))
    model.add_module(name='relu_1_2', module=nn.ReLU(inplace=True))

    model.add_module(name='reflection_padding_1_3', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_3', module=nn.Conv2d(256, 256, 3, 1, 0))
    model.add_module(name='relu_1_3', module=nn.ReLU(inplace=True))

    model.add_module(name='reflection_padding_1_4', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_4', module=nn.Conv2d(256, 256, 3, 1, 0))
    model.add_module(name='relu_1_4', module=nn.ReLU(inplace=True))

    model.add_module(name='reflection_padding_1_5', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_5', module=nn.Conv2d(256, 128, 3, 1, 0))
    model.add_module(name='relu_1_5', module=nn.ReLU(inplace=True))

    model.add_module(name='upsampling_2', module=nn.UpsamplingNearest2d(scale_factor=2))

    model.add_module(name='reflection_padding_1_6', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_6', module=nn.Conv2d(128, 128, 3, 1, 0))
    model.add_module(name='relu_1_6', module=nn.ReLU(inplace=True))

    model.add_module(name='reflection_padding_1_7', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_7', module=nn.Conv2d(128, 64, 3, 1, 0))
    model.add_module(name='relu_1_7', module=nn.ReLU(inplace=True))

    model.add_module(name='upsampling_3', module=nn.UpsamplingNearest2d(scale_factor=2))

    model.add_module(name='reflection_padding_1_8', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_8', module=nn.Conv2d(64, 64, 3, 1, 0))
    model.add_module(name='relu_1_8', module=nn.ReLU(inplace=True))

    model.add_module(name='reflection_padding_1_9', module=nn.ReflectionPad2d((1, 1, 1, 1)))
    model.add_module(name='conv_1_9', module=nn.Conv2d(64, 3, 3, 1, 0))

    model.add_module(name='sigmoid_1', module=nn.Sigmoid())

    return model


def get_ada_in_model(configuration, use_list=False, list_index=-1):
    encoder_model_path = configuration['encoder_model_path']

    if use_list:
        use_moment_alignment_model = configuration['use_moment_alignment_model_list'][list_index]
        moment_alignment_model_path = configuration['moment_alignment_model_path_list'][list_index]
        number_moments = configuration['number_moments_list'][list_index]
        ada_in_module = configuration['ada_in_module_list'][list_index]
        number_moments_mom_al = configuration['moment_alignment_model_number_list'][list_index]
    else:
        use_moment_alignment_model = configuration['use_moment_alignment_model']
        moment_alignment_model_path = configuration['moment_alignment_model_path']
        number_moments = configuration['number_moments']
        ada_in_module = configuration['ada_in_module']
        number_moments_mom_al = configuration['number_moments_mom_al']

    encoder = Encoder()

    vgg_pre_trained_state_dict = torch.load(encoder_model_path, map_location='cpu')
    encoder.load_state_dict(vgg_pre_trained_state_dict)

    print('using {} moments for loss computation'.format(number_moments))

    if use_moment_alignment_model:
        # the MomAl layer
        ada_in_layer = MomentAlignment(mode=number_moments_mom_al, in_channels=2*number_moments_mom_al+1)
        checkpoint = torch.load(moment_alignment_model_path, map_location='cpu')

        for name, param in ada_in_layer.named_parameters():
            param.requires_grad = False

        # original saved file with DataParallel
        # create new OrderedDict that does not contain `module.`

        # uncomment here if not working #
        # uncomment here if not working #
        # uncomment here if not working #
        # uncomment here if not working #

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['model_state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # ada_in_layer.load_state_dict(new_state_dict)

    else:
        # the AdaIN layer
        if ada_in_module == 1:
            ada_in_layer = AdaptiveInstanceNormalizationMean()
        else:
            ada_in_layer = AdaptiveInstanceNormalization()

    # the decoder
    # decoder = get_decoder()

    decoder = Decoder()

    # the model
    ada_in_model = AdaINTorchModel(encoder, decoder, ada_in_layer, number_moments=number_moments,
                                   use_moment_alignment_model=use_moment_alignment_model,
                                   number_moments_mom_al=number_moments_mom_al)

    print('the ada-in model')
    print(ada_in_model)

    # use the max amount of GPUs possible
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ada_in_model.to(device)
        ada_in_model = nn.DataParallel(ada_in_model)
    else:
        ada_in_model.to(device)
        print('using {}'.format(device))

    print('printing whole model params which require_grad')
    for name, param in ada_in_model.named_parameters():
        if param.requires_grad:
            print(name)

    return ada_in_model


class encoder4(nn.Module):
    def __init__(self):
        super(encoder4,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self,x,sF=None,matrix11=None,matrix21=None,matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])

        out = self.conv3(out)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12'])
        out = self.reflecPad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])

        out = self.conv5(out)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if(matrix31 is not None):
            feature3,transmatrix3 = matrix31(output['r31'],sF['r31'])
            out = self.reflecPad7(feature3)
        else:
            out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)

        return output


class decoder4(nn.Module):
    def __init__(self):
        super(decoder4,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out

class Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, 3, 3, 1, 0)

        # sigmoid
        self.sig_4_2 = nn.Sigmoid()

    def forward(self, input):
        # first block
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)

        # sigmoid
        out = self.sig_4_2(out)

        return out


class Encoder(nn.Module):
    """
    the encoder network
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        self.conv_1_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = {}

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)

        output['r11'] = out

        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)

        out = self.maxPool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)

        output['r21'] = out

        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)

        out = self.maxPool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)

        output['r31'] = out

        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)

        out = self.maxPool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)

        output['r41'] = out

        return output
