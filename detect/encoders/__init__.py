import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders


encoders = {}
encoders.update(resnet_encoders)


def get_encoder(name, pretrained=True):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(encoders[name]['pretrained_url']))
    return encoder


def get_encoder_names():
    return list(encoders.keys())

