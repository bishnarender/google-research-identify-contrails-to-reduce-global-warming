"""
1024x1024 input -> 512x512 output
Drop one upscale decoder layer
"""
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.initialization import initialize_decoder

from unet_decoder import UnetDecoder


def _check_reduction(reduction_factors):
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise AssertionError('Reduction assumed to increase by 2: {}'.format(reduction_factors))
        r_prev = r


def get_asym_conv(nc):
    if nc == 256:
        hidden_size = 9
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    elif nc == 512 or nc == 1024:
        hidden_size = 25
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(5, 5), padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    else:
        raise NotImplementedError

    return asym_conv


class Model(nn.Module):
    # See also TimmUniversalEncoder in segmentation_models_pytorch
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        name = cfg['model']['encoder']
        dropout = cfg['model']['dropout']
        pretrained = pretrained and cfg['model']['pretrained']

        self.encoder = timm.create_model(name, features_only=True, pretrained=pretrained)
        encoder_channels = self.encoder.feature_info.channels()

        _check_reduction(self.encoder.feature_info.reduction())

        decoder_channels = cfg['model']['decoder_channels']  # (256, 128, 64, 32, 16)
        print('Encoder channels:', name, encoder_channels)
        print('Decoder channels:', decoder_channels[:-1])    # -1 for 1024

        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-2],  # adhoc change for 1024. Usually -1
            out_channels=1, activation=None, kernel_size=3,
        )

        initialize_decoder(self.decoder)

        self.asym_conv = get_asym_conv(cfg['data']['resize'])

    def forward(self, x):
        #self.training = True # for torchview my_ 
        
        features = self.encoder(x)

        decoder_output = self.decoder(features)

        y_sym = self.segmentation_head(decoder_output)

        y_pred = self.asym_conv(y_sym)

        return y_sym, y_pred
