import torch
import torch.nn as nn
import timm
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class TimmEncoder(nn.Module):
    def __init__(self, cfg, output_stride=32):
        super().__init__()

        depth = len(cfg.out_indices)
        self.model = timm.create_model(
            cfg.backbone,
            in_chans=cfg.in_channels,
            pretrained=True,
            num_classes=0,
            features_only=True,
            output_stride=output_stride if output_stride != 32 else None,
            out_indices=cfg.out_indices,
        )
        self._in_channels = cfg.in_channels
        self._out_channels = [
            cfg.in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride  # 32

    def forward(self, x):
        features = self.model(x)

        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


class Unet(nn.Module):
    def __init__(self, args, decoder_use_batchnorm: bool = True):
        super().__init__()

        encoder_name = args.backbone

        self.encoder = TimmEncoder(args)

        encoder_depth = len(self.encoder.out_channels) - 1

        decoder_channels = args.dec_channels[:encoder_depth]
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=args.dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], args.n_classes, kernel_size=3, padding=1)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)
        features = [None] + features
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        return masks
