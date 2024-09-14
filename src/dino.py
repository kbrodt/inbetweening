import itertools
import math
import warnings
from functools import partial

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from PIL import Image


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `out_channels==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 ):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    #@auto_fp16()
    #@abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        #self.bn = nn.SyncBatchNorm(self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    @staticmethod
    def get_pad(multiple, size):
        new_size = math.ceil(size / multiple) * multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    #@torch.inference_mode()
    def forward(self, x):
        #print('centerpadding', x.shape)
        pads = list(itertools.chain.from_iterable(self.get_pad(self.multiple, m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        #print('centerpadding', output.shape)
        return output


class UnetDecoderINH(UnetDecoder):
    def forward(self, *features):
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if i < len(skips):
                skip = skips[i]
                skip = F.interpolate(skip, scale_factor=2 ** (i + 1), mode="nearest")
            else:
                skip = None
            x = decoder_block(x, skip)

        return x


class Unet(nn.Module):
    def __init__(
        self,
        backbone_name="vit_small_patch14_dinov2",
        n=4,#[8, 9, 10, 11],
        img_size=(518, 518),
        n_classes=1,
        decoder_channels=[256, 240, 224, 208, 192],
        decoder_chkp_path=None,
        decoder_use_batchnorm=True,
        dec_attn_type=None,
        pretrained=True,  # False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,  # False,
            #dynamic_img_size=True,
        )
        ps = self.backbone.patch_embed.proj.kernel_size
        assert ps[0] == ps[1]
        self.img_size = tuple(m + sum(CenterPadding.get_pad(ps[0], m)) for m in img_size)
        self.backbone.patch_embed.grid_size = (self.img_size[0] // ps[0], self.img_size[1] // ps[1])

        #backbone_name = "dinov2_vits14"
        #self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

        self.backbone.eval()
        if isinstance(n, int):
            assert 0 <= n < len(self.backbone.blocks)

        self.n = n if isinstance(n, list) else [len(self.backbone.blocks) - i for i in range(1, n + 1)]
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=self.n,
            reshape=True,
        )
        self.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(ps[0])(x[0])
        )

        emb_dim = self.backbone.embed_dim
        #self.decoder = BNHead(
        #    in_channels=[emb_dim] * len(self.n),
        #    in_index=list(range(len(self.n))),
        #    input_transform="resize_concat",
        #    channels=emb_dim * len(self.n),
        #    dropout_ratio=0,
        #    num_classes=n_classes,
        #    norm_cfg=dict(type="SyncBN", requires_grad=True),
        #    align_corners=False,
        #)

        #if decoder_chkp_path is not None:
        #    decoder_chkp = torch.load(
        #        decoder_chkp_path,
        #        map_location="cpu",
        #    )
        #    dsd = decoder_chkp["state_dict"]
        #    nn.modules.utils.consume_prefix_in_state_dict_if_present(
        #        dsd,
        #        prefix="decode_head.",
        #    )
        #    self.decoder.load_state_dict(dsd, strict=True)

        self.decoder = UnetDecoderINH(
            encoder_channels=[emb_dim] * (len(self.n) + 1),
            decoder_channels=decoder_channels[:len(self.n)],
            n_blocks=len(self.n),
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-2], n_classes, kernel_size=3, padding=1)

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        #with torch.inference_mode():
        self.backbone.eval()
        x = self.backbone(x)
        #x = self.decoder.forward(x)

        #features = self.encoder(x)
        decoder_output = self.decoder(*x)
        x = self.segmentation_head(decoder_output)

        return x

    def get_parameter_section(self, parameters, lr=None, wd=None, skip_list=()):
        parameter_settings = []

        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for _, (n, p) in enumerate(parameters):
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
                    #break

            if not layer_no:
                layer_no = 0

            if lr_is_dict:
                for k, v in lr.items():
                    if layer_no < k:
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k, v in wd.items():
                    if layer_no < k:
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            #weight_decay = 0.0 if 'bias' in n else temp_wd
            weight_decay = 0.0 if len(p.shape) == 1 or n.endswith(".bias") or n in skip_list else temp_wd
            parameter_setting = {
                "params": p,
                "lr": temp_lr,
                "weight_decay": weight_decay,
            }
            #print(n, layer_no, temp_lr)
            parameter_settings.append(parameter_setting)

        return parameter_settings



def render_segmentation(segmentation_logits):
    colormap = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
        (255, 255, 255),
    ]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]

    return Image.fromarray(segmentation_values)


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


def main():
    image = Image.open("./example.jpg").convert("RGB")
    print(image.size)
    image.show()
    model = Unet(
        img_size=(512, 512),#image.size[::-1],
        n_classes=21,
        decoder_chkp_path="../dinov2_vits14_voc2012_ms_head.pth",
    )

    #transform = mmcv.transforms.MultiScaleFlipAug(
    #    scales=(99999999, 640),
    #    scale_factor=[1.0],
    #    #allow_flip=True,
    #    transforms=[
    #        dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    #        dict(
    #            type='Normalize',
    #            mean=[123.675, 116.28, 103.53],
    #            std=[58.395, 57.12, 57.375],
    #            to_rgb=True),
    #        dict(type='ImageToTensor', keys=['img']),
    #        dict(type='ImageToTensor', keys=['img']),
    #    ],
    #)

    normalize = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(
                mean=np.array([123.675, 116.28, 103.53]) / 255,
                std=np.array([58.395, 57.12, 57.375]) / 255,
            ),
        ]
    )
    image = np.array(image)#[:, :, ::-1].copy()  # BGR
    x = normalize(image=image)["image"]
    print(x)
    print(x.shape)
    #x = transform({"img": image.copy(), "inputs": image.copy(), "data_sample": image.copy()})
    #print(x)
    x = np.transpose(x, (2, 0, 1)).astype("float32")
    x = torch.from_numpy(x).unsqueeze(0)
    print(x.shape)

    #x = torch.randn(1, 3, 512, 512)
    with torch.inference_mode():
        logits = model(x)

    logits = torch.nn.functional.interpolate(
        logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    print(logits)
    print(logits.shape)
    logits = logits.argmax(1)

    logits = logits.squeeze(0).cpu().numpy()
    logits = render_segmentation(logits)
    logits.show()

    #print(logits.shape)


if __name__ == "__main__":
    main()
