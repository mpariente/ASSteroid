from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import warnings

from .. import complex_nn
from . import norms, activations
from .base import BaseDCUMaskNet
from .norms import GlobLN
from ..utils import has_arg
from ._dcunet_architectures import DCUNET_ARCHITECTURES
from ._local import _DilatedConvNorm, _NormAct, _ConvNormAct, _ConvNorm
from ..utils.torch_utils import script_if_tracing, pad_x_to_y


class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm.
            -  ``'cLN'``: channelwise Layernorm.
            -  ``'cgLN'``: cumulative global Layernorm.
            -  Any norm supported by :func:`~.norms.get`
        causal (bool, optional) : Whether or not the convolutions are causal


    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        causal (bool, optional) : Whether or not the convolutions are causal.

    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2**x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2**x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "causal": self.causal,
        }
        return config


class TDConvNetpp(nn.Module):
    """Improved Temporal Convolutional network used in [1] (TDCN++)

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.

    References
        [1] : Kavalerov, Ilya et al. “Universal Sound Separation.” in WASPAA 2019

    .. note::
        The differences wrt to ConvTasnet's TCN are:

        1. Channel wise layer norm instead of global
        2. Longer-range skip-residual connections from earlier repeat inputs
           to later repeat inputs after passing them through dense layer.
        3. Learnable scaling parameter after each dense layer. The scaling
           parameter for the second dense  layer  in  each  convolutional
           block (which  is  applied  rightbefore the residual connection) is
           initialized to an exponentially decaying scalar equal to 0.9**L,
           where L is the layer or block index.

    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="fgLN",
        mask_act="relu",
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2**x // 2
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                    )
                )
        # Dense connection in TDCNpp
        self.dense_skip = nn.ModuleList()
        for r in range(n_repeats - 1):
            self.dense_skip.append(nn.Conv1d(bn_chan, bn_chan, 1))

        scaling_param = torch.Tensor([0.9**l for l in range(1, n_blocks)])
        scaling_param = scaling_param.unsqueeze(0).expand(n_repeats, n_blocks - 1).clone()
        self.scaling_param = nn.Parameter(scaling_param, requires_grad=True)

        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        out_size = skip_chan if skip_chan else bn_chan
        self.consistency = nn.Linear(out_size, n_src)

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output_copy = output

        skip_connection = 0.0
        for r in range(self.n_repeats):
            # Long range skip connection TDCNpp
            if r != 0:
                # Transform the input to repeat r-1 and add to new repeat inp
                output = self.dense_skip[r - 1](output_copy) + output
                # Copy this for later.
                output_copy = output
            for x in range(self.n_blocks):
                # Common to w. skip and w.o skip architectures
                i = r * self.n_blocks + x
                tcn_out = self.TCN[i](output)
                if self.skip_chan:
                    residual, skip = tcn_out
                    skip_connection = skip_connection + skip
                else:
                    residual, _ = tcn_out
                # Initialized exp decay scale factor TDCNpp for residual connections
                scale = self.scaling_param[r, x - 1] if x > 0 else 1.0
                residual = residual * scale
                output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)

        weights = self.consistency(mask_inp.mean(-1))
        weights = torch.nn.functional.softmax(weights, -1)

        return est_mask, weights

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
        }
        return config


class DCUNetComplexEncoderBlock(nn.Module):
    """Encoder block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (Tuple[int, int]): Convolution kernel size.
        stride (Tuple[int, int]): Convolution stride.
        padding (Tuple[int, int]): Convolution padding.
        norm_type (str, optional): Type of normalization to use.
            See :mod:`~asteroid.masknn.norms` for valid values.
        activation (str, optional): Type of activation to use.
            See :mod:`~asteroid.masknn.activations` for valid values.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        norm_type="bN",
        activation="leaky_relu",
    ):
        super().__init__()

        self.conv = complex_nn.ComplexConv2d(
            in_chan, out_chan, kernel_size, stride, padding, bias=norm_type is None
        )

        self.norm = norms.get_complex(norm_type)(out_chan)

        activation_class = activations.get_complex(activation)
        self.activation = activation_class()

    def forward(self, x: complex_nn.ComplexTensor):
        return self.activation(self.norm(self.conv(x)))


class DCUNetComplexDecoderBlock(nn.Module):
    """Decoder block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        out_chan (int): Number of output channels.
        kernel_size (Tuple[int, int]): Convolution kernel size.
        stride (Tuple[int, int]): Convolution stride.
        padding (Tuple[int, int]): Convolution padding.
        norm_type (str, optional): Type of normalization to use.
            See :mod:`~asteroid.masknn.norms` for valid values.
        activation (str, optional): Type of activation to use.
            See :mod:`~asteroid.masknn.activations` for valid values.

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        output_padding=(0, 0),
        norm_type="bN",
        activation="leaky_relu",
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.deconv = complex_nn.ComplexConvTranspose2d(
            in_chan, out_chan, kernel_size, stride, padding, output_padding, bias=norm_type is None
        )

        self.norm = norms.get_complex(norm_type)(out_chan)

        activation_class = activations.get_complex(activation)
        self.activation = activation_class()

    def forward(self, x: complex_nn.ComplexTensor):
        return self.activation(self.norm(self.deconv(x)))


class DCUMaskNet(BaseDCUMaskNet):
    r"""Masking part of DCUNet, as proposed in [1].

    Valid `architecture` values for the ``default_architecture`` classmethod are:
    "Large-DCUNet-20", "DCUNet-20", "DCUNet-16", "DCUNet-10" and "mini".

    Valid `fix_length_mode` values are [None, "pad", "trim"].

    Input shape is expected to be $(batch, nfreqs, time)$, with $nfreqs - 1$ divisible
    by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides of the encoders,
    and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$ where $t_N$ are the time
    strides of the encoders. If `fix_length_mode` is not `None`, the time dimension
    may is automatically padded or trimmed to a valid size before running it through
    the network.

    .. note::
       If using `fix_length_mode="trim"`, the network's output will be all-zero at the
       trimmed time-steps. You might want to ignore those time-steps in your loss function.

       The time-domain length of the network's internal working size (the trimmed size)
       can be retrieved using :meth:`~asteroid.models.BaseDCUNet.get_masker_working_size`::

          >>> dcu16 = DCUNet("DCUNet-16", fix_length_mode="trim")
          >>> dcu16.get_masker_working_size(3 * 16000)
          45568

    References
        [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
        Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    _architectures = DCUNET_ARCHITECTURES

    def __init__(self, encoders, decoders, fix_length_mode=None, **kwargs):
        self.fix_length_mode = fix_length_mode
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _ in encoders], axis=0
        )

        # Avoid circual import
        from .convolutional import DCUNetComplexDecoderBlock, DCUNetComplexEncoderBlock

        super().__init__(
            encoders=[DCUNetComplexEncoderBlock(*args) for args in encoders],
            decoders=[DCUNetComplexDecoderBlock(*args) for args in decoders[:-1]],
            output_layer=complex_nn.ComplexConvTranspose2d(*decoders[-1]),
            **kwargs,
        )

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    def fix_output_dims(self, out, x):
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


@script_if_tracing
def _fix_dcu_input_dims(fix_length_mode: Optional[str], x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[1] - 1) % freq_prod:
        raise TypeError(
            f"Input shape must be [batch, freq + 1, time + 1] with freq divisible by "
            f"{freq_prod}, got {x.shape} instead"
        )
    time_remainder = (x.shape[2] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(
                f"Input shape must be [batch, freq + 1, time + 1] with time divisible by "
                f"{time_prod}, got {x.shape} instead. Set the 'fix_length_mode' argument "
                f"in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically."
            )
        elif fix_length_mode == "pad":
            pad_shape = [0, time_prod - time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        elif fix_length_mode == "trim":
            pad_shape = [0, -time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


@script_if_tracing
def _fix_dcu_output_dims(fix_length_mode: Optional[str], out, x):
    """Fix shape of `out` to the original shape of `x`."""
    return pad_x_to_y(out, x)


class SuDORMRF(nn.Module):
    """SuDORMRF mask network, as described in [1].

    Args:
        in_chan (int): Number of input channels. Also number of output channels.
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks.
        upsampling_depth (int): Depth of upsampling.
        mask_act (str): Name of output activation.

    References
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
        Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        in_chan,
        n_src,
        bn_chan=128,
        num_blocks=16,
        upsampling_depth=4,
        mask_act="softmax",
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.bn_chan = bn_chan
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.mask_act = mask_act

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.GroupNorm(1, in_chan, eps=1e-08)
        self.l1 = nn.Conv1d(in_chan, bn_chan, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UBlock(
                    out_chan=bn_chan,
                    in_chan=in_chan,
                    upsampling_depth=upsampling_depth,
                )
                for _ in range(num_blocks)
            ]
        )

        if bn_chan != in_chan:
            self.reshape_before_masks = nn.Conv1d(bn_chan, in_chan, kernel_size=1)

        # Masks layer
        self.m = nn.Conv2d(
            1,
            n_src,
            kernel_size=(in_chan + 1, 1),
            padding=(in_chan - in_chan // 2, 0),
        )

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.bn_chan != self.in_chan:
            x = self.reshape_before_masks(x)

        # Get output + activation
        x = self.m(x.unsqueeze(1))
        x = self.output_act(x)
        return x

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "n_src": self.n_src,
            "bn_chan": self.bn_chan,
            "num_blocks": self.num_blocks,
            "upsampling_depth": self.upsampling_depth,
            "mask_act": self.mask_act,
        }
        return config


class SuDORMRFImproved(nn.Module):
    """Improved SuDORMRF mask network, as described in [1].

    Args:
        in_chan (int): Number of input channels. Also number of output channels.
        n_src (int): Number of sources in the input mixtures.
        bn_chan (int, optional): Number of bins in the bottleneck layer and the UNet blocks.
        num_blocks (int): Number of of UBlocks
        upsampling_depth (int): Depth of upsampling
        mask_act (str): Name of output activation.


    References
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
        Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        in_chan,
        n_src,
        bn_chan=128,
        num_blocks=16,
        upsampling_depth=4,
        mask_act="relu",
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.bn_chan = bn_chan
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.mask_act = mask_act

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(in_chan)
        self.bottleneck = nn.Conv1d(in_chan, bn_chan, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UConvBlock(
                    out_chan=bn_chan,
                    in_chan=in_chan,
                    upsampling_depth=upsampling_depth,
                )
                for _ in range(num_blocks)
            ]
        )

        mask_conv = nn.Conv1d(bn_chan, n_src * in_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, x):
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.n_src, self.in_chan, -1)
        x = self.output_act(x)
        return x

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "n_src": self.n_src,
            "bn_chan": self.bn_chan,
            "num_blocks": self.num_blocks,
            "upsampling_depth": self.upsampling_depth,
            "mask_act": self.mask_act,
        }
        return config


class _BaseUBlock(nn.Module):
    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4, use_globln=False):
        super().__init__()
        self.proj_1x1 = _ConvNormAct(
            out_chan, in_chan, 1, stride=1, groups=1, use_globln=use_globln
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            _DilatedConvNorm(
                in_chan,
                in_chan,
                kSize=5,
                stride=1,
                groups=in_chan,
                d=1,
                use_globln=use_globln,
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                _DilatedConvNorm(
                    in_chan,
                    in_chan,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_chan,
                    d=1,
                    use_globln=use_globln,
                )
            )
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(
                scale_factor=2,
                # align_corners=True,
                # mode='bicubic'
            )


class UBlock(_BaseUBlock):
    """Upsampling block.

    Based on the following principle: ``REDUCE ---> SPLIT ---> TRANSFORM --> MERGE``
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=False)
        self.conv_1x1_exp = _ConvNorm(in_chan, out_chan, 1, 1, groups=1)
        self.final_norm = _NormAct(in_chan)
        self.module_act = _NormAct(out_chan)

    def forward(self, x):
        """
        Args:
            x: input feature map

        Returns:
            transformed feature map
        """

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k[..., : output[-1].shape[-1]]

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)


class UConvBlock(_BaseUBlock):
    """Block which performs successive downsampling and upsampling
    in order to be able to analyze the input features in multiple resolutions.
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=True)
        self.final_norm = _NormAct(in_chan, use_globln=True)
        self.res_conv = nn.Conv1d(in_chan, out_chan, 1)

    def forward(self, x):
        """
        Args
            x: input feature map

        Returns:
            transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k[..., : output[-1].shape[-1]]

        expanded = self.final_norm(output[-1])

        return self.res_conv(expanded) + residual
