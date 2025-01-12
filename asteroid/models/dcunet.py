import torch
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import from_torch_complex, to_torch_complex
from ..masknn.convolutional import DCUMaskNet
from .base_models import BaseEncoderMaskerDecoder


class BaseDCUNet(BaseEncoderMaskerDecoder):
    """Base class for ``DCUNet`` and ``DCCRNet`` classes.

    Args:
        architecture (str): The architecture to use. Overriden by subclasses.
        stft_n_filters (int) Number of filters for the STFT.
        stft_kernel_size (int): STFT frame length to use.
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.
        masknet_kwargs (optional): Passed to the masknet constructor.
    """

    masknet_class = NotImplemented

    def __init__(
        self,
        architecture,
        stft_n_filters=1024,
        stft_kernel_size=1024,
        stft_stride=256,
        sample_rate=16000.0,
        **masknet_kwargs,
    ):
        self.architecture = architecture
        self.stft_n_filters = stft_n_filters
        self.stft_kernel_size = stft_kernel_size
        self.stft_stride = stft_stride
        self.masknet_kwargs = masknet_kwargs

        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=stft_n_filters,
            kernel_size=stft_kernel_size,
            stride=stft_stride,
            sample_rate=sample_rate,
        )
        masker = self.masknet_class.default_architecture(architecture, **masknet_kwargs)
        super().__init__(encoder, masker, decoder)

    def forward_encoder(self, wav):
        tf_rep = self.encoder(wav)
        return to_torch_complex(tf_rep)

    def apply_masks(self, tf_rep, est_masks):
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return from_torch_complex(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "architecture": self.architecture,
            "stft_n_filters": self.stft_n_filters,
            "stft_kernel_size": self.stft_kernel_size,
            "stft_stride": self.stft_stride,
            "sample_rate": self.sample_rate,
            **self.masknet_kwargs,
        }
        return model_args


class DCUNet(BaseDCUNet):
    """DCUNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, any of
            "DCUNet-10", "DCUNet-16", "DCUNet-20", "Large-DCUNet-20".
        stft_n_filters (int) Number of filters for the STFT.
        stft_kernel_size (int): STFT frame length to use.
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.
        masknet_kwargs (optional): Passed to :class:`DCUMaskNet`

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    masknet_class = DCUMaskNet
