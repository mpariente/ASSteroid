import torch
import warnings
from typing import Optional

from .. import separate
from ..masknn import activations
from ..utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from ..utils.hub_utils import cached_download, SR_HASHTABLE
from ..utils.deprecation_utils import is_overridden, mark_deprecated


@script_if_tracing
def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


class BaseModel(torch.nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures, and separation interface to `separate`.
    Need to overwrite the `forward` and `get_model_args` methods.

    Models inheriting from `BaseModel` can be used by :mod:`asteroid.separate`
    and by the `asteroid-infer` CLI. For models whose `forward` doesn't go from
    waveform to waveform tensors, overwrite `forward_wav` to return
    waveform tensors.

    Args:
        sample_rate (float): Operating sample rate of the model.
        in_channels: Number of input channels in the signal.
            If None, no checks will be performed.
    """

    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn(
            "Other sub-components of the model might have a `sample_rate` "
            "attribute, be sure to modify them for consistency.",
            UserWarning,
        )
        self.__sample_rate = new_sample_rate

    def separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.separate`."""
        return separate.separate(self, *args, **kwargs)

    def torch_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.torch_separate`."""
        return separate.torch_separate(self, *args, **kwargs)

    def numpy_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.numpy_separate`."""
        return separate.numpy_separate(self, *args, **kwargs)

    def file_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.file_separate`."""
        return separate.file_separate(self, *args, **kwargs)

    def forward_wav(self, wav, *args, **kwargs):
        """Separation method for waveforms.

        In case the network's `forward` doesn't have waveforms as input/output,
        overwrite this method to separate from waveform to waveform.
        Should return a single torch.Tensor, the separated waveforms.

        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        from . import get  # Avoid circular imports

        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        if "sample_rate" not in conf["model_args"] and isinstance(
            pretrained_model_conf_or_path, str
        ):
            conf["model_args"]["sample_rate"] = SR_HASHTABLE.get(
                pretrained_model_conf_or_path, None
            )
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        else:
            model = model_class(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        from .. import __version__ as asteroid_version  # Avoid circular imports

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version=asteroid_version,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()

    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def get_masker_working_size(self, n_samples: int) -> int:
        """Get the masker's internal working size for an input of `n_samples`.

        Generally, if `fix_length_mode = "pad"`, the internal working size is
        `>= n_samples`, otherwise it is `<= n_samples`.
        """
        x = torch.zeros(1, 1, n_samples)
        tf_rep = self.forward_encoder(x)
        tf_rep = self.masker.fix_input_dims(tf_rep)
        masked = self.apply_masks(tf_rep, 1)
        decoded = self.forward_decoder(masked)
        return decoded.shape[-1]


    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args


@script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


# Backwards compatibility
BaseTasNet = BaseEncoderMaskerDecoder
