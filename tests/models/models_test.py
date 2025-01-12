import torch
import pytest
from torch.testing import assert_close
import numpy as np
import soundfile as sf
import asteroid
from asteroid import models
from asteroid_filterbanks import make_enc_dec
from asteroid.dsp import LambdaOverlapAdd
from asteroid.models.fasnet import FasNetTAC
from asteroid.separate import separate
from asteroid.models import (
    ConvTasNet,
    DCCRNet,
    DCUNet,
    DeMask,
    DPRNNTasNet,
    DPTNet,
    LSTMTasNet,
    SuDORMRFImprovedNet,
    SuDORMRFNet,
)
from asteroid.models.base_models import BaseModel
from asteroid.utils.deprecation_utils import VisibleDeprecationWarning

HF_EXAMPLE_MODEL_IDENTIFER = "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"
# An actual model hosted on huggingface.co


def test_set_sample_rate_raises_warning():
    model = BaseModel(sample_rate=8000.0)
    with pytest.warns(UserWarning):
        model.sample_rate = 16000.0


def test_no_sample_rate_raises():
    with pytest.raises(TypeError):
        BaseModel()


def test_multichannel_model_loading():
    class MCModel(BaseModel):
        def __init__(self, sample_rate=8000.0, in_channels=2):
            super().__init__(sample_rate=sample_rate, in_channels=in_channels)

        def forward(self, x, **kwargs):
            return x

        def get_model_args(self):
            return {"sample_rate": self.sample_rate, "in_channels": self.in_channels}

    model = MCModel()
    model_conf = model.serialize()

    new_model = MCModel.from_pretrained(model_conf)
    assert model.in_channels == new_model.in_channels


def test_convtasnet_sep():
    nnet = ConvTasNet(
        n_src=2,
        n_repeats=2,
        n_blocks=3,
        bn_chan=16,
        hid_chan=4,
        skip_chan=8,
        n_filters=32,
    )
    # Test torch input
    wav = torch.rand(1, 800)
    out = nnet.separate(wav)
    assert isinstance(out, torch.Tensor)
    # Test numpy input
    wav = np.random.randn(1, 800).astype("float32")
    out = nnet.separate(wav)
    assert isinstance(out, np.ndarray)
    # Test str input
    sf.write("tmp.wav", wav[0], 8000)
    nnet.separate("tmp.wav")
    # Warning when overwriting
    with pytest.warns(UserWarning):
        nnet.separate("tmp.wav")

    # Test with bad samplerate
    sf.write("tmp.wav", wav[0], 16000)
    # Raises
    with pytest.raises(RuntimeError):
        nnet.separate("tmp.wav", force_overwrite=True)
    # Resamples
    nnet.separate("tmp.wav", force_overwrite=True, resample=True)


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
def test_save_and_load_convtasnet(fb, sample_rate):
    _default_test_model(
        ConvTasNet(
            n_src=2,
            n_repeats=2,
            n_blocks=2,
            bn_chan=16,
            hid_chan=4,
            skip_chan=8,
            n_filters=32,
            fb_name=fb,
            sample_rate=sample_rate,
        )
    )


@pytest.mark.parametrize("use_mulcat", [True, False])
def test_dprnntasnet_sep(use_mulcat):
    nnet = DPRNNTasNet(
        n_src=2,
        n_repeats=2,
        bn_chan=16,
        hid_size=4,
        chunk_size=20,
        n_filters=32,
        use_mulcat=use_mulcat,
    )
    # Test torch input
    wav = torch.rand(1, 800)
    out = nnet.separate(wav)
    assert isinstance(out, torch.Tensor)
    # Test numpy input
    wav = np.random.randn(1, 800).astype("float32")
    out = nnet.separate(wav)
    assert isinstance(out, np.ndarray)


def test_dprnntasnet_sep_from_hf():
    model = DPRNNTasNet.from_pretrained(HF_EXAMPLE_MODEL_IDENTIFER)
    assert isinstance(model, DPRNNTasNet)


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
@pytest.mark.parametrize("use_mulcat", [True, False])
def test_save_and_load_dprnn(fb, sample_rate, use_mulcat):
    _default_test_model(
        DPRNNTasNet(
            n_src=2,
            n_repeats=2,
            bn_chan=16,
            hid_size=4,
            chunk_size=20,
            n_filters=32,
            fb_name=fb,
            sample_rate=sample_rate,
            use_mulcat=use_mulcat,
        )
    )


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
def test_save_and_load_tasnet(fb, sample_rate):
    _default_test_model(
        LSTMTasNet(
            n_src=2,
            hid_size=4,
            n_layers=1,
            n_filters=32,
            dropout=0.0,
            fb_name=fb,
            sample_rate=sample_rate,
        )
    )


@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
def test_sudormrf(sample_rate):
    _default_test_model(
        SuDORMRFNet(
            2,
            bn_chan=10,
            num_blocks=4,
            upsampling_depth=2,
            kernel_size=21,
            n_filters=12,
            sample_rate=sample_rate,
        )
    )


@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
def test_sudormrf_imp(sample_rate):
    _default_test_model(
        SuDORMRFImprovedNet(
            2,
            bn_chan=10,
            num_blocks=4,
            upsampling_depth=2,
            kernel_size=21,
            n_filters=12,
            sample_rate=sample_rate,
        )
    )


@pytest.mark.filterwarnings("ignore: DPTransformer input dim")
@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
@pytest.mark.parametrize("sample_rate", [8000.0, 16000.0])
def test_dptnet(fb, sample_rate):
    _default_test_model(DPTNet(2, ff_hid=10, chunk_size=4, n_repeats=2, sample_rate=sample_rate))


@pytest.mark.parametrize("use_tac", [True, False])
def test_fasnet(use_tac):
    _default_test_model(
        FasNetTAC(n_src=2, feature_dim=8, hidden_dim=10, n_layers=2, use_tac=use_tac),
        test_input=torch.randn(3, 2, 8372),
    )


def test_dcunet():
    n_fft = 1024
    _, istft = make_enc_dec(
        "stft", n_filters=n_fft, kernel_size=1024, stride=256, sample_rate=16000
    )
    input_samples = istft(torch.zeros((n_fft + 2, 17))).shape[0]
    _default_test_model(DCUNet("DCUNet-10"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-16"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-20"), input_samples=input_samples)
    _default_test_model(DCUNet("Large-DCUNet-20"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-10", n_src=2), input_samples=input_samples)

    # DCUMaskNet should fail with wrong freqency dimensions
    DCUNet("mini").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    with pytest.raises(TypeError):
        DCUNet("mini").masker(torch.zeros((1, 42, 17), dtype=torch.complex64))

    # DCUMaskNet should fail with wrong time dimensions if fix_length_mode is not used
    DCUNet("mini", fix_length_mode="pad").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    DCUNet("mini", fix_length_mode="trim").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    with pytest.raises(TypeError):
        DCUNet("mini").masker(torch.zeros((1, 9, 16), dtype=torch.complex64))


def test_masker_working_size():
    dcu_mini_trim = DCUNet("mini", fix_length_mode="trim")
    dcu_mini_pad = DCUNet("mini", fix_length_mode="pad")
    dccrn_mini = DCCRNet("mini")
    inp_size = 3 * 16000
    assert dcu_mini_trim.get_masker_working_size(inp_size) == 47616
    assert dcu_mini_pad.get_masker_working_size(inp_size) == 49664
    assert dccrn_mini.get_masker_working_size(inp_size) == 47872


def test_dccrnet():
    n_fft = 512
    _, istft = make_enc_dec("stft", n_filters=n_fft, kernel_size=400, stride=100, sample_rate=16000)
    input_samples = istft(torch.zeros((n_fft + 2, 16))).shape[0]
    _default_test_model(DCCRNet("DCCRN-CL"), input_samples=input_samples)
    _default_test_model(DCCRNet("DCCRN-CL", n_src=2), input_samples=input_samples)

    # DCCRMaskNet should fail with wrong input dimensions
    DCCRNet("mini").masker(torch.zeros((1, 256, 3), dtype=torch.complex64))
    with pytest.raises(TypeError):
        DCCRNet("mini").masker(torch.zeros((1, 42, 3), dtype=torch.complex64))


def _default_test_model(model, input_samples=801, test_input=None):
    if test_input is None:
        test_input = torch.randn(1, input_samples)

    model_conf = model.serialize()
    reconstructed_model = model.__class__.from_pretrained(model_conf)
    assert_close(model(test_input), reconstructed_model(test_input))

    # Load with and without SR
    sr = model_conf["model_args"].pop("sample_rate")
    reconstructed_model_nosr = model.__class__.from_pretrained(model_conf)
    reconstructed_model = model.__class__.from_pretrained(model_conf, sample_rate=sr)

    assert reconstructed_model.sample_rate == model.sample_rate


@pytest.mark.parametrize(
    "model", [LSTMTasNet, ConvTasNet, DPRNNTasNet, DPTNet, SuDORMRFImprovedNet, SuDORMRFNet]
)
def test_get(model):
    retrieved = models.get(model.__name__)
    assert retrieved == model


@pytest.mark.parametrize("wrong", ["wrong_string", 12, object()])
def test_get_errors(wrong):
    with pytest.raises(ValueError):
        # Should raise for anything not a Optimizer instance + unknown string
        models.get(wrong)


def test_register():
    class Custom(BaseModel):
        def __init__(self):
            super().__init__()

    models.register_model(Custom)
    cls = models.get("Custom")
    assert cls == Custom

    with pytest.raises(ValueError):
        models.register_model(models.DPRNNTasNet)


def test_show():
    asteroid.show_available_models()


def test_available_models():
    all_models = asteroid.available_models()


@pytest.mark.parametrize("fb", ["free", "stft", "analytic_free", "param_sinc"])
def test_demask(fb):
    model = DeMask(fb_name=fb)
    test_input = torch.randn(1, 801)

    model_conf = model.serialize()
    reconstructed_model = DeMask.from_pretrained(model_conf)
    assert_close(model(test_input), reconstructed_model(test_input))


def test_separate():
    nnet = ConvTasNet(
        n_src=2,
        n_repeats=2,
        n_blocks=3,
        bn_chan=16,
        hid_chan=4,
        skip_chan=8,
        n_filters=32,
    )
    # Test torch input
    wav = torch.rand(1, 1, 8000)
    model = LambdaOverlapAdd(nnet, None, window_size=1000)
    out = separate(model, wav)
