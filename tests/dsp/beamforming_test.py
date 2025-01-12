import torch
import pytest
from asteroid_filterbanks import make_enc_dec, transforms as tr

from asteroid.dsp.beamforming import (
    Beamformer,
    SCM,
    RTFMVDRBeamformer,
    SoudenMVDRBeamformer,
    SDWMWFBeamformer,
    GEVBeamformer,
    GEVDBeamformer,
    stable_cholesky,
)


torch_has_complex_support = tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 8)

_stft, _istft = make_enc_dec("stft", kernel_size=512, n_filters=512, stride=128)
stft = lambda x: tr.to_torch_complex(_stft(x))
istft = lambda x: _istft(tr.from_torch_complex(x))


@pytest.mark.skipif(not torch_has_complex_support, "No complex support ")
def _default_beamformer_test(beamformer: Beamformer, batch_size=2, n_mics=4, **forward_kwargs):
    scm = SCM()

    speech = torch.randn(batch_size, n_mics, 16000 * 2)
    noise = torch.randn(batch_size, n_mics, 16000 * 2)
    mix = speech + noise
    # GeV Beamforming
    mix_stft = stft(mix)
    speech_stft = stft(speech)
    noise_stft = stft(noise)
    sigma_ss = scm(speech_stft)
    sigma_nn = scm(noise_stft)

    Ys_gev = beamformer.forward(
        mix=mix_stft, target_scm=sigma_ss, noise_scm=sigma_nn, **forward_kwargs
    )
    ys_gev = istft(Ys_gev)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gev(n_mics, batch_size):
    _default_beamformer_test(GEVBeamformer(), n_mics=n_mics, batch_size=batch_size)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mvdr(n_mics, batch_size):
    _default_beamformer_test(RTFMVDRBeamformer(), n_mics=n_mics, batch_size=batch_size)
    _default_beamformer_test(SoudenMVDRBeamformer(), n_mics=n_mics, batch_size=batch_size)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("mu", [1.0, 2.0, 0])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mwf(n_mics, mu, batch_size):
    _default_beamformer_test(SDWMWFBeamformer(mu=mu), n_mics=n_mics, batch_size=batch_size)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mwf_indices(n_mics, batch_size):
    _default_beamformer_test(SDWMWFBeamformer(), n_mics=n_mics, batch_size=batch_size, ref_mic=0)
    _default_beamformer_test(SDWMWFBeamformer(), n_mics=n_mics, batch_size=batch_size, ref_mic=None)
    _default_beamformer_test(
        SDWMWFBeamformer(),
        n_mics=n_mics,
        batch_size=batch_size,
        ref_mic=torch.randint(0, n_mics, size=(batch_size,)),
    )
    _default_beamformer_test(
        SDWMWFBeamformer(),
        n_mics=n_mics,
        batch_size=batch_size,
        ref_mic=torch.randn(batch_size, 1, n_mics, 1, dtype=torch.complex64),
    )


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("mu", [2.0, 1.0, 0.5])
@pytest.mark.parametrize("rank", [1, 2, None])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gevd(n_mics, mu, rank, batch_size):
    _default_beamformer_test(GEVDBeamformer(mu=mu, rank=rank), n_mics=n_mics, batch_size=batch_size)


def test_stable_cholesky():
    a = torch.randn(3, 3)
    a = torch.mm(a, a.t())  # make symmetric positive-definite
    stable_cholesky(a)
