import pytest
import torch

from asteroid.losses import MixITLossWrapper
from asteroid.losses import pairwise_neg_sisdr, multisrc_neg_sisdr


def good_batch_loss_func(y_pred, y_true):
    batch, *_ = y_true.shape
    return torch.randn(batch)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("time", [16000])
def test_mixitwrapper_as_pit_wrapper(batch_size, n_src, time):
    targets = torch.randn(batch_size, n_src, time)
    est_targets = torch.randn(batch_size, n_src, time)

    # mix_it base case: targets == mixtures / With and without return estimates
    loss = MixITLossWrapper(good_batch_loss_func, generalized=False)
    loss(est_targets, targets)
    loss_value, reordered_est = loss(est_targets, targets, return_est=True)
    assert reordered_est.shape == est_targets.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("n_mix", [2, 3])
@pytest.mark.parametrize("time", [2000])
def test_mixit_wrapper(batch_size, factor, n_mix, time):
    mixtures = torch.randn(batch_size, n_mix, time)
    n_src = n_mix * factor
    est_targets = torch.randn(batch_size, n_src, time)

    # mix_it / With and without return estimates
    loss = MixITLossWrapper(good_batch_loss_func, generalized=False)
    loss(est_targets, mixtures)
    loss_value, reordered_mix = loss(est_targets, mixtures, return_est=True)
    assert reordered_mix.shape == mixtures.shape


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("n_src", [2, 3, 4, 5])
@pytest.mark.parametrize("n_mix", [2])
@pytest.mark.parametrize("time", [16000])
def test_mixit_gen_wrapper(batch_size, n_src, n_mix, time):
    mixtures = torch.randn(batch_size, n_mix, time)
    est_targets = torch.randn(batch_size, n_src, time)

    # mix_it_gen / With and without return estimates. Works only with two mixtures
    loss = MixITLossWrapper(good_batch_loss_func)
    loss(est_targets, mixtures)
    loss_value, reordered_est = loss(est_targets, mixtures, return_est=True)
    assert reordered_est.shape == mixtures.shape


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("n_mix", [2])
@pytest.mark.parametrize("time", [16000])
@pytest.mark.parametrize("generalized", [True, False])
def test_mixitwrapper_checks_loss_shape(batch_size, factor, n_mix, time, generalized):
    mixtures = torch.randn(batch_size, n_mix, time)
    n_src = n_mix * factor
    est_targets = torch.randn(batch_size, n_src, time)

    # correct usage
    loss = MixITLossWrapper(multisrc_neg_sisdr, generalized=generalized)
    loss(est_targets, mixtures)

    # incorrect usage
    loss = MixITLossWrapper(pairwise_neg_sisdr, generalized=generalized)
    with pytest.raises(ValueError):
        loss(est_targets, mixtures)
