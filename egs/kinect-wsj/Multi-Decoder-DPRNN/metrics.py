import torch
import torch.nn as nn
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch.nn.modules.loss import _Loss
from scipy.optimize import linear_sum_assignment


class PairwiseNegSDR_Loss(_Loss):
    """
    Same as asteroid.losses.PairwiseNegSDR, but supports speaker number mismatch
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(PairwiseNegSDR_Loss, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets):
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class Penalized_PIT_Wrapper(nn.Module):
    """
    Implementation of P-Si-SNR, as purposed in [1]
    References:
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(self, loss_func, penalty=30, perm_reduce=None):
        super().__init__()
        assert penalty > 0, "penalty term should be positive"
        self.neg_penalty = -penalty
        self.perm_reduce = perm_reduce
        self.loss_func = loss_func

    def forward(self, est_targets, targets, **kwargs):
        """
        est_targets: torch.Tensor, $(est_nsrc, ...)$
        targets: torch.Tensor, $(gt_nsrc, ...)$
        """
        est_nsrc, T = est_targets.size()
        gt_nsrc = est_targets.size(0)
        pw_losses = self.loss_func(est_targets.unsqueeze(0), targets.unsqueeze(0)).squeeze(0)
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pw_losses.transpose(-1, -2)
        # Loop over batch + row indices are always ordered for square matrices.
        row, col = [torch.Tensor(x).long() for x in linear_sum_assignment(pwl.detach().cpu())]
        avg_neg_sdr = pwl[row, col].mean()
        p_si_snr = (
            -avg_neg_sdr * min(est_nsrc, gt_nsrc) + self.neg_penalty * abs(est_nsrc - gt_nsrc)
        ) / max(est_nsrc, gt_nsrc)
        return p_si_snr


# alias
pairwise_neg_sisdr_loss = PairwiseNegSDR_Loss("sisdr")
