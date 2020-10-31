from ..utils.deprecation_utils import DeprecationMixin
from torch.nn.modules.loss import _Loss


class PairwiseMSE(_Loss):
    """Measure pairwise mean square error on a batch.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of training targets

    Returns:
        :class:`torch.Tensor`: with shape [batch, nsrc, nsrc]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseMSE(), pit_from='pairwise')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, *], got {targets.size()} and {est_targets.size()} instead"
            )
        targets = targets.unsqueeze(1)
        est_targets = est_targets.unsqueeze(2)
        pw_loss = (targets - est_targets) ** 2
        # Need to return [batch, nsrc, nsrc]
        mean_over = list(range(3, pw_loss.ndim))
        return pw_loss.mean(dim=mean_over)


class SingleSrcMSE(_Loss):
    """Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, *].
            The batch of training targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # singlesrc_mse / multisrc_mse support both 'pw_pt' and 'perm_avg'.
        >>> loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        loss = (targets - est_targets) ** 2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)


# aliases
MultiSrcMSE = SingleSrcMSE
pairwise_mse = PairwiseMSE()
singlesrc_mse = SingleSrcMSE()
multisrc_mse = MultiSrcMSE()


# Legacy
class NoSrcMSE(SingleSrcMSE, DeprecationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


NonPitMSE = NoSrcMSE
nosrc_mse = singlesrc_mse
nonpit_mse = multisrc_mse
