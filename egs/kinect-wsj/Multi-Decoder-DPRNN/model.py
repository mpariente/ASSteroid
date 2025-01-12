"""
Author: Joseph(Junzhe) Zhu, 2021/5. Email: josefzhu@stanford.edu / junzhe.joseph.zhu@gmail.com
For the original code for the paper[1], please refer to https://github.com/JunzheJosephZhu/MultiDecoder-DPRNN
Demo Page: https://junzhejosephzhu.github.io/Multi-Decoder-DPRNN/
Multi-Decoder DPRNN is a method for source separation when the number of speakers is unknown. 
Our contribution is using multiple output heads, with each head modelling a distinct number of source outputs. 
In addition, we design a selector network which determines which output head to use, i.e. estimates the number of sources. 
The "DPRNN" part of the architecture is orthogonal to our contribution, and can be replaced with any other separator, e.g. Conv/LSTM-TasNet. 
References:
    [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
        Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
"""
import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import fold, unfold

from asteroid import torch_utils
from asteroid.models import BaseModel
from asteroid_filterbanks import make_enc_dec
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import activations, norms
from asteroid.masknn.recurrent import DPRNNBlock
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.utils.generic_utils import has_arg
from asteroid.utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


def make_model_and_optimizer(conf, sample_rate):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    model = MultiDecoderDPRNN(**conf["masknet"], **conf["filterbank"], sample_rate=sample_rate)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


class MultiDecoderDPRNN(BaseModel):
    """Multi-Decoder Dual-Path RNN as proposed in [1].

    Args:
        n_srcs (list of int): range of possible number of sources
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        kernel_size (int): Length of the filters.
        n_filters (int): Number of filters / Input dimension of the masker net.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.

    References
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(
        self,
        n_srcs,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        use_mulcat=False,
        sample_rate=8000,
    ):
        super().__init__(sample_rate=sample_rate)
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.encoder, _ = make_enc_dec(
            "free",
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
        )
        # Update in_chan
        self.masker = DPRNN_MultiStage(
            in_chan=n_filters,
            bn_chan=bn_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_mulcat=use_mulcat,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder_select = Decoder_Select(
            kernel_size=kernel_size,
            stride=stride,
            in_chan=n_filters,
            n_srcs=n_srcs,
            bn_chan=bn_chan,
            chunk_size=chunk_size,
            hop_size=hop_size,
            mask_act=mask_act,
        )

        """
        Args:
            wav: 2D or 3D Tensor, Tensor of shape $(batch, T)$
            ground_truth: oracle number of speakers, None or list of $(batch)$ ints 
        Return:
            reconstructed: torch.Tensor, $(batch, num_stages, max_spks, T)$
                where max_spks is the maximum possible number of speakers.
                if training, num_stages=n_repeats; otherwise num_stages=0
            Speaker dimension is zero-padded for examples with num_spks < max_spks
        """

    def forward(self, wav, ground_truth=None):
        shape = jitable_shape(wav)
        # [batch, 1, T]
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks_list = self.masker(tf_rep)
        decoded, selector_output = self.decoder_select(
            est_masks_list, tf_rep, ground_truth=ground_truth
        )
        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), _shape_reconstructed(
            selector_output, shape
        )

    def forward_wav(self, wav, slice_size=32000, *args, **kwargs):
        """Separation method for waveforms.
        Unfolds a full audio into slices, estimate
        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        Return:
            output_cat (torch.Tensor): concatenated output tensor.
                [num_spks, T]
        """
        assert not self.training, "forward_wav is only used for test mode"
        T = wav.size(-1)
        if wav.ndim == 1:
            wav = wav.reshape(1, wav.size(0))
        assert wav.ndim == 2  # [1, T]
        slice_stride = slice_size // 2
        # pad wav to integer multiple of slice_stride
        T_padded = max(int(np.ceil(T / slice_stride)), 2) * slice_stride
        wav = F.pad(wav, (0, T_padded - T))
        slices = wav.unfold(
            dimension=-1, size=slice_size, step=slice_stride
        )  # [1, slice_nb, slice_size]
        slice_nb = slices.size(1)
        slices = slices.squeeze(0).unsqueeze(1)
        tf_rep = self.enc_activation(self.encoder(slices))
        est_masks_list = self.masker(tf_rep)
        selector_input = est_masks_list[-1]  # [slice_nb, bn_chan, chunk_size, n_chunks]
        selector_output = self.decoder_select.selector(selector_input).reshape(
            slice_nb, -1
        )  # [slice_nb, num_decs]
        est_idx, _ = selector_output.argmax(-1).mode()
        est_spks = self.decoder_select.n_srcs[est_idx]
        output_wavs, _ = self.decoder_select(
            est_masks_list, tf_rep, ground_truth=[est_spks] * slice_nb
        )  # [slice_nb, 1, n_spks, slice_size]
        output_wavs = output_wavs.squeeze(1)[:, :est_spks, :]
        # TODO: overlap and add (with division)
        output_cat = output_wavs.new_zeros(est_spks, slice_nb * slice_size)
        output_cat[:, :slice_size] = output_wavs[0]
        start = slice_stride
        for i in range(1, slice_nb):
            end = start + slice_size
            overlap_prev = output_cat[:, start : start + slice_stride].unsqueeze(0)
            overlap_next = output_wavs[i : i + 1, :, :slice_stride]
            pw_losses = pairwise_neg_sisdr(overlap_next, overlap_prev)
            _, best_indices = PITLossWrapper.find_best_perm(pw_losses)
            reordered = PITLossWrapper.reorder_source(output_wavs[i : i + 1, :, :], best_indices)
            output_cat[:, start : start + slice_size] += reordered.squeeze(0)
            output_cat[:, start : start + slice_stride] /= 2
            start += slice_stride
        return output_cat[:, :T]


class DPRNN_MultiStage(nn.Module):
    """Implementation of the Dual-Path-RNN model,
    with multi-stage output, without Conv2D projection
    """

    def __init__(
        self,
        in_chan,
        bn_chan,
        hid_size,
        chunk_size,
        hop_size,
        n_repeats,
        norm_type,
        bidirectional,
        rnn_type,
        use_mulcat,
        num_layers,
        dropout,
    ):
        super(DPRNN_MultiStage, self).__init__()
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        self.net = nn.ModuleList([])
        for i in range(self.n_repeats):
            self.net.append(
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    use_mulcat=use_mulcat,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )

    def forward(self, mixture_w):
        """Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            list of (:class:`torch.Tensor`): Tensor of shape $(batch, bn_chan, chunk_size, n_chunks)
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output_list = []
        for i in range(self.n_repeats):
            output = self.net[i](output)
            output_list.append(output)
        return output_list


class SingleDecoder(nn.Module):
    """
    Base decoder module, including the projection layer from (bn_chan) to (n_src * bn_chan).
    Takes a single example mask and encoding, outputs waveform
    """

    def __init__(
        self, kernel_size, stride, in_chan, n_src, bn_chan, chunk_size, hop_size, mask_act
    ):
        super(SingleDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_src = n_src
        self.mask_act = mask_act

        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, in_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        _, self.trans_conv = make_enc_dec(
            "free", kernel_size=kernel_size, stride=stride, n_filters=in_chan
        )

    def forward(self, output, mixture_w):
        """
        Args:
            output: LSTM output, Tensor of shape $(num_stages, bn_chan, chunk_size, n_chunks)$
            mixture_w: Encoder output, Tensor of shape $(num_stages, in_chan, nframes)
        outputs:
            est_wavs: Signal, Tensor of shape $(num_stages, n_src, T)
        """
        batch, bn_chan, chunk_size, n_chunks = output.size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.bn_chan == bn_chan
        assert self.in_chan == in_chan
        assert self.chunk_size == chunk_size
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.reshape(batch, self.n_src, self.in_chan, n_frames)
        mixture_w = mixture_w.unsqueeze(1)
        source_w = est_mask * mixture_w
        source_w = source_w.reshape(batch * self.n_src, self.in_chan, n_frames)
        est_wavs = self.trans_conv(source_w)
        est_wavs = est_wavs.reshape(batch, self.n_src, -1)
        return est_wavs


class Decoder_Select(nn.Module):
    """Selects which SingleDecoder to use, as well as whether to use multiloss, as proposed in [1]
    References
        [1] "Multi-Decoder DPRNN: High Accuracy Source Counting and Separation",
            Junzhe Zhu, Raymond Yeh, Mark Hasegawa-Johnson. https://arxiv.org/abs/2011.12022
    """

    def __init__(
        self, kernel_size, stride, in_chan, n_srcs, bn_chan, chunk_size, hop_size, mask_act
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_chan = in_chan
        self.n_srcs = n_srcs
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.mask_act = mask_act

        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.decoders = torch.nn.ModuleList()
        for n_src in n_srcs:
            self.decoders.append(
                SingleDecoder(
                    kernel_size=kernel_size,
                    stride=stride,
                    in_chan=in_chan,
                    n_src=n_src,
                    bn_chan=bn_chan,
                    chunk_size=chunk_size,
                    hop_size=hop_size,
                    mask_act=mask_act,
                )
            )
        self.selector = nn.Sequential(
            nn.Conv2d(bn_chan, in_chan, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Conv2d(in_chan, len(n_srcs), 1),
        )

    def forward(self, output_list, mixture_w, ground_truth):
        """Forward
        Args:
            output_list: list of $(batch, bn_chan, chunk_size, n_chunks)$
            mixture_w: torch.Tensor, $(batch, in_chan, n_frames)$
            ground_truth: None, or list of [B] ints, or Long Tensor of $(B)
                if None, use inferred number of speakers to determine output shape
        Output:
            output_wavs: torch.Tensor, $(batch, num_stages, max_spks, T)$
                where the speaker dimension is padded for examples with num_spks < max_spks
                if training, num_stages=n_repeats; otherwise, num_stages=1
            selector_output: output logits from selector module. torch.Tensor, $(batch, num_stages, num_decoders)$
        """
        batch, bn_chan, chunk_size, n_chunks = output_list[0].size()
        _, in_chan, n_frames = mixture_w.size()
        assert self.chunk_size == chunk_size
        if not self.training:
            output_list = output_list[-1:]
        num_stages = len(output_list)
        # [batch, num_stages, bn_chan, chunk_size, n_chunks]
        output = torch.stack(output_list, 1).reshape(
            batch * num_stages, bn_chan, chunk_size, n_chunks
        )
        selector_output = self.selector(output).reshape(batch, num_stages, -1)
        output = output.reshape(batch, num_stages, bn_chan, chunk_size, n_chunks)
        # [batch, num_stages, in_chan, n_frames]
        mixture_w = mixture_w.unsqueeze(1).repeat(1, num_stages, 1, 1)
        if ground_truth is not None:  # oracle
            decoder_selected = torch.LongTensor([self.n_src2idx[truth] for truth in ground_truth])
        else:
            assert num_stages == 1  # can't use select with multistage
            decoder_selected = selector_output.reshape(batch, -1).argmax(1)
        T = self.kernel_size + self.stride * (n_frames - 1)
        output_wavs = torch.zeros(batch, num_stages, max(self.n_srcs), T).to(output.device)
        for i in range(batch):
            output_wavs[i, :, : self.n_srcs[decoder_selected[i]], :] = self.decoders[
                decoder_selected[i]
            ](output[i], mixture_w[i])
        return output_wavs, selector_output


def load_best_model(train_conf, exp_dir, sample_rate):
    """Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` of `checkpoints` directory in it.

    Returns:
        nn.Module the best (or last) pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf, sample_rate=sample_rate)
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, "best_k_models.json"), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, "checkpoints/"))
        all_ckpt = [
            (ckpt, int("".join(filter(str.isdigit, os.path.basename(ckpt)))))
            for ckpt in all_ckpt
            if ckpt.find("ckpt") >= 0
        ]
        all_ckpt.sort(key=lambda x: x[1])
        best_model_path = os.path.join(exp_dir, "checkpoints", all_ckpt[-1][0])
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.eval()
    return model


# Training notes:
# Weight different stages in accordance with facebook code
if __name__ == "__main__":
    network = MultiDecoderDPRNN(n_srcs=[2, 3], bn_chan=32, hid_size=32, n_filters=16)
    # training
    input = torch.rand(2, 3200)
    wavs, selector_output = network(input, [3, 2])
    print(wavs.shape)
    assert (wavs[1, :, 2] == 0).all()
    # validation
    network.eval()
    wavs, selector_output = network(input)
    print(wavs.shape)
    # test
    input_wav = torch.rand(64351)
    output_wavs = network.forward_wav(input_wav)
    print(output_wavs.shape)
