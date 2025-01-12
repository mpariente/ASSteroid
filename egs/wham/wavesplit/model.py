from torch import nn
import torch
from asteroid.masknn import norms
from kmeans_pytorch import kmeans, kmeans_predict


class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding, dilation, norm_type="gLN"):
        super(Conv1DBlock, self).__init__()

        conv_norm = norms.get(norm_type)
        depth_conv1d = nn.Conv1d(in_chan, hid_chan, kernel_size, padding=padding, dilation=dilation)
        torch.nn.init.kaiming_uniform_(depth_conv1d.weight)

        self.out = nn.Sequential(depth_conv1d, nn.PReLU(), conv_norm(hid_chan))

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""

        return self.out(x)


class SepConv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_chan,
        spk_vec_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        use_FiLM=True,
    ):
        super(SepConv1DBlock, self).__init__()

        self.use_FiLM = use_FiLM
        conv_norm = norms.get(norm_type)
        self.depth_conv1d = nn.Conv1d(
            in_chan, hid_chan, kernel_size, padding=padding, dilation=dilation
        )
        torch.nn.init.kaiming_uniform_(self.depth_conv1d.weight)
        self.out = nn.Sequential(nn.PReLU(), conv_norm(hid_chan))

        # FiLM conditioning
        if self.use_FiLM:
            self.mul_lin = nn.Linear(spk_vec_chan, hid_chan)
            torch.nn.init.kaiming_uniform_(self.mul_lin.weight)
        self.add_lin = nn.Linear(spk_vec_chan, hid_chan)
        torch.nn.init.kaiming_uniform_(self.add_lin.weight)

    def apply_conditioning(self, spk_vec, squeezed):
        spk_vec = spk_vec.unsqueeze(-1)
        bias = self.add_lin(spk_vec.transpose(1, -1)).transpose(1, -1)
        if self.use_FiLM:
            mul = self.mul_lin(spk_vec.transpose(1, -1)).transpose(1, -1)
            return mul * squeezed + bias
        else:
            return squeezed + bias.unsqueeze(-1)

    def forward(self, x, spk_vec):
        """ Input shape [batch, feats, seq]"""

        conditioned = self.apply_conditioning(spk_vec, self.depth_conv1d(x))

        return self.out(conditioned)


class SpeakerStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(
        self, n_src, embed_dim=512, n_blocks=14, n_repeats=1, kernel_size=3, norm_type="gLN"
    ):

        super(SpeakerStack, self).__init__()
        self.embed_dim = embed_dim
        self.n_src = n_src
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                if r == 0 and x == 0:
                    in_chan = 1
                else:
                    in_chan = embed_dim
                self.TCN.append(
                    Conv1DBlock(
                        in_chan,
                        embed_dim,
                        kernel_size,
                        padding=padding,
                        dilation=2 ** x,
                        norm_type=norm_type,
                    )
                )
        mask_conv = nn.Conv1d(embed_dim, n_src * embed_dim, 1)
        self.mask_net = nn.Sequential(mask_conv)

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]

        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, _, n_frames = mixture_w.size()
        output = mixture_w
        for i in range(len(self.TCN)):
            if i == 0:
                output = self.TCN[i](output)
            else:
                residual = self.TCN[i](output)
                output = output + residual
        emb = self.mask_net(output)

        emb = emb.view(batch, self.n_src, self.embed_dim, n_frames)
        emb = emb / torch.sqrt(torch.sum(emb ** 2, 2, keepdim=True))
        return emb


class SeparationStack(nn.Module):
    def __init__(
        self,
        src,
        embed_dim=512,
        spk_vec_dim=512,
        n_blocks=10,
        n_repeats=4,
        kernel_size=3,
        norm_type="gLN",
        return_all_layers=True,
    ):

        super(SeparationStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.src = src
        self.embed_dim = embed_dim
        self.return_all = return_all_layers
        self.TCN = nn.ModuleList()
        if not self.return_all:
            self.out = nn.Conv1d(embed_dim, self.src, 1)

        for r in range(n_repeats):
            for x in range(n_blocks):
                if r == 0 and x == 0:
                    in_chan = 1
                else:
                    in_chan = embed_dim
                padding = (kernel_size - 1) * 2 ** x // 2
                if not self.return_all:
                    self.TCN.append(
                        SepConv1DBlock(
                            in_chan,
                            embed_dim,
                            spk_vec_dim * self.src,
                            kernel_size,
                            padding=padding,
                            dilation=2 ** x,
                            norm_type=norm_type,
                        )
                    )
                else:
                    conv = nn.Conv1d(embed_dim, self.src, 1)
                    torch.nn.init.kaiming_uniform_(conv.weight)
                    self.TCN.append(
                        nn.ModuleList(
                            [
                                SepConv1DBlock(
                                    in_chan,
                                    embed_dim,
                                    spk_vec_dim * self.src,
                                    kernel_size,
                                    padding=padding,
                                    dilation=2 ** x,
                                    norm_type=norm_type,
                                ),
                                conv,
                            ]
                        )
                    )

    def forward(self, mixture_w, spk_vectors):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]

        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        output = mixture_w
        outputs = []
        # output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            if i == 0:
                if self.return_all:
                    conv, linear = self.TCN[i]
                    output = conv(output, spk_vectors)
                    outputs.append(linear(output))
                else:
                    output = self.TCN[i](output, spk_vectors)
            else:
                if self.return_all:
                    conv, linear = self.TCN[i]
                    residual = conv(output, spk_vectors)
                    output = output + residual
                    outputs.append(linear(output))
                else:
                    residual = self.TCN[i](output, spk_vectors)
                    output = output + residual

        if self.return_all:
            out = outputs
        else:
            out = self.out(output)

        return out


class Wavesplit(nn.Module):
    def __init__(self, n_src, spk_stack_kwargs={}, sep_stack_kwargs={}):
        super().__init__()

        self.n_src = n_src
        self.spk_stack = SpeakerStack(n_src, **spk_stack_kwargs)
        self.sep_stack = SeparationStack(n_src, **sep_stack_kwargs)

    def _check_input_shape(self, x):
        if x.ndim < 3:
            x = x.unsqueeze(1)
        return x

    def get_speaker_vectors(self, x):
        x = self._check_input_shape(x)
        spk_embeddings = self.spk_stack(x)
        return spk_embeddings

    def split_waves(self, x, reordered_spk_vectors):
        x = self._check_input_shape(x)
        batch_sz, self.n_src, spk_vec_size = reordered_spk_vectors.size()
        return self.sep_stack(x, reordered_spk_vectors.reshape(batch_sz, self.n_src * spk_vec_size))

    def forward(self, x):
        # use only in inference
        x = self._check_input_shape(x)
        spk_embeddings = self.spk_stack(x)
        batch_sz, self.n_src, spk_vec_size, samples = spk_embeddings.size()
        reordered = []
        for b in range(spk_embeddings.shape[0]):
            cluster_ids, cluster_centers = kmeans(
                spk_embeddings[b].transpose(1, 2).reshape(self.n_src * samples, spk_vec_size),
                self.n_src,
                device=spk_embeddings.device,
            )
            reordered.append(cluster_centers)

        reordered = torch.stack(reordered)
        return self.split_waves(x, reordered)


if __name__ == "__main__":

    sep = SeparationStack(2)
    wave = torch.rand((2, 1600))

    wavesplit = Wavesplit(2)
    wavesplit(wave)
