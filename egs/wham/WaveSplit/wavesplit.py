from torch import nn
import torch
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg


class Conv1DBlock(nn.Module):

    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, norm_type="gLN"):
        super(Conv1DBlock, self).__init__()

        conv_norm = norms.get(norm_type)
        depth_conv1d = nn.Conv1d(in_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation)

        self.out = nn.Sequential(depth_conv1d, nn.PReLU(), conv_norm(hid_chan))

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""

        return self.out(x)


class SepConv1DBlock(nn.Module):

    def __init__(self, in_chan, hid_chan, spk_vec_chan, kernel_size, padding,
                 dilation, norm_type="gLN", use_FiLM=True):
        super(SepConv1DBlock, self).__init__()

        self.use_FiLM = use_FiLM
        conv_norm = norms.get(norm_type)
        self.depth_conv1d = nn.Conv1d(in_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation)
        self.out = nn.Sequential(nn.PReLU(),
                                          conv_norm(hid_chan))

        # FiLM conditioning
        if self.use_FiLM:
            self.mul_lin = nn.Linear(spk_vec_chan, hid_chan)
        self.add_lin = nn.Linear(spk_vec_chan, hid_chan)

    def apply_conditioning(self, spk_vec, squeezed):
        bias = self.add_lin(spk_vec)
        if self.use_FiLM:
            mul = self.mul_lin(spk_vec)
            return mul.unsqueeze(-1)*squeezed + bias.unsqueeze(-1)
        else:
            return squeezed + bias.unsqueeze(-1)

    def forward(self, x, spk_vec):
        """ Input shape [batch, feats, seq]"""

        conditioned = self.apply_conditioning(spk_vec, self.depth_conv1d(x))

        return self.out(conditioned)


class SpeakerStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, n_src, embed_dim, n_blocks=14, n_repeats=1,
                 kernel_size=3,
                 norm_type="gLN"):
        
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
                self.TCN.append(Conv1DBlock(in_chan, embed_dim,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
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
        batch, n_frames = mixture_w.size()
        output = mixture_w.unsqueeze(1)
        for i in range(len(self.TCN)):
            if i == 0:
                output = self.TCN[i](output)
            else:
                residual = self.TCN[i](output)
                output = output + residual
        emb = self.mask_net(output)

        emb = emb.view(batch, self.n_src, self.embed_dim, n_frames)
        emb = emb / torch.sqrt(torch.sum(emb**2, 2, keepdim=True))
        return emb


class SeparationStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, src, embed_dim=256, spk_vec_dim=512, n_blocks=10, n_repeats=4,
                 kernel_size=3,
                 norm_type="gLN", return_all_layers=True):

        super(SeparationStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.src = src
        self.embed_dim = embed_dim
        self.return_all = return_all_layers

        # layer_norm = norms.get(norm_type)(in_chan)
        # bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        # self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if r == 0 and x == 0:
                    in_chan = 1
                else:
                    in_chan = embed_dim
                padding = (kernel_size - 1) * 2 ** x // 2
                if not self.return_all:
                    self.TCN.append(SepConv1DBlock(in_chan, embed_dim, spk_vec_dim,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
                else:
                    self.TCN.append(nn.ModuleList([ SepConv1DBlock(in_chan, embed_dim, spk_vec_dim,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type), nn.Conv1d(embed_dim, self.src, 1)]))

        self.out = nn.Conv1d(embed_dim, self.src, 1)

    def forward(self, mixture_w, spk_vectors):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        output = mixture_w.unsqueeze(1)
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
            out = output

        return out


if __name__ == "__main__":
    sep = SeparationStack(2, 256, 512, 10, 3, kernel_size=3)
    wave = torch.rand((2, 16000))
    spk_vectors = torch.rand((2, 2, 256))
    out = sep(wave, spk_vectors.reshape(2, 2*256))


