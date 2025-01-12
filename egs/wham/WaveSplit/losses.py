from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from itertools import permutations
from asteroid.losses.sdr import MultiSrcNegSDR, SingleSrcNegSDR
from asteroid.losses import PITLossWrapper, PairwiseNegSDR,pairwise_neg_sisdr
import math


class ClippedSDR(nn.Module):

    def __init__(self, clip_value=-30):
        super(ClippedSDR, self).__init__()

        self.snr = MultiSrcNegSDR("snr")
        self.clip_value = float(clip_value)

    def forward(self, est_targets, targets):

        return torch.clamp(self.snr(est_targets, targets), min=self.clip_value)


class SpeakerVectorLoss(nn.Module):

    def __init__(self, n_speakers, embed_dim=32, learnable_emb=True, loss_type="global",
                 weight=2, distance_reg=0.3, gaussian_reg=0.2, return_oracle=False):
        super(SpeakerVectorLoss, self).__init__()

        self.learnable_emb = learnable_emb
        self.loss_type = loss_type
        self.weight = float(weight)
        self.distance_reg = float(distance_reg)
        self.gaussian_reg = float(gaussian_reg)
        self.return_oracle = return_oracle

        assert loss_type in ["distance", "global", "local"]

        spk_emb = torch.eye(max(n_speakers, embed_dim)) # one-hot init works better according to Neil
        spk_emb = spk_emb[:n_speakers, :embed_dim]

        if learnable_emb == True:
            self.spk_embeddings = nn.Parameter(spk_emb)
        else:
            self.register_buffer("spk_embeddings", spk_emb)

        if loss_type != "distance":
            self.alpha = nn.Parameter(torch.Tensor([1.]))
            self.beta = nn.Parameter(torch.Tensor([0.]))

    @staticmethod
    def _l_dist_speaker(c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        c_spk = c_spk_vec_perm[:, 0]
        pair_dist = ((c_spk.unsqueeze(1) - c_spk_vec_perm)**2).mean(2)
        pair_dist = pair_dist[:, 1:]
        distance = ((c_spk_vec_perm - utt_embeddings)**2).mean(dim=(1,2))
        return distance + F.relu(1. - pair_dist).mean(dim=(1))

    def _l_local_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        alpha = torch.clamp(self.alpha, 1e-8)

        distance = alpha*((c_spk_vec_perm - utt_embeddings)**2).mean(2) + self.beta
        distances = alpha * ((c_spk_vec_perm.unsqueeze(1) - utt_embeddings.unsqueeze(2)) ** 2).mean(3) + self.beta
        distances = torch.exp(-distances).mean(1)

        return (distance + torch.log(distances)).mean(1)

    def _l_global_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        alpha = torch.clamp(self.alpha, 1e-8)

        distance_utt = alpha*((c_spk_vec_perm - utt_embeddings)**2).sum(2) + self.beta

        B, src, embed_dim, frames = c_spk_vec_perm.size()
        spk_embeddings = spk_embeddings.reshape(1, spk_embeddings.shape[0], embed_dim, 1).expand(B, -1, -1, frames)
        distances = alpha * ((c_spk_vec_perm.unsqueeze(1) - spk_embeddings.unsqueeze(2)) ** 2).sum(3) + self.beta
        distances = torch.exp(-distances).sum(1)

        return (distance_utt + torch.log(distances)).sum(1)


        # exp normalize trick
        #with torch.no_grad():
         #   b = torch.max(distances, dim=1, keepdim=True)[0]
        #out = -distance_utt + b.squeeze(1) - torch.log(torch.exp(-distances + b).sum(1))
        #return out.sum(1)


    def forward(self, speaker_vectors, spk_mask, spk_labels):

        if self.gaussian_reg:
            noise = torch.randn(self.spk_embeddings.size(), device=speaker_vectors.device)*math.sqrt(self.gaussian_reg)
            spk_embeddings = self.spk_embeddings + noise
        else:
            spk_embeddings = self.spk_embeddings

        if self.learnable_emb or self.gaussian_reg:  # re project on unit sphere

            spk_embeddings = spk_embeddings / torch.sum(spk_embeddings ** 2, -1, keepdim=True).sqrt()

        if self.distance_reg:

            pairwise_dist = (torch.abs(spk_embeddings.unsqueeze(0) - spk_embeddings.unsqueeze(1))).mean(-1).fill_diagonal_(np.inf)
            distance_reg = -torch.sum(torch.min(torch.log(pairwise_dist), dim=-1)[0])

        # speaker vectors B, n_src, dim, frames
        # spk mask B, n_src, frames boolean mask
        # spk indxs list of len B of list which contains spk label for current utterance
        B, n_src, embed_dim, frames = speaker_vectors.size()

        n_src = speaker_vectors.shape[1]
        perms = list(permutations(range(n_src)))
        if self.loss_type == "distance":
            loss_set = torch.stack([self._l_dist_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)
        elif self.loss_type == "local":
            loss_set = torch.stack([self._l_local_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)
        else:
            loss_set = torch.stack([self._l_global_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)

        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # reorder sources for each frame !!
        perms = min_loss.new_tensor(perms, dtype=torch.long)
        perms = perms[..., None, None].expand(-1, -1, B, frames)
        min_loss_idx = min_loss_idx[None, None,...].expand(1, n_src, -1, -1)
        min_loss_perm = torch.gather(perms, dim=0, index=min_loss_idx)[0]
        min_loss_perm = min_loss_perm.transpose(0, 1).reshape(B, n_src, 1, frames).expand(-1, -1, embed_dim, -1)
        # tot_loss

        spk_loss = self.weight*min_loss.mean()
        if self.distance_reg:
            spk_loss += self.distance_reg*distance_reg
        reordered_sources = torch.gather(speaker_vectors, dim=1, index=min_loss_perm)

        if self.return_oracle:
            utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
            return spk_loss, reordered_sources, utt_embeddings

        return spk_loss, reordered_sources


if __name__ == "__main__":
    n_speakers = 101
    emb_speaker = 256

    # testing exp normalize average
    #distances = torch.ones((1, 101, 4000))
    #with torch.no_grad():
     #   b = torch.max(distances, dim=1, keepdim=True)[0]
    #out = b.squeeze(1) - torch.log(torch.exp(-distances + b).sum(1))
    #out2 = - torch.log(torch.exp(-distances).sum(1))

    loss_spk = SpeakerVectorLoss(n_speakers, emb_speaker, loss_type="distance")

    speaker_vectors = torch.rand(2, 3, emb_speaker, 200)
    speaker_labels = torch.from_numpy(np.array([[1, 2, 0], [5, 2, 10]]))
    speaker_mask = torch.randint(0, 2, (2, 3, 200)) # silence where there are no speakers actually thi is test
    speaker_mask[:, -1, :] = speaker_mask[:, -1, :]*0
    loss_spk(speaker_vectors, speaker_mask, speaker_labels)

    c = ClippedSDR(-30)
    a = torch.rand((2, 3, 200))
    print(c(a, a))
















