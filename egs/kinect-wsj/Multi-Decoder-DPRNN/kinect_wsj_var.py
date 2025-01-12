import torch
from torch.utils import data
import os
import numpy as np
import soundfile as sf
from asteroid.data.wsj0_mix import Wsj0mixDataset


def make_dataloaders(
    train_dir,
    valid_dir,
    n_src=2,
    sample_rate=16000,
    segment=4.0,
    batch_size=4,
    num_workers=None,
    **kwargs,
):
    num_workers = num_workers if num_workers else batch_size
    train_set = KinectWsjMixDataset(train_dir,
                                    n_src=n_src,
                                    sample_rate=sample_rate,
                                    segment=segment)
    val_set = KinectWsjMixDataset(valid_dir,
                                  n_src=n_src,
                                  sample_rate=sample_rate,
                                  segment=segment)
    train_loader = data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = data.DataLoader(
        val_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader


class KinectWsjMixDataset(Wsj0mixDataset):
    """Dataset class for the KinectWSJ-mix source separation dataset.

    Args:
        json_dir (str): The path to the directory containing the json files.
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        n_src (int, optional): Number of sources in the training targets.

    References
        "Analyzing the impact of speaker localization errors on speech separation
        for automatic speech recognition", Sunit Sivasankaran et al. 2020.
    """

    dataset_name = "Kinect-WSJ"

    def __init__(self, json_dir, n_src=2, sample_rate=16000, segment=4.0):
        super().__init__(json_dir,
                         n_src=n_src,
                         sample_rate=sample_rate,
                         segment=segment)
        noises = []
        for i in range(len(self.mix)):
            path = self.mix[i][0]
            # Warning: linux specific
            path_splits = path.split("/")
            path_splits[-2] = "noise"
            noise_path = "/" + os.path.join(*path_splits)
            noises.append([noise_path, self.mix[i][1]])
        self.noises = noises

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, stack([source_arrays]), noise
            mixture is of dimension [samples, channels]
            sources are of dimension [n_src, samples, channels]
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0],
                       start=rand_start,
                       stop=stop,
                       dtype="float32",
                       always_2d=True)
        noise, _ = sf.read(self.noises[idx][0],
                           start=rand_start,
                           stop=stop,
                           dtype="float32",
                           always_2d=True)
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros_like(x)
            else:
                s, _ = sf.read(src[idx][0],
                               start=rand_start,
                               stop=stop,
                               dtype="float32",
                               always_2d=True)
            source_arrays.append(s)
        sources = torch.from_numpy(np.stack(source_arrays))
        mixture = torch.from_numpy(x)[..., 0]
        sources = sources[..., 0]

        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = super().get_infos()
        infos["licenses"].append(chime5_license)
        return infos


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = batch_size, each entry is a tuple of (mixture, sources)
    Returns:
        mixtures_tensor: B x T, torch.Tensor, padded mixtures
        source_tensor: B x C x T, torch.Tensor, padded in both channel and time dimension
        ilens : B, torch.Tensor, length of each mixture
        num_sources : B, torch.Tensor, number of sources for each mixture
    """
    ilens = [len(mixture) for mixture, _ in batch]
    num_sources = [len(sources) for _, sources in batch]
    mixture_tensor = torch.zeros(len(batch), max(ilens))
    source_tensor = torch.zeros(len(batch), max(num_sources), max(ilens))

    for i, (mixture, sources) in enumerate(batch):  # compute length to pad to
        assert len(mixture) == len(sources[0])
        mixture_tensor[i, :ilens[i]] = torch.Tensor(mixture).float()
        source_tensor[i, :num_sources[i], :ilens[i]] = torch.Tensor(
            np.stack(sources, axis=0)).float()
    ilens = torch.Tensor(np.stack(ilens)).int()
    num_sources = torch.Tensor(np.stack(num_sources)).int()

    return mixture_tensor, source_tensor, ilens, num_sources


chime5_license = dict(
    title="The CHiME-5 speech corpus",
    title_link="http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/index.html",
    author="Jon Barker, Shinji Watanabe and Emmanuel Vincent",
    author_link=
    "http://spandh.dcs.shef.ac.uk/chime_challenge/chime2018/contact.html",
    license="CHiME-5 data licence - non-commercial 1.00",
    license_link="https://licensing.sheffield.ac.uk/i/data/chime5.html",
    non_commercial=True,
)
