import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf

DATASET = "WHAM"
# WHAM tasks
enh_single = {"mixture": "mix_single", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
enh_both = {"mixture": "mix_both", "sources": ["mix_clean"], "infos": ["noise"], "default_nsrc": 1}
sep_clean = {"mixture": "mix_clean", "sources": ["s1", "s2"], "infos": [], "default_nsrc": 2}
sep_noisy = {"mixture": "mix_both", "sources": ["s1", "s2"], "infos": ["noise"], "default_nsrc": 2}

WHAM_TASKS = {
    "enhance_single": enh_single,
    "enhance_both": enh_both,
    "sep_clean": sep_clean,
    "sep_noisy": sep_noisy,
}
# Aliases.
WHAM_TASKS["enh_single"] = WHAM_TASKS["enhance_single"]
WHAM_TASKS["enh_both"] = WHAM_TASKS["enhance_both"]


class WHAMID(data.Dataset):
    """Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
    """

    def __init__(self, json_dir, task, sample_rate=8000, segment=4.0, nondefault_nsrc=None):
        super(WHAMID, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(task, WHAM_TASKS.keys())
            )
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = WHAM_TASKS[task]
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json examples
        ex_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")

        with open(ex_json, "r") as f:
            examples = json.load(f)

        # Filter out short utterances only when segment is specified
        self.examples = []
        orig_len = len(examples)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for ex in examples:  # Go backward
                if ex["length"] < self.seg_len:
                    drop_utt += 1
                    drop_len += ex["length"]
                else:
                    self.examples.append(ex)

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )

        # count total number of speakers
        speakers = set()
        for ex in self.examples:
            for spk in ex["spk_id"]:
                speakers.add(spk[:3])

        print("Total number of speakers {}".format(len(list(speakers))))

        # convert speakers id into integers
        indx = 0
        spk2indx = {}
        for spk in list(speakers):
            spk2indx[spk] = indx
            indx += 1
        self.spk2indx = spk2indx

        for ex in self.examples:
            new = []
            for spk in ex["spk_id"]:
                new.append(spk2indx[spk[:3]])
            ex["spk_id"] = new

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        c_ex = self.examples[idx]
        # Random start
        if c_ex["length"] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, c_ex["length"] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(c_ex["mix"], start=rand_start, stop=stop, dtype="float32")
        # seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in c_ex["sources"]:
            s, _ = sf.read(src, start=rand_start, stop=stop, dtype="float32")
            source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))

        if np.random.random() > 0.5:  # randomly permute (not sure if it can help but makes sense)
            sources = torch.stack((sources[1], sources[0]))
            c_ex["spk_id"] = [c_ex["spk_id"][1], c_ex["spk_id"][0]]

        return torch.from_numpy(x), sources, torch.Tensor(c_ex["spk_id"]).long()


if __name__ == "__main__":
    a = WHAMID(
        "/media/sam/bx500/wavesplit/asteroid/egs/wham/wavesplit/data/wav8k/min/tt", "sep_clean"
    )

    for i in a:
        print(i[-1])
