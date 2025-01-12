import os
import argparse
import json
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.nn.parallel import DistributedDataParallel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import deep_clustering_loss
from asteroid_filterbanks.transforms import mag
from asteroid.dsp.vad import ebased_vad
from kinect_wsj_var import make_dataloaders

from model import make_model_and_optimizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir",
                    default="exp/tmp",
                    help="Full path to save best validation model")
parser.add_argument("--resume_from", default=None, help="Model to resume from")


def main(conf):
    train_loader, val_loader = make_dataloaders(**conf["data"],
                                                **conf["training"])

    model, optimizer = make_model_and_optimizer(
        conf, sample_rate=conf["data"]["sample_rate"])
    scheduler = []
    if conf["training"]["half_lr"]:
        scheduler.append(
            ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5))
    if conf["training"]["lr_decay"]:
        scheduler.append(ExponentialLR(optimizer=optimizer, gamma=0.99))
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    loss_func = WeightedPITLoss(n_srcs=conf["masknet"]["n_srcs"],
                                lamb=conf["loss"]["lambda"])
    # Put together in System
    system = VarSpkrSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}",
        monitor="avg_sdr",
        mode="max",
        save_top_k=5,
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(monitor="avg_sdr",
                          mode="max",
                          patience=30,
                          verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "dp" if torch.cuda.is_available() else None

    # Train model
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=200,
        resume_from_checkpoint=conf["main_args"]["resume_from"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)
    # Save last model for convenience
    torch.save(system.model.state_dict(),
               os.path.join(exp_dir, "final_model.pth"))


class VarSpkrSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mixture_tensor, source_tensor, ilens, num_sources = batch
        pred_tensor, selector_output = self(mixture_tensor,
                                            num_sources.tolist())
        batch_size, num_stages, _, T = pred_tensor.size()
        avg_loss = 0
        spks_sdr = []
        accuracy = 0
        for i in range(batch_size):
            est_src = pred_tensor[i, :, :num_sources[i], :ilens[i]]
            src = source_tensor[i, :num_sources[i], :ilens[i]]
            logits = selector_output[i]
            loss, pos_sdr, correctness = self.loss_func(est_src, logits, src)
            avg_loss = avg_loss + loss / batch_size
            spks_sdr.append((num_sources[i], pos_sdr))
            accuracy += correctness / batch_size
        return avg_loss, spks_sdr, accuracy

    def training_step(self, batch, batch_nb):
        avg_loss, spks_sdr, accuracy = self.common_step(batch,
                                                        batch_nb,
                                                        train=True)
        self.log("loss", avg_loss)
        for num_spks, sdr in spks_sdr:
            self.log(f"{num_spks}spks_sdr_tr", sdr)
        self.log("acc_tr", accuracy)
        return avg_loss

    def validation_step(self, batch, batch_nb):
        avg_loss, spks_sdr, accuracy = self.common_step(batch,
                                                        batch_nb,
                                                        train=False)
        self.log("val_loss", avg_loss)
        for num_spks, sdr in spks_sdr:
            self.log(f"{num_spks}spks_sdr_val", sdr)
        self.log("acc_val", accuracy)
        # SDR averaged across number of sources
        avg_sdr = torch.mean(torch.Tensor([sdr for _, sdr in spks_sdr]))
        self.log("avg_sdr", avg_sdr)

    def unpack_data(self, batch):
        """
        Args:
            batch: list, len(batch) = batch_size, each entry is a tuple of (mixture, sources)
        Returns:
            mixtures_tensor: B x T, torch.Tensor, padded mixtures
            source_tensor: B x C x T, torch.Tensor, padded in both channel and time dimension
            ilens : B, torch.Tensor, length of each mixture
            num_sources : B, torch.Tensor, number of sources for each mixture
        """

        #print("*" * 100)
        #print(len(batch))
        #mixture, source, noise = batch
        #print("*"*100)
        #print(mixture.shape)
        #print(batch.shape)
        ilens = [len(mixture[..., 0]) for mixture, source, noise in batch]
        num_sources = [len(source[..., 0]) for mixture, source, noise in batch]
        mixture_tensor = torch.zeros(len(batch), max(ilens))
        source_tensor = torch.zeros(len(batch), max(num_sources), max(ilens))

        for i, (mixture,
                sources) in enumerate(batch[:2]):  # compute length to pad to
            assert len(mixture) == len(sources[..., 0][0])
            mixture_tensor[i, :ilens[i]] = torch.Tensor(mixture[...,
                                                                0]).float()
            source_tensor[i, :num_sources[i], :ilens[i]] = torch.Tensor(
                np.stack(sources[..., 0], axis=0)).float()
        ilens = torch.Tensor(np.stack(ilens)).int()
        num_sources = torch.Tensor(np.stack(num_sources)).int()

        return mixture_tensor, source_tensor, ilens, num_sources
        '''
        ilens = [len(mixture) for mixture, _ in batch]
        num_sources = [len(sources) for _, sources in batch]
        mixture_tensor = torch.zeros(len(batch), max(ilens))
        source_tensor = torch.zeros(len(batch), max(num_sources), max(ilens))

        for i, (mixture, sources) in enumerate(batch):  # compute length to pad to
            assert len(mixture) == len(sources[0])
            mixture_tensor[i, : ilens[i]] = torch.Tensor(mixture).float()
            source_tensor[i, : num_sources[i], : ilens[i]] = torch.Tensor(
                np.stack(sources, axis=0)
            ).float()
        ilens = torch.Tensor(np.stack(ilens)).int()
        num_sources = torch.Tensor(np.stack(num_sources)).int()

        return mixture_tensor, source_tensor, ilens, num_sources
        '''


class WeightedPITLoss(nn.Module):
    """
    This loss has two components. One is the standard PIT loss, with Si-SDR summed(not mean, but sum) over each source
    under the best matching permutation. The other component is the classification loss, which is cross entropy for the
    speaker number classification head network.
    """
    def __init__(self, n_srcs, lamb=0.05):
        super().__init__()
        self.n_src2idx = {n_src: i for i, n_src in enumerate(n_srcs)}
        self.cce = nn.CrossEntropyLoss(reduction="none")
        self.lamb = lamb

    def forward(self, est_src, logits, src):
        """Forward
        Args:
            est_src: $(num_stages, n_src, T)
            logits: $(num_stages, num_decoders)
            src: $(n_src, T)
        """
        assert est_src.size()[1:] == src.size()
        num_stages, n_src, T = est_src.size()
        target_src = src.unsqueeze(0).repeat(num_stages, 1, 1)
        target_idx = self.n_src2idx[n_src]

        pw_losses = pairwise_neg_sisdr(est_src, target_src)
        sdr_loss, _ = PITLossWrapper.find_best_perm(pw_losses)
        pos_sdr = -sdr_loss[-1]

        cls_target = torch.LongTensor([target_idx] * num_stages).to(
            logits.device)
        cls_loss = self.cce(logits, cls_target)
        correctness = logits[-1].argmax().item() == target_idx

        coeffs = torch.Tensor([
            (c_idx + 1) * (1 / num_stages) for c_idx in range(num_stages)
        ]).to(logits.device)
        assert coeffs.size() == sdr_loss.size() == cls_loss.size()
        # use sum of SDR for each channel, not mean
        loss = torch.sum(coeffs * (sdr_loss * n_src + cls_loss * self.lamb))

        return loss, pos_sdr, correctness


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
