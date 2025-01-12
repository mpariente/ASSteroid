import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import Wavesplit
from dataloading import WHAMID

from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from losses import ClippedSDR, SpeakerVectorLoss

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


class WavesplitTrainer(System):
    def on_train_start(self) -> None:
        self.loss_func["spk_loss"] = self.loss_func["spk_loss"].to(self.device)

    def on_validation_epoch_start(self) -> None:
        self.loss_func["spk_loss"] = self.loss_func["spk_loss"].to(self.device)

    def training_step(self, batch, batch_nb):
        mixtures, oracle_s, oracle_ids = batch
        b, n_spk, frames = oracle_s.size()

        # spk_vectors = self.model.get_speaker_vectors(mixtures)
        # b, n_spk, embed_dim, frames = spk_vectors.size()
        # spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixtures)
        # spk_loss, reordered = self.loss_func["spk_loss"](spk_vectors, spk_activity_mask, oracle_ids)
        spk_loss = 0
        # reordered = reordered.mean(-1) # take centroid
        reordered = self.loss_func["spk_loss"].spk_embeddings[oracle_ids]

        separated = self.model.split_waves(mixtures, reordered)

        if self.model.sep_stack.return_all:
            n_layers = len(separated)
            separated = torch.stack(separated).transpose(0, 1)
            separated = separated.reshape(
                b * n_layers, n_spk, frames
            )  # in validation take only last layer
            oracle_s = (
                oracle_s.unsqueeze(1).repeat(1, n_layers, 1, 1).reshape(b * n_layers, n_spk, frames)
            )

        sep_loss = self.loss_func["sep_loss"](separated, oracle_s).mean()
        tot_loss = sep_loss + spk_loss

        tqdm_log = {"spk_loss": spk_loss, "sep_loss": sep_loss}
        tensorboard_logs = {"spk_loss/train": spk_loss, "sep_loss/train": sep_loss}

        return {"loss": tot_loss, "log": tensorboard_logs, "progress_bar": tqdm_log}

    def validation_step(self, batch, batch_nb):
        mixtures, oracle_s, oracle_ids = batch
        b, n_spk, frames = oracle_s.size()
        # spk_vectors = self.model.get_speaker_vectors(mixtures)
        ##b, n_spk, embed_dim, frames = spk_vectors.size()
        # spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixtures)
        # spk_loss, reordered = self.loss_func["spk_loss"](spk_vectors,
        #                                                spk_activity_mask,
        #                                               oracle_ids)
        # reordered = reordered.mean(-1)  # take centroid
        reordered = self.loss_func["spk_loss"].spk_embeddings[oracle_ids]
        spk_loss = 0

        separated = self.model.split_waves(mixtures, reordered)

        if self.model.sep_stack.return_all:
            separated = separated[-1]

        sep_loss = self.loss_func["sep_loss"](separated, oracle_s).mean()
        tot_loss = sep_loss + spk_loss

        tensorboard_logs = {"spk_loss/val": spk_loss, "sep_loss/val": sep_loss}

        return {"val_loss": tot_loss.item(), "log": tensorboard_logs}


def main(conf):
    train_set = WHAMID(
        conf["data"]["train_dir"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        nondefault_nsrc=conf["data"]["nondefault_nsrc"],
    )
    val_set = WHAMID(
        conf["data"]["valid_dir"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        nondefault_nsrc=conf["data"]["nondefault_nsrc"],
        segment=conf["data"]["segment"] * 2,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    # Update number of source values (It depends on the task)
    conf["masknet"].update({"n_src": train_set.n_src})

    model = Wavesplit(
        conf["masknet"]["n_src"],
        {"embed_dim": 512},
        {"embed_dim": 512, "spk_vec_dim": 512, "n_repeats": 4, "return_all_layers": False},
    )

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_spk = SpeakerVectorLoss(
        len(train_set.spk2indx), embed_dim=512, loss_type="distance", gaussian_reg=0, distance_reg=0
    )
    loss_sep = ClippedSDR()

    # optimizer takes also loss speaker as spk oracle embeddings are trainable
    optimizer = make_optimizer(
        list(model.parameters()) + list(loss_spk.parameters()), **conf["optim"]
    )
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    system = WavesplitTrainer(
        model=model,
        loss_func={"spk_loss": loss_spk, "sep_loss": loss_sep},
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
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


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
