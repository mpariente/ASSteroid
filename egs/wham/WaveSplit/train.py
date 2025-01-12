import os
import argparse
import json

import torch
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from collections import OrderedDict

from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


from losses import SpeakerVectorLoss, ClippedSDR
from wavesplit import SpeakerStack, SeparationStack
from asteroid.filterbanks import make_enc_dec
from wavesplitwham import WaveSplitWhamDataset
from argparse import Namespace
from asteroid.utils import flatten_dict
from copy import deepcopy

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
# This can be changed: `python train.py --gpus 0,1` will only train on 2 GPUs.
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')

warnings.simplefilter("ignore", UserWarning)


class Wavesplit(pl.LightningModule): # redefinition

    def __init__(self, spk_stack, sep_stack, optimizer, spk_loss, sep_loss, train_loader,
                 val_loader=None, scheduler=None, config=None):
        super().__init__()

        #self.spk_stack = SpeakerStack(256, 2, 1, 1)


        #self.spk_loss = SpeakerVectorLoss(101, 256, False, "distance", 10)
        self.spk_stack = spk_stack
        self.sep_stack = sep_stack
        self.optimizer = optimizer
        self.sep_loss = sep_loss
        self.spk_loss = spk_loss

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        # hparams will be logged to Tensorboard as text variables.
        # torch doesn't support None in the summary writer for now, convert
        # None to strings temporarily.
        # See https://github.com/pytorch/pytorch/issues/33140
        self.hparams = Namespace(**self.none_to_string(flatten_dict(config)))

        #n_speakers = 100
        #embed_dim = 128
        #spk_emb = torch.eye(max(n_speakers, embed_dim))  # one-hot init works better according to Neil
        #spk_emb = spk_emb[:n_speakers, :embed_dim]

        #self.oracle = spk_emb.cuda()

    def forward(self, *args, **kwargs):
        """ Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """

        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb):
        """ Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note:: This is typically the method to overwrite when subclassing
            `System`. If the training and validation steps are different
            (except for loss.backward() and optimzer.step()), then overwrite
            `training_step` and `validation_step` instead.
        """
        inputs, targets, spk_ids = batch
        spk_embed = self.spk_stack(inputs)

        spk_loss, reordered_embed = self.spk_loss(spk_embed, torch.ones((spk_embed.shape[0],
                                                                    spk_embed.shape[1],spk_embed.shape[-1])).to(spk_embed.device), spk_ids)
        reordered_embed = reordered_embed.mean(-1)

        #reordered_embed = self.oracle[spk_ids]
        #b, n_spk, spk_vec_size = reordered_embed.size()

        separated = self.sep_stack(inputs, torch.cat((reordered_embed[:, 0], reordered_embed[:, 1]), 1))

        sep_loss = 0
        for i, o in enumerate(separated):
            o = self.pad_output_to_inp(o, inputs)
            last = self.sep_loss(o, targets).mean()
            sep_loss += last
        sep_loss = sep_loss / (i+1)
        loss = sep_loss + spk_loss

        return loss, spk_loss, last.mean()

    @staticmethod
    def pad_output_to_inp(output, inp):
        """ Pad first argument to have same size as second argument"""
        inp_len = inp.size(-1)
        output_len = output.size(-1)
        return torch.nn.functional.pad(output, [0, inp_len - output_len])

    def training_step(self, batch, batch_nb):
        """ Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'loss'``: loss

            ``'log'``: dict with tensorboard logs

        """
        loss, spk_loss, sep_loss = self.common_step(batch, batch_nb)
        tqdm_dict = {'train_loss': loss, "spk_loss": spk_loss, "sep_loss": sep_loss}
        tensorboard_logs = {'train_loss': loss, "spk_loss": spk_loss, "sep_loss": sep_loss}

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tensorboard_logs
        })
        return output

    def validation_step(self, batch, batch_nb):
        """ Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'val_loss'``: loss
        """
        loss, spk_loss, sep_loss = self.common_step(batch, batch_nb)
        return {'val_loss': loss, 'val_spk_loss': spk_loss, 'val_sep_loss': sep_loss}

    def validation_end(self, outputs):
        """ How to aggregate outputs of `validation_step` for logging.

        Args:
           outputs (list[dict]): List of validation losses, each with a
           ``'val_loss'`` key

        Returns:
            dict: Average loss

            ``'val_loss'``: Average loss on `outputs`

            ``'log'``: Tensorboard logs

            ``'progress_bar'``: Tensorboard logs
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_spk_loss = torch.stack([x['val_spk_loss'] for x in outputs]).mean()
        avg_sep_loss = torch.stack([x['val_sep_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, "val_spk_loss": avg_spk_loss, "val_sep_loss": avg_sep_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def unsqueeze_if_dp_or_ddp(self, *values):
        """ Apply unsqueeze(0) to all values if training is done with dp
            or ddp. Unused now."""
        if self.trainer.use_dp or self.trainer.use_ddp2:
            values = [v.unsqueeze(0) for v in values]
        if len(values) == 1:
            return values[0]
        return values

    def configure_optimizers(self):
        """ Required by pytorch-lightning. """
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    @pl.data_loader
    def tng_dataloader(self):  # pragma: no cover
        """ Deprecated."""
        pass

    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint['training_config'] = self.config
        return checkpoint

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    @staticmethod
    def none_to_string(dic):
        """ Converts `None` to  ``'None'`` to be handled by torch summary writer.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
        return dic


def main(conf):
    train_set = WaveSplitWhamDataset(conf['data']['train_dir'], conf['data']['task'],
                            sample_rate=conf['data']['sample_rate'], segment=conf['data']['segment'],
                            nondefault_nsrc=conf['data']['nondefault_nsrc'])
    val_set = WaveSplitWhamDataset(conf['data']['valid_dir'], conf['data']['task'],
                          sample_rate=conf['data']['sample_rate'],
                          nondefault_nsrc=conf['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    # Update number of source values (It depends on the task)
    conf['masknet'].update({'n_src': train_set.n_src})
    spk_stack = SpeakerStack(2, 256) # inner dim is 256 instead of 512 from paper to spare mem 13 layers as in the paper.
    sep_stack = SeparationStack(2, 256, 512, 10, 4) # 40 layers.
    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    # Define scheduler
    spk_loss = SpeakerVectorLoss(101, 256, loss_type="distance") # 100 speakers in WHAM dev and train, 256 embed dim
    sep_loss = ClippedSDR(-30)

    params = list(spk_stack.parameters()) + list(sep_stack.parameters()) + list(spk_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=0.003)
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    system = Wavesplit(spk_stack, sep_stack, optimizer, spk_loss, sep_loss, train_loader, val_loader, scheduler, conf)
    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    if not torch.cuda.is_available():
        print('No available GPU were found, set gpus to None')
        conf['main_args']['gpus'] = None
    trainer = pl.Trainer(max_nb_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_save_path=exp_dir,
                         gpus=conf['main_args']['gpus'],
                         )
    trainer.fit(system)

    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(checkpoint.best_k_models, f, indent=0)


if __name__ == '__main__':
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
