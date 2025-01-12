import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.utils import tensors_to_device

from model import load_best_model

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True,
                    help='One of `enh_single`, `enh_both`, '
                         '`sep_clean` or `sep_noisy`')
parser.add_argument('--test_dir', type=str, required=True,
                    help='Test directory including the json files')
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=50,
                    help='Number of audio examples to save, -1 means all')

compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi']


def main(conf):
    model = load_best_model(conf['train_conf'], conf['exp_dir'])
    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = WhamDataset(conf['test_dir'], conf['task'],
                           sample_rate=conf['sample_rate'],
                           nondefault_nsrc=model.masker.n_src,
                           segment=None)  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix[None, None])
        loss, reordered_sources = loss_func(est_sources, sources[None],
                                            return_est=True)
        mix_np = mix[None].cpu().data.numpy()
        sources_np = sources.squeeze().cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze().cpu().data.numpy()
        utt_metrics = get_metrics(mix_np, sources_np, est_sources_np,
                                  sample_rate=conf['sample_rate'])
        utt_metrics['mix_path'] = test_set.mix[idx][0]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0],
                     conf['sample_rate'])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx+1), src,
                         conf['sample_rate'])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(local_save_dir + "s{}_estimate.wav".format(src_idx+1),
                         est_src, conf['sample_rate'])
            # Write local metrics to the example folder.
            with open(local_save_dir + 'metrics.json', 'w') as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf['exp_dir'], 'all_metrics.csv'))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = 'input_' + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + '_imp'] = ldf.mean()
    print('Overall metrics :')
    pprint(final_results)
    with open(os.path.join(conf['exp_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(final_results, f, indent=0)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic['sample_rate'] = train_conf['data']['sample_rate']
    arg_dic['train_conf'] = train_conf

    if args.task != arg_dic['train_conf']['data']['task']:
        print("Warning : the task used to test is different than "
              "the one from training, be sure this is what you want.")

    main(arg_dic)
