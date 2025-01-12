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
from metrics import Penalized_PIT_Wrapper, pairwise_neg_sisdr_loss
import os
import json
import yaml
import argparse
import random
import torch
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from pprint import pprint

from asteroid.utils import tensors_to_device
from asteroid.metrics import get_metrics

from model import load_best_model, make_model_and_optimizer
from asteroid.data import KinectWsjMixDataset


parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the json files"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)


def main(conf):
    best_model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    if not os.path.exists(best_model_path):
        # make pth from checkpoint
        model = load_best_model(
            conf["train_conf"], conf["exp_dir"], sample_rate=conf["sample_rate"]
        )
        torch.save(model.state_dict(), best_model_path)
    else:
        model, _ = make_model_and_optimizer(conf["train_conf"], sample_rate=conf["sample_rate"])
        model.eval()
        model.load_state_dict(torch.load(best_model_path))
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    
    test_set = KinectWsjMixDataset(conf["test_dir"], n_src=conf["n_src"], segment=None)

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = [
            torch.Tensor(x) for x in tensors_to_device(test_set[idx], device=model_device)
        ]
        est_sources = model.separate(mix[None])
        p_si_snr = Penalized_PIT_Wrapper(pairwise_neg_sisdr_loss)(est_sources, sources)
        utt_metrics = {
            "P-Si-SNR": p_si_snr.item(),
            "counting_accuracy": float(sources.size(0) == est_sources.size(0)),
        }
        utt_metrics["mix_path"] = test_set.data[idx][0]
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            mix_np = mix[None].cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = est_sources.cpu().data.numpy()
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np[0], conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in ["P-Si-SNR", "counting_accuracy"]:
        final_results[metric_name] = all_metrics_df[metric_name].mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
