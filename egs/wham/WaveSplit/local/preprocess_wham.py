import argparse
import json
import os
import soundfile as sf
import glob


def preprocess_task(task, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if task == "mix_both":
        mix_both = glob.glob(os.path.join(in_dir, "mix_both", "*.wav"))
        examples = []
        for mix in mix_both:
            filename = mix.split("/")[-1]
            spk1_id = filename.split("_")[0][:3]
            spk2_id = filename.split("_")[2][:3]
            length = len(sf.SoundFile(mix))

            noise = os.path.join(in_dir, "noise", filename)
            s1 = os.path.join(in_dir, "s1", filename)
            s2 = os.path.join(in_dir, "s2", filename)

            ex = {"mix": mix, "sources": [s1 ,s2], "noise": noise, "spk_id": [spk1_id, spk2_id], "length": length}
            examples.append(ex)

        with open(os.path.join(out_dir, 'mix_both.json'), 'w') as f:
            json.dump(examples, f, indent=4)

    elif task == "mix_clean":
        mix_clean = glob.glob(os.path.join(in_dir, "mix_clean", "*.wav"))
        examples = []
        for mix in mix_clean:
            filename = mix.split("/")[-1]
            spk1_id = filename.split("_")[0][:3]
            spk2_id = filename.split("_")[2][:3]
            length = len(sf.SoundFile(mix))

            s1 = os.path.join(in_dir, "s1", filename)
            s2 = os.path.join(in_dir, "s2", filename)

            ex = {"mix": mix, "sources": [s1, s2], "spk_id": [spk1_id, spk2_id], "length": length}
            examples.append(ex)

        with open(os.path.join(out_dir, 'mix_clean.json'), 'w') as f:
            json.dump(examples, f, indent=4)

    elif task == "mix_single":
        mix_single = glob.glob(os.path.join(in_dir, "mix_single", "*.wav"))
        examples = []
        for mix in mix_single:
            filename = mix.split("/")[-1]
            spk1_id = filename.split("_")[0][:3]
            length = len(sf.SoundFile(mix))

            s1 = os.path.join(in_dir, "s1", filename)

            ex = {"mix": mix, "sources": [s1], "spk_id": [spk1_id], "length": length}
            examples.append(ex)

        with open(os.path.join(out_dir, 'mix_single.json'), 'w') as f:
            json.dump(examples, f, indent=4)
    else:
        raise EnvironmentError


def preprocess(inp_args):
    tasks = ['mix_both', 'mix_clean', 'mix_single']
    for split in ["tr", "cv", "tt"]:
        for task in tasks:
            preprocess_task(task, os.path.join(inp_args.in_dir, split), os.path.join(inp_args.out_dir, split))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("WHAM data preprocessing")
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of wham including tr, cv and tt')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    args = parser.parse_args()
    print(args)
    preprocess(args)
