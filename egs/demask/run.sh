#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=

librispeech_dir=$storage_dir/LibriSpeech
rir_dir=$storage_dir/rir_data
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
sample_rate=16000

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading required Datasets"
  echo "Downloading LibriSpeech/train-clean-360 into $librispeech_dir"
	# If downloading stalls for more than 20s, relaunch from previous state.
	wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $librispeech_dir
	tar -xzf $storage_dir/train-clean-360.tar.gz -C $librispeech_dir
	rm -rf $storage_dir/train-clean-360.tar.gz

	wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $librispeech_dir
	tar -xzf $storage_dir/dev-clean.tar.gz -C $librispeech_dir
	rm -rf $storage_dir/dev-clean.tar.gz


  wget -c --tries=0 --read-timeout=20 https://zenodo.org/record/3743844/files/FUSS_rir_data.tar.gz $rir_dir
	tar -xzf $rir_dir/FUSS_rir_data.tar.gz -C $rir_dir
	rm -rf $rir_dir/FUSS_rir_data.tar.gz

fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: parsing the datasets"
  python local/parse_data.py --librispeech_path $librispeech_dir --rir_path $rir_dir --out $dumpdir
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_dprnn_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 2 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--clean_speech_train $clean_speech_train \
		--clean_speech_valid $clean_speech_valid \
	  --rir_train $rir_train \
	  --rir_valid $rir_valid \
		--sample_rate $sample_rate \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "wham/DPRNN" > $expdir/publish_dir/recipe_name.txt
fi
