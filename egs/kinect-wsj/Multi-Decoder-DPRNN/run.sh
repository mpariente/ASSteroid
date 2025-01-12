#!/bin/bash

echo "pretrained model can be found at: https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN"

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=
# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir= # Directory containing sphere files
# If you already have wsj0 wav files (converted from sphere format).
wsj0_wav_dir="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/wsj0_wav/"
# If you already have kinect_wsj  specify the path in the kinect_wsj_path and  and start from stage 2.
wsj0mix_wav_dir=
chime_path="/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/CHiME5/audio/"
dihard_path="/srv/storage/talc@talc-data.nancy/multispeech/corpus/DIHARD2/LDC2019E31_Second_DIHARD_Challenge_Development_Data/data/multichannel/sad/"
# Path to save the data
kinect_wsj_path="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid/egs/kinect-wsj/DeepClustering/dataset/2speakers_reverb_kinect"
# After running the recipe a first time, you can run it from stage 3 directly to train new models.
# Path to final kinect-wsj data, run from stage 3
data="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid/egs/kinect-wsj/DeepClustering/data"

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python
# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=3 # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=16000
mode=max
#n_srcs=(2 3 4 5)
n_srcs=(2)

# Training
batch_size=32
num_workers=8
optimizer=rmsprop
lr=0.0001
weight_decay=0.0
epochs=200
lambda=0.05
resume_from=

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate / 1000))
suffix=2speakers/wav${sr_string}k/$mode
dumpdir=$data/$suffix # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

if [[ $stage -le 0 ]]; then
	# echo "Create wsj0-mix files and start again from stage 1"; exit 1
	mkdir -p $dumpdir/kinect_wsj/code
	git clone https://github.com/sunits/Reverberated_WSJ_2MIX.git $dumpdir/kinect_wsj/code
fi

if [[ $stage -le 1 ]]; then
	# mkdir -p $dumpdir/kinect_wsj/code
	# git clone https://github.com/sunits/Reverberated_WSJ_2MIX.git  $dumpdir/kinect_wsj/code
	cd $dumpdir/kinect_wsj/code
	echo "Stage 1: create_corrupted_speech"
	./create_corrupted_speech.sh --stage 0 --wsj_data_path $wsj0_wav_dir \
		--chime5_wav_base $chime_path \
		--dihard_sad_label_path $dihard_path --dest $kinect_wsj_path
fi

if [[ $stage -le 2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 16; do
		for mode_option in max; do
			#for mode_option in min; do
			for tmp_nsrc in 2; do
				tmp_dumpdir=data/${tmp_nsrc}speakers/wav${sr_string}k/$mode_option
				echo "Generating json files in $tmp_dumpdir"
				[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
				#local_kinect_dir=$kinect_wsj_path/wav${sr_string}k/$mode_option/
				local_kinect_dir=$kinect_wsj_path/2speakers_reverb_kinect_chime_noise_corrected/wav${sr_string}k/$mode_option/
				$python_path local/preprocess_kinect_wsj.py --in_dir $local_kinect_dir --n_src $tmp_nsrc \
					--out_dir $tmp_dumpdir
			done
		done
	done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=$(
		IFS=$''
		echo "${n_srcs[*]}"
	)sep_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_mddprnn_${tag}
mkdir -p $expdir && echo $uuid >>$expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"
	echo "visible cuda devices are ${id[*]}"
	mkdir -p logs
	CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--sample_rate $sample_rate \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--epochs $epochs \
		--batch_size $batch_size \
		--lambda $lambda \
		--num_workers $num_workers \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	--resume_from $resume_from
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	expdir=exp/tmp
	echo "Stage 4 : Evaluation"
	echo "If you want to change n_srcs, please change the config file"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
