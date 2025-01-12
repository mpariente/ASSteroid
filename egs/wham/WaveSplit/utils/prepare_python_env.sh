#!/bin/bash
# Usage ./utils/install_env.sh --install_dir A --asteroid_root B --pip_requires C
install_dir=~
asteroid_root=../../../../
pip_requires=../../../requirements.txt  # Expects a requirement.txt

. utils/parse_options.sh || exit 1

mkdir -p $install_dir
cd $install_dir
echo "Download and install latest version of miniconda3 into ${install_dir}"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3
pip_path=$PWD/miniconda3/bin/pip

rm Miniconda3-latest-Linux-x86_64.sh
cd -

if [[ ! -z ${pip_requires} ]]; then
	$pip_path install -r $pip_requires
fi
$pip_path install soundfile
$pip_path install -e $asteroid_root
#$pip_path install ${asteroid_root}/\[""evaluate""\]
echo -e "\nAsteroid has been installed in editable mode. Feel free to apply your changes !"