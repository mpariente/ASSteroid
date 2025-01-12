#!/bin/bash
# MIT Copyright (c) 2018 Kaituo XU


sphere_dir=tmp
wav_dir=tmp

. utils/parse_options.sh || exit 1;


echo "Download sph2pipe_v2.5 into egs/tools"
mkdir -p ../../tools
wget http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz -P ../../tools
cd ../../tools && tar -xzvf sph2pipe_v2.5.tar.gz && gcc -o sph2pipe_v2.5/sph2pipe sph2pipe_v2.5/*.c -lm && cd -

echo "Convert sphere format to wav format"
sph2pipe=../../tools/sph2pipe_v2.5/sph2pipe

if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

tmp=data/local/
mkdir -p $tmp

[ ! -f $tmp/sph.list ] && find $sphere_dir -iname '*.wv*' | grep -e 'si_tr_s' -e 'si_dt_05' -e 'si_et_05' > $tmp/sph.list

if [ ! -d $wav_dir ]; then
  while read line; do
    wav=`echo "$line" | sed "s:wv1:wav:g" | awk -v dir=$wav_dir -F'/' '{printf("%s/%s/%s/%s", dir, $(NF-2), $(NF-1), $NF)}'`
    echo $wav
    mkdir -p `dirname $wav`
    $sph2pipe -f wav $line > $wav
  done < $tmp/sph.list > $tmp/wav.list
else
  echo "Do you already get wav files? if not, please remove $wav_dir"
fi
