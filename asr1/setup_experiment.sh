#!/bin/bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./setup_experiment.sh <expname>"
  exit 1;
fi

expname=$1
cd ..
mkdir ${expname}
cd ${expname}

cp ../asr1/{cmd,path,run,recog_wav}.sh .
cp -P ../asr1/steps .
cp -P ../asr1/utils .
ln -s ../asr1/local .
ln -s ../asr1/conf .
