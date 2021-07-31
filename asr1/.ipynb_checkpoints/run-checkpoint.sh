#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


exptype=

# general configuration
backend=pytorch
stage=0      # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=5         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume= # Resume the training from snapshot
do_delta=false

preprocess_config=conf/specaug.yaml #uncomment if using specaugment
train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_pytorch_transformer_large.yaml

# rnnlm related
use_lm=true
train_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=''            # tag for managing LMs
perturb=true

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10 # use 1 for RNN models

# exp tag
tag="" # tag for managing experiments.#specaugment_v1_nospeed

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=
train_dev=
test_set=

char_type="char"

data_dir_kz='/home/datasets/ISSAI_KSC_335RS_v3/'
data_dir_ru='/home/datasets/ISSAI_OpenSTT_CS334/'
data_dir_en='/home/datasets/ISSAI_CV_330/'

if [[ $exptype == 'mono_kz' ]]; then
 echo "running monolingual kz experiment"
 train_set=train_kz
 train_dev=dev_kz
 test_set=test_kz
elif [[ $exptype == 'mono_ru' ]]; then
 echo "running monolingual ru experiment"
 train_set=train_ru
 train_dev=dev_ru
 test_set="test_youtube test_books"
elif [[ $exptype == 'mono_en' ]]; then
 echo "running monolingual en experiment"
 train_set=train_en
 train_dev=dev_en
 test_set="test_en test_sf" 
elif [[ $exptype == 'mlt_independent' ]] || [[ $exptype == 'mlt_combined' ]]; then
 echo "running multilingual experiment"
 train_set="train_kz train_ru train_en"
 train_dev="dev_kz dev_ru dev_en"
 test_set="test_youtube test_books test_kz test_en test_sf" 
else
 echo "Please select the experiment type"
 exit 1;
fi

if [[ $exptype == 'mlt_independent' ]]; then
 char_type="phn"
fi

# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Setting up directories"
  
  arg_opts="--char_type ${char_type}"
  mkdir -p data
  for x in ${train_set} ${train_dev} ${test_set}; do
    mkdir -p data/$x
  done
   if [ -d $data_dir_kz ]; then
    arg_opts="--data_dir_kz ${data_dir_kz} ${arg_opts}"
   fi

   if [ -d $data_dir_ru ]; then
    arg_opts="--data_dir_ru ${data_dir_ru} ${arg_opts}"
   fi

   if [ -d $data_dir_en ]; then
    arg_opts="--data_dir_en ${data_dir_en} ${arg_opts}"
   fi
  echo "preparing data"
  local/data_prep.py $arg_opts
fi

if [[ $exptype == 'mlt_independent' ]] || [[ $exptype == 'mlt_combined' ]]; then
  train_set=train
  train_dev=dev
fi

lmexpname=${train_set}_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  if [[ $exptype == 'mlt_independent' ]] || [[ $exptype == 'mlt_combined' ]]; then
    for x in ${train_set} ${train_dev}; do
      mkdir -p data/$x
    done
    for x in 'wav.scp' 'utt2spk' 'text'; do
          for y in ${train_set} ${train_dev}; do
              cat data/${y}_kz/${x} data/${y}_ru/${x} data/${y}_en/${x} | sort -s -k1,1 > data/${y}/${x}; 
          done
    done
  fi

  for x in ${train_set} ${train_dev} ${test_set}; do
    utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
fi

if $perturb; then
    rm -r data/${train_set}_sp
    cp data/${train_set} data/${train_set}_sp -r
    train_set=${train_set}_sp
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
for x in ${test_set}; do
    feat_recog_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_recog_dir}
done

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Feature extraction"
  
  if $perturb; then
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
  fi
  
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
  for x in ${train_set} ${train_dev} ${test_set}; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
          data/${x} exp/make_fbank/${x} ${fbankdir}
      utils/fix_data_dir.sh data/${x}
  done

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
  utils/fix_data_dir.sh data/${train_set}

  exp_name=$(basename $PWD)

  dump.sh --cmd "$train_cmd" --nj 20 --do_delta ${do_delta} \
      data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
      data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
  for x in ${test_set}; do  
    feat_recog_dir=${dumpdir}/${x}/delta${do_delta};
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/${x}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${x} ${feat_recog_dir}
  done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    #cut -d " " -f 2- data/train/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    #cat " " > ${nlsyms}
    touch ${nlsyms}

    echo "make a dictionary"

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text -t ${char_type} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | grep -v '<unk>' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    echo "make json files"

    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} --trans_type ${char_type}  \
     data/${train_set} ${dict} > ${feat_tr_dir}/data.json

    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} --trans_type ${char_type}  \
     data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    
    
    for x in ${test_set}; do  
      feat_recog_dir=${dumpdir}/${x}/delta${do_delta};
      data2json.sh --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} --trans_type ${char_type}  \
       data/${x} ${dict} > ${feat_recog_dir}/data.json
    done



fi


if ${use_lm} && ${train_lm}; then
  lm_train_set=data/local/train.txt
  lm_valid_set=data/local/dev.txt
  
  echo "Preparing LM data"
  
  mkdir -p data/local/
  
  text2token.py --nchar 1 \
                --space "<space>" \
                --trans_type ${char_type} \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_set}/text) \
                > ${lm_train_set}

  text2token.py --nchar 1 \
                --space "<space>" \
                --trans_type ${char_type} \
                --non-lang-syms data/lang_1char/non_lang_syms.txt \
                <(cut -d' ' -f2- data/${train_dev}/text) \
                > ${lm_valid_set}

  ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
          lm_train.py \
          --config ${lm_config} \
          --ngpu ${ngpu} \
          --backend ${backend} \
          --verbose 1 \
          --outdir ${lmexpdir} \
          --tensorboard-dir tensorboard/${lmexpname} \
          --train-label ${lm_train_set} \
          --valid-label ${lm_valid_set} \
          --resume ${lm_resume} \
          --dict ${dict}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi
    extra_opts=""
    if ${use_lm}; then
      extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    fi

    pids=() # initialize pids
    for rtask in ${test_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if ${use_lm}; then
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${extra_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
