# MultilingualASR


This repository provides the recipe for the paper [A Study of Multilingual End-to-End Speech Recognition for Kazakh, Russian, and English](https://arxiv.org/abs/2108.01280).



## Setup and Requirements 

Our code builds upon [ESPnet](https://github.com/espnet/espnet), and requires prior installation of the framework. Please follow the [installation guide](https://espnet.github.io/espnet/installation.html) and put the repository inside `espnet/egs/` directory.

After succesfull installation of ESPnet & Kaldi, go to `MultilingualASR/asr1` directory and create links to the dependencies:
```
ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils
```
The directory for running the experiments (`MultilingualASR/<exp-name`) can be created by running the following script:

```
./setup_experiment.sh <exp-name>
```

## Downloading the dataset
 
Download [KSC](https://issai.nu.edu.kz/kz-speech-corpus/) and [OpenSTT & CV datasets](https://issai.nu.edu.kz/multilingual-asr/) and untar in the directory of your choice. Specify the path to the data in  `./run.sh` script:
```
data_dir_kz='path-to-KSC'
data_dir_ru='path-to-OpenSTT'
data_dir_en='-path-to-CV'
```
## Training

To train the models, run the script `./run.sh` inside `MultilingualASR/<exp-name>/` folder. Specify the experiment type you would like to run:
```
./run.sh --exptype "experiment type" --stage 0 --stop_stage 0
```
**experiment type**: "mono_kz", "mono_ru", "mono_en", "mlt_independent", "mlt_combined".

**Note:** We suggest to run the experiments one step at a time to avoid errors. 
During decoding, change `beam-size:30` in conf/decode_pytorch_transformer_large.yaml for monolingual experiments. 

## Pre-trained models

| Model | Large Transformer |  Large Transformer with Speed Perturbation (SP) |  Large Transformer with SP and SpecAugment|
| --- | --- | --- | --- |
| monolingual kazakh | [mono_kz_transformer_large.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_kz_transformer_large.tar.gz) | [mono_kz_transformer_large_sp.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_kz_transformer_large_sp.tar.gz) | [mono_kz_transformer_large_sp_specaug.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_kz_transformer_large_sp_specaug.tar.gz) |
| monolingual russian | [mono_ru_transformer_large.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_ru_transformer_large.tar.gz) | [mono_ru_transformer_large_sp.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_ru_transformer_large_sp.tar.gz) | [mono_ru_transformer_large_sp_specaug.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_ru_transformer_large_sp_specaug.tar.gz) |
| monolingual english | [mono_en_transformer_large.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_en_transformer_large.tar.gz) | [mono_en_transformer_large_sp.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_en_transformer_large_sp.tar.gz) | [mono_en_transformer_large_sp_specaug.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/mono_en_transformer_large_sp_specaug.tar.gz) |
| multilingual combined | [multi_combined_transformer_large.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_combined_transformer_large.tar.gz) | [multi_combined_transformer_large_sp.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_combined_transformer_large_sp.tar.gz) | [multi_combined_transformer_large_sp_specaug.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_combined_transformer_large_sp_specaug.tar.gz) |
| multilingual independent | [multi_independent_transformer_large.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_independent_transformer_large.tar.gz) | [multi_independent_transformer_large_sp.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_independent_transformer_large_sp.tar.gz) | [multi_independent_transformer_large_sp_specaug.tar.gz](https://issai.nu.edu.kz/wp-content/uploads/2021/07/multi_independent_transformer_large_sp_specaug.tar.gz) |



## Inference
To decode a single audio, specify paths to the following files inside `recog_wav.sh` script:
```
lang_model= path to rnnlm.model.best
cmvn= path to cmvn.ark, for example data/train/cmvn.ark
recog_model= path to e2e model, in case of large transformer: model.last10.avg.best 
```
Then, run the following script:
```
./recog_wav.sh <path-to-audio-file>
```

```python

```
