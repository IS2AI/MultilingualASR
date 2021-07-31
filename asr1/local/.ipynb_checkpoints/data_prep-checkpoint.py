#!/usr/bin/env python
import sys, argparse, re, os, random, glob
import pandas as pd
from pathlib import Path
import wave
import contextlib

seed=1234

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--char_type", help="char or phn(for independent)")
    parser.add_argument("--data_dir_kz", help="path to ISSAI_KSC")
    parser.add_argument("--data_dir_ru", help="path to ISSAI_OpenSTT")
    parser.add_argument("--data_dir_en", help="path to ISSAI_CV")
    print(' '.join(sys.argv))
    args = parser.parse_args()
    return args


def read_meta(path):
    meta = pd.read_csv(path, sep=" ") 
    return list(meta['uttID'])


def get_duration(file_path):
    duration = None
    if os.path.exists(file_path) and Path(file_path).stat().st_size > 0:
        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            if frames>0:
                rate = f.getframerate()
                duration = frames / float(rate)
    return duration if duration else 0

def get_text(dataset_dir, file):
    txt_file = os.path.join(dataset_dir, file + '.txt') 
    with open(txt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

# +
def prepare_ksc(dataset_dir, files):
    file2data = {}
    for filename in files:
        with open(os.path.join(dataset_dir, 'Transcriptions', filename+'.txt'), 'r', encoding='utf-8') as f:
            text = f.read().strip()
        wav_path = os.path.join(dataset_dir, 'Audios', filename+'.wav')
        file2data[filename] = (wav_path, text)
    return file2data  

def prepare_others(dataset_dir, data_source):
    file2data = {}
    files = glob.glob(os.path.join(dataset_dir, data_source) + '/*.wav')
    for wav_path in files:
        filename = os.path.basename(wav_path).replace('.wav', '')
        txt_path = wav_path.replace('.wav', '.txt')
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        file2data[filename] = (wav_path, text)
    return file2data


# -

def append_lang(text, lang):
    new_text = ""
    for char in text: 
        if char != ' ': char += "_" + lang + ' '
        else: char = "sil "
        new_text += char
    return new_text.rstrip()


def create_data(data_files, data_source, char_type, lang):
    wav_format = '-r 16000 -c 1 -b 16 -t wav - downsample |'
    files = sorted(data_files.keys())
    write_path = 'data/' + data_source
    total_duration = 0
    with open(write_path + '/text', 'w', encoding="utf-8") as f1, \
    open(write_path + '/utt2spk', 'w', encoding="utf-8") as f2, \
    open(write_path + '/wav.scp', 'w', encoding="utf-8") as f3:
        for filename in files:
            wav_path, transcription = data_files[filename]
            total_duration += get_duration(wav_path) 
            if char_type=='phn': transcription = append_lang(transcription, lang)
            f1.write(filename + ' ' + transcription + '\n')
            f2.write(filename + ' ' + filename + '\n')
            f3.write(filename + ' sox ' + wav_path  + ' ' + wav_format +  '\n') 
        print(data_source + " duration:", total_duration / 3600)


def main():
    args = get_args()
    char_type = args.char_type

    if os.path.exists(args.data_dir_kz):
        data_dir_kz = args.data_dir_kz
        print('preparing KSC-335')
        for metas in [('train.csv', 'train_kz'), ('dev.csv', 'dev_kz'), ('test.csv', 'test_kz')]:
            meta_csv, meta_filename = metas
            if os.path.exists('data/' + meta_filename):
                meta_file = read_meta(os.path.join(data_dir_kz, 'Meta/', meta_csv))
                data = prepare_ksc(data_dir_kz, meta_file) 
                create_data(data, meta_filename, char_type, 'kz')
    else: "path to ISSAI_KSC does not exist"
                
    if os.path.exists(args.data_dir_en):
        data_dir_en = args.data_dir_en
        print('preparing CV-330')
        for metas in [('train','train_en'), ('dev','dev_en'), ('test','test_en'),('test_sf','test_sf')]:
            meta_dir, meta_filename = metas
            if os.path.exists('data/' + meta_filename):
                data = prepare_others(data_dir_en, meta_dir)
                create_data(data, meta_filename, char_type, 'en')
    else: "path to ISSAI_CV does not exist"    
        
    if os.path.exists(args.data_dir_ru):
        data_dir_ru = args.data_dir_ru
        print('preparing OpenSTT-334')
        for metas in [('train','train_ru'),('dev','dev_ru'),('test_youtube','test_youtube'),('test_books','test_books')]:
            meta_dir, meta_filename = metas
            if os.path.exists('data/' + meta_filename):
                data = prepare_others(data_dir_ru, meta_dir)
                create_data(data, meta_filename, char_type, 'ru')
    else: "path to ISSAI_OpenSTT does not exist"
    


if __name__ == "__main__":
    main()
