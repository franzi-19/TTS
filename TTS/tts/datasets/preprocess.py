import os
import random
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm
from TTS.tts.utils.generic_utils import split_dataset


def load_meta_data(datasets, dataset_folder, split_info):
    meta_data_train_all = []
    meta_data_test_all = []

    for dataset in datasets:
        name = dataset['name']
        root_path = Path(dataset_folder) / Path(dataset['path'])

        meta_file_train = dataset['meta_file_train'] if 'meta_file_train' in dataset else None
        meta_file_test = dataset['meta_file_test'] if 'meta_file_test' in dataset else None

        meta_data_train = []
        meta_data_test = []

        preprocessor = get_preprocessor_by_name(name)

        if meta_file_train is not None:
            meta_data_train = preprocessor(root_path, Path(dataset_folder) / Path(meta_file_train))
            
        if meta_file_test is not None:
            meta_data_test = preprocessor(root_path, Path(dataset_folder) / Path(meta_file_test))

        print(f" | > Using {len(meta_data_train)} files in {Path(root_path).resolve()} for training")
        print(f" | > Using {len(meta_data_test)} files in {Path(root_path).resolve()} for testing")

        meta_data_train_all += meta_data_train
        meta_data_test_all += meta_data_test

    meta_data_train_all, meta_data_test = split_train_set(split_info, meta_data_train_all)
    meta_data_test_all += meta_data_test

    return meta_data_train_all, meta_data_test_all


def get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())

def split_train_set(split_info, meta_data_train_all):
    meta_data_train = meta_data_train_all
    meta_data_test = []

    if split_info != None:
        assert split_info in ["no_split", "random_10_percent", "split_4_speaker"], f"split_train_data needs to be in {['no_split', 'random_10_percent', 'split_4_speaker']}"

        if split_info == "random_10_percent":
            idx_test = random.sample(range(len(meta_data_train_all)), int(len(meta_data_train_all)*0.1))
            test_selected = np.array(meta_data_train_all)[idx_test]
            train_selected = np.delete(np.array(meta_data_train_all), idx_test, axis=0)

            meta_data_test = list(test_selected)
            meta_data_train = list(train_selected)

        elif split_info == "split_4_speaker":
            selected_classes = random.sample(_get_classes(meta_data_train_all), 4)
            selected_items_idx = _get_all_id_from_classes(meta_data_train_all, selected_classes)

            test_selected = np.array(meta_data_train_all)[selected_items_idx]
            train_selected = np.delete(np.array(meta_data_train_all), selected_items_idx, axis=0)

            meta_data_test = list(test_selected)
            meta_data_train = list(train_selected)

        print(f" | > Train/Test split with {split_info}: Splitting all {len(meta_data_train_all)} found training data in {len(meta_data_train)} files for training and {len(meta_data_test)} files for testing")
            
    return meta_data_train, meta_data_test

def _get_classes(all):
    return list(set([_get_label(item) for item in all]))

def _get_label(item):
    _, _, label = item
    return label

def _get_all_id_from_classes(all, selected_classes):
    return [idx for idx, item in enumerate(all) if _get_label(item) in selected_classes]


############### dataset functions

def youtube_dataset(root_path, meta_file):
    """Normalize deepfake youtube dataset.
    https://github.com/franzi-19/deepfake_datasets
    """
    items = []
    with open(meta_file, "r") as file:
        next(file) # skip header
        for line in file:
            line = line.strip()
            infos = line.split(',')

            if len(infos) == 10:
                speaker, _, attack_id, url, _, _, _, _, _, _ = infos # speaker,language,channel,url,start,end,label,quality,topic
            elif len(infos) == 9:
                speaker, _, url, _, _, _, _, _, _ = infos # speaker,language,channel,url,start,end,label,quality,topic
                attack_id = 'benign'
            else:
                raise AssertionError('Metadata information file is malformed. Each line should have 9 or 10 columns.')

            wav_folder = os.path.join(root_path, speaker.replace(" ", "_"), url, 'wav/')
            assert os.path.exists(wav_folder), f'Failure: Folder {wav_folder} is missing'

            for filename in os.listdir(wav_folder):
                filepath = os.path.join(wav_folder, filename)
                assert os.path.exists(filepath), f'Failure: File {filepath} is missing'

                items.append(['', filepath,  "Youtube_" + attack_id])
    return items

def asvspoof_19(root_path, meta_file):
    """Normalize asvspoof19 dataset.
    :param root_path: path to dataset
    :param meta_file: Path from root_path to asvspoof info file (The .txt that has locations and info of every sample)
    """
    items = []
    with open(os.path.join(root_path, meta_file), 'r') as file:
        for line in file.readlines():
            line = line.strip()
            infos = line.split(' ')
            
            if len(infos) != 5:
                raise AssertionError('ASVspoof information file is malformed. Each line should have 5 columns.')

            _, file_name, _, attack_id, _ = infos

            wav_file = os.path.join(root_path, 'flac' , file_name + '.wav')
            if not os.path.exists(wav_file):
                wav_file = os.path.join(root_path, 'flac' , file_name + '.flac')
            assert os.path.exists(wav_file), f'Failure: File {wav_file} is missing'

            items.append(["", wav_file, "ASVSPOOF19_" + attack_id]) # text, wav_file_path, label
    return items

def asvspoof_21(root_path, meta_file):
    """Normalize asvspoof21 dataset.
    https://zenodo.org/record/4837263 or https://zenodo.org/record/4835108
    """
    items = []
    with open(os.path.join(root_path, meta_file), 'r') as file:
        for line in file.readlines():
            file_name = line.strip()

            wav_file = os.path.join(root_path, 'flac' , file_name + '.wav')
            if not os.path.exists(wav_file):
                wav_file = os.path.join(root_path, 'flac' , file_name + '.flac')

            # TODO: add assert back in
            # assert os.path.exists(wav_file), f'Failure: File {wav_file} is missing'
            if not os.path.exists(wav_file):
                continue

            items.append(["", wav_file, "ASVSPOOF21_unknown"]) 
    return items


def tweb(root_path, meta_file):
    """Normalize TWEB dataset.
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('\t')
            wav_file = os.path.join(root_path, cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


# def kusal(root_path, meta_file):
#     txt_file = os.path.join(root_path, meta_file)
#     texts = []
#     wavs = []
#     with open(txt_file, "r", encoding="utf8") as f:
#         frames = [
#             line.split('\t') for line in f
#             if line.split('\t')[0] in self.wav_files_dict.keys()
#         ]
#     # TODO: code the rest
#     return  {'text': texts, 'wavs': wavs}


def mozilla(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mozilla_de(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r', encoding="ISO 8859-1") as ttf:
        for line in ttf:
            cols = line.strip().split('|')
            wav_file = cols[0].strip()
            text = cols[1].strip()
            folder_name = f"BATCH_{wav_file.split('_')[0]}_FINAL"
            wav_file = os.path.join(root_path, folder_name, wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mailabs(root_path, meta_files=None):
    """Normalizes M-AI-Labs meta data files to TTS format"""
    speaker_regex = re.compile(
        "by_book/(male|female)/(?P<speaker_name>[^/]+)/")
    if meta_files is None:
        csv_files = glob(root_path + "/**/metadata.csv", recursive=True)
    else:
        csv_files = meta_files
    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        txt_file = os.path.join(root_path, csv_file)
        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        print(" | > {}".format(csv_file))
        with open(txt_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                if meta_files is None:
                    wav_file = os.path.join(folder, 'wavs', cols[0] + '.wav')
                else:
                    wav_file = os.path.join(root_path,
                                            folder.replace("metadata.csv", ""),
                                            'wavs', cols[0] + '.wav')
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append([text, wav_file, speaker_name])
                else:
                    raise RuntimeError("> File %s does not exist!" %
                                       (wav_file))
    return items


def ljspeech(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


def nancy(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1:line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append([text, wav_file, speaker_name])
    return items


def common_voice(root_path, meta_file):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            wav_file = os.path.join(root_path, "clips", cols[1].replace(".mp3", ".wav"))
            items.append([text, wav_file, 'MCV_' + speaker_name])
    return items


def libri_tts(root_path, meta_files=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if meta_files is None:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split('.')[0]
        speaker_name = _meta_file.split('_')[0]
        chapter_id = _meta_file.split('_')[1]
        _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
        with open(meta_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('\t')
                wav_file = os.path.join(_root_path, cols[0] + '.wav')
                text = cols[1]
                items.append([text, wav_file, 'LTTS_' + speaker_name])
    for item in items:
        assert os.path.exists(
            item[1]), f" [!] wav files don't exist - {item[1]}"
    return items


def custom_turkish(root_path, meta_file):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "turkish-female"
    skipped_files = []
    with open(txt_file, 'r', encoding='utf-8') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs',
                                    cols[0].strip() + '.wav')
            if not os.path.exists(wav_file):
                skipped_files.append(wav_file)
                continue
            text = cols[1].strip()
            items.append([text, wav_file, speaker_name])
    print(f" [!] {len(skipped_files)} files skipped. They don't exist...")
    return items


# ToDo: add the dataset link when the dataset is released publicly
def brspeech(root_path, meta_file):
    '''BRSpeech 3.0 beta'''
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("wav_filename"):
                continue
            cols = line.split('|')
            #print(cols)
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]
            speaker_name = cols[3]
            items.append([text, wav_file, speaker_name])
    return items

def vctk(root_path, meta_files=None, wavs_path='wav48'):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    test_speakers = meta_files
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file,
                                                  root_path).split(os.sep)
        file_id = txt_file.split('.')[0]
        if isinstance(test_speakers,
                      list):  # if is list ignore this speakers ids
            if speaker_id in test_speakers:
                continue
        with open(meta_file) as file_text:
            text = file_text.readlines()[0]
        wav_file = os.path.join(root_path, wavs_path, speaker_id,
                                file_id + '.wav')
        items.append([text, wav_file, 'VCTK_' + speaker_id])

    return items


def vctk_slim(root_path, meta_files=None, wavs_path='wav48'):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    items = []
    txt_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for text_file in txt_files:
        _, speaker_id, txt_file = os.path.relpath(text_file,
                                                  root_path).split(os.sep)
        file_id = txt_file.split('.')[0]
        if isinstance(meta_files, list):  # if is list ignore this speakers ids
            if speaker_id in meta_files:
                continue
        wav_file = os.path.join(root_path, wavs_path, speaker_id,
                                file_id + '.wav')
        items.append([None, wav_file, 'VCTK_' + speaker_id])

    return items

# ======================================== VOX CELEB ===========================================
def voxceleb2(root_path, meta_file=None):
    """
    :param meta_file   Used only for consistency with load_meta_data api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="2")


def voxceleb1(root_path, meta_file=None):
    """
    :param meta_file   Used only for consistency with load_meta_data api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="1")


def _voxcel_x(root_path, meta_file, voxcel_idx):
    assert voxcel_idx in ["1", "2"]
    expected_count = 148_000 if voxcel_idx == "1" else 1_000_000
    voxceleb_path = Path(root_path)
    cache_to = voxceleb_path / f"metafile_voxceleb{voxcel_idx}.csv"
    cache_to.parent.mkdir(exist_ok=True)

    # if not exists meta file, crawl recursively for 'wav' files
    if meta_file is not None:
        with open(str(meta_file), 'r') as f:
            return [x.strip().split('|') for x in f.readlines()]

    elif not cache_to.exists():
        cnt = 0
        meta_data = ""
        wav_files = voxceleb_path.rglob("**/*.wav")
        for path in tqdm(wav_files, desc=f"Building VoxCeleb {voxcel_idx} Meta file ... this needs to be done only once.",
                         total=expected_count):
            speaker_id = str(Path(path).parent.parent.stem)
            assert speaker_id.startswith('id')
            text = None  # VoxCel does not provide transciptions, and they are not needed for training the SE
            meta_data += f"{text}|{path}|voxcel{voxcel_idx}_{speaker_id}\n"
            cnt += 1
        with open(str(cache_to), 'w') as f:
            f.write(meta_data)
        if cnt < expected_count:
            raise ValueError(f"Found too few instances for Voxceleb. Should be around {expected_count}, is: {cnt}")

    with open(str(cache_to), 'r') as f:
        return [x.strip().split('|') for x in f.readlines()]
