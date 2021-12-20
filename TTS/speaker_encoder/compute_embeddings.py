import argparse
import glob
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config

import TTS.speaker_encoder.create_plots as create_plots
import TTS.speaker_encoder.attack_signatures as attack_signatures


# if number == None: use all files
def compute_embeddings(number=1000):

    parser = argparse.ArgumentParser(
        description='Compute embedding vectors for each wav file in a dataset. ')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).')
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Data path for wav files - directory or CSV file')
    parser.add_argument(
        'output_path',
        type=str,
        help='path for training outputs.')
    parser.add_argument(
        '--use_cuda', type=bool, help='flag to set cuda.', default=False
    )
    parser.add_argument(
        '--separator', type=str, help='Separator used in file if CSV is passed for data_path', default='|'
    )
    parser.add_argument(
        '--plot_path', type=str, help='Some of the generated embeddings will be plotted here', default=None
    )
    parser.add_argument(
        '--title', type=str, help='title of the resulting plot', default=None
    )
    parser.add_argument(
        '--silence_label', type=bool, help='Use the length of the silence as label', default=False
    )
    args = parser.parse_args()


    c = load_config(args.config_path)
    ap = AudioProcessor(**c['audio'])

    data_path = args.data_path
    split_ext = os.path.splitext(data_path)
    sep = args.separator
    labels = []

    if len(split_ext) > 0 and split_ext[1].lower() == '.csv':
        print(f'CSV file: {data_path}')
        if "youtube_dataset" in data_path:
            wav_path = "/opt/franzi/datasets/deepfake_datasets/youtube_dataset_malicious/"
            wav_files, labels = youtube_dataset(wav_path, data_path)

            print(f'{len(wav_files)} files found, {len(labels)} labels found')
            from collections import Counter
            print(f"all unique classes in labels: {Counter(labels).keys()}")
            print(f"frequency: {Counter(labels).values()}")

            labels = np.array(labels)
            smaller_classes_idx = np.where((labels == 'Youtube_Speaking of AI') | (labels == 'Youtube_Dessa') | (labels == 'Youtube_millennials')| (labels == 'Youtube_Will Kwan 2'))[0]
            rest_idx = np.where((labels != 'Youtube_Speaking of AI') & (labels != 'Youtube_Dessa') & (labels != 'Youtube_millennials') & (labels != 'Youtube_Will Kwan 2'))[0]
            print("smaller_classes_idx size", len(smaller_classes_idx))
            print("all rest_idx size", len(rest_idx))

            if number != None and len(smaller_classes_idx) < number:
                rest_idx = random.sample(list(rest_idx), number-len(smaller_classes_idx))
                print("picked rest_idx size", len(rest_idx))
                idx = list(rest_idx) + list(smaller_classes_idx)
                labels = labels[idx]
            elif number != None:
                idx = random.sample(list(smaller_classes_idx), number)
                labels = labels[idx]
            
            labels = list(labels)

    else:
        # Parse all wav/flac files in data_path
        wav_path = data_path
        wav_files = glob.glob(data_path + '/**/*.wav', recursive=True)
        if len(wav_files) == 0:
            wav_files = glob.glob(data_path + '/**/*.flac', recursive=True)

    assert len(wav_files) != 0, "No audio files found"

    output_files = [wav_file.replace(wav_path, args.output_path).replace(
        '.wav', '.npy').replace('.flac', '.npy') for wav_file in wav_files]

    for output_file in output_files:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model = SpeakerEncoder(**c.model)
    model.load_state_dict(torch.load(args.model_path)['model'])
    model.eval()
    if args.use_cuda:
        model.cuda()

    if number != None:
        idx = random.sample(range(len(wav_files)), number)
        wav_files = list(np.array(wav_files)[idx])
        if labels != []: 
            labels = list(np.array(labels)[idx])
            from collections import Counter
            print(f"picked unique classes in labels: {Counter(labels).keys()}")
            print(f"frequency: {Counter(labels).values()}")
            
            

    all_embedds = []
    for idx, wav_file in enumerate(tqdm(wav_files)):
        if not os.path.exists(output_files[idx]):
            mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate)).T
            mel_spec = torch.FloatTensor(mel_spec[None, :, :])
            if args.use_cuda:
                mel_spec = mel_spec.cuda()
            embedd = model.compute_embedding(mel_spec)
            embedd = embedd.detach().cpu().numpy()
            np.save(output_files[idx], embedd)
        else:
            embedd = np.load(output_files[idx])
        
        if args.silence_label:
            wav = ap.load_wav(wav_file, sr=ap.sample_rate)
            labels.append(_get_silence_label(wav))
        
        all_embedds.append(embedd)

    if args.plot_path != None:
        sample_ids = random.sample(range(len(all_embedds)), len(all_embedds))
        create_plots.plot_embeddings([all_embedds[s_id][0] for s_id in sample_ids], [output_files[s_id] for s_id in sample_ids], args.plot_path, labels=labels, title=args.title)

def _get_silence_label(wav):
    import librosa
    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    silence_length = librosa.get_duration(wav) - librosa.get_duration(trimmed)

    a=0.25
    b=0.5
    c=1.5
    d=2.5
    if silence_length <= a: return f"0-{a}"
    elif silence_length <= b: return f"{a}-{b}"
    elif silence_length <= c: return f"{b}-{c}"
    elif silence_length <= d: return f"{c}-{d}"
    elif silence_length > d: return f"{d}-inf"

def youtube_dataset(root_path, meta_file):
    """Normalize deepfake youtube dataset.
    https://github.com/franzi-19/deepfake_datasets
    """
    wav_files = []
    labels = []
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

                wav_files.append(filepath)
                labels.append("Youtube_" + attack_id)
    return wav_files, labels

#-----------------------

def start_assigning_labels():
    parser = argparse.ArgumentParser(
        description='Assign labels of unlabeled dataset based on knn')
    parser.add_argument(
        'model_path', type=str, help='Path to model outputs (checkpoint, tensorboard etc.).')
    parser.add_argument(
        'config_path', type=str, help='Path to config file.')
    parser.add_argument(
        'train_path', type=str, help='Train data path for wav files - directory')
    parser.add_argument(
        'train_output_path', type=str,  help='Path for training outputs.')
    parser.add_argument(
        'test_path', type=str, help='Test data path for wav files - directory') 
    parser.add_argument(
        'test_output_path', type=str,  help='Path for test outputs.')
    parser.add_argument(
        '--plot_path', type=str, help='Path to the result plot', default=None)
    parser.add_argument(
        '--title', type=str, help='title of the resulting plot', default=None)
    parser.add_argument(
        '--use_cuda', type=bool, help='flag to set cuda.', default=False
    )
    args = parser.parse_args()

    assign_labels_via_cosine_similarity(args.train_path, args.test_path, args.train_output_path, args.test_output_path, args.config_path, args.model_path, args.use_cuda, args.plot_path, args.title)

# TODO: train path only the real files used for training
def assign_labels_via_cosine_similarity(train_path, test_path, train_output_path, test_output_path, config_path, model_path, use_cuda, plot_path, title, size=1000):
    model, ap = _load_model(config_path, model_path, use_cuda)

    train_wav_files, train_output_files, train_labels = _get_files(train_path, train_output_path, size)
    test_wav_files, test_output_files, _ = _get_files(test_path, test_output_path, int(size*0.5))

    train_embedd = _create_embeddings(train_wav_files, train_output_files, ap, model, use_cuda)
    test_embedd = _create_embeddings(test_wav_files, test_output_files, ap, model, use_cuda)

    test_labels = _get_labels(train_embedd, train_labels, test_embedd)

    create_plots.plot_two_sets(train_embedd, train_labels, test_embedd, test_labels, train_wav_files, test_wav_files, plot_path, title)

def _load_model(config_path, model_path, use_cuda=True):
    c = load_config(config_path)
    ap = AudioProcessor(**c['audio'])
    model = SpeakerEncoder(**c.model)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    if use_cuda:
        model.cuda()

    return model, ap

# TODO use trim_silence model
def _get_files(folder_path, output_path, size):
    all_wav_files = []
    all_labels = []
    if "ASVspoof2019_LA_cm_protocols" in folder_path:
        label_files = glob.glob(folder_path + '/**/*.txt', recursive=True)
        wav_paths = ["/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_train/", "/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_dev/", "/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_eval/"]
        for label_file in label_files:
            print(f'Label file: {label_file}')
            if 'train' in label_file:
                wav_path = wav_paths[0]
            elif 'dev' in label_file:
                wav_path = wav_paths[1]
            elif 'eval' in label_file:
                wav_path = wav_paths[2]

            wav_files, labels = asvspoof_19(wav_path, label_file) 
            all_wav_files = all_wav_files + wav_files
            all_labels = all_labels + labels
            
            assert len(all_wav_files) == len(all_labels), "found different number of wav files and labels"
    else:
        all_wav_files = glob.glob(folder_path + '/**/*.wav', recursive=True)
        if len(all_wav_files) == 0:
            all_wav_files = glob.glob(folder_path + '/**/*.flac', recursive=True)

    assert len(all_wav_files) != 0, "No audio files found"

    output_files = [wav_file.replace(folder_path, output_path).replace(
        '.wav', '.npy').replace('.flac', '.npy') for wav_file in all_wav_files]

    for output_file in output_files:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    idx = random.sample(range(len(all_wav_files)), size)
    all_wav_files = list(np.array(all_wav_files)[idx])
    output_files = list(np.array(output_files)[idx])
    if all_labels != []: all_labels = list(np.array(all_labels)[idx])

    return all_wav_files, output_files, all_labels

def _create_embeddings(wav_files, output_files, ap, model, use_cuda):
    all_embedds = []
    for idx, wav_file in enumerate(tqdm(wav_files)):
        if not os.path.exists(output_files[idx]):
            mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate)).T
            mel_spec = torch.FloatTensor(mel_spec[None, :, :])
            if use_cuda:
                mel_spec = mel_spec.cuda()
            embedd = model.compute_embedding(mel_spec)
            embedd = embedd.detach().cpu().numpy()
            np.save(output_files[idx], embedd)
        else:
            embedd = np.load(output_files[idx])
        
        all_embedds.append(embedd[0])
    return all_embedds

# TODO check if get be done without for loop
def _get_labels(train, train_labels, test):
    return [_get_label_with_knn_for_one_sample(train, train_labels, [t]) for t in test]

def _get_label_with_knn_for_one_sample(train_, train_labels, test_):
    train = torch.from_numpy(np.array(train_))
    test = torch.from_numpy(np.array(test_))

    x = torch.randn(100, 10) 
    y = torch.randn(1, 10)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    dist = cos(train, test)

    knn = dist.topk(3)

    # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
    poss_label = [train_labels[num] for num in knn.indices.numpy()]

    return max(set(poss_label), key=poss_label.count)

def asvspoof_19(wav_folder, meta_file):
    """Normalize asvspoof19 dataset.
    :param wav_folder: path to dataset
    :param meta_file: Path from root_path to asvspoof info file (The .txt that has locations and info of every sample)
    """
    wav_files = []
    labels = []
    with open(meta_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            infos = line.split(' ')
            
            if len(infos) != 5:
                raise AssertionError('ASVspoof information file is malformed. Each line should have 5 columns.')

            _, file_name, _, attack_id, _ = infos

            wav_file = os.path.join(wav_folder, 'flac' , file_name + '.wav')
            if not os.path.exists(wav_file):
                wav_file = os.path.join(wav_folder, 'flac' , file_name + '.flac')
            assert os.path.exists(wav_file), f'Failure: File {wav_file} is missing'

            wav_files.append(wav_file)
            labels.append("asvspoof_19_" + attack_id)
    return wav_files, labels


# ---------------------
# TODO prevent AudioProcessor from removing silence
def label_based_on_signatures(train_path, train_output_path, config_path, model_path, use_cuda, plot_path, title, size=1000):
    model, ap = _load_model(config_path, model_path, use_cuda)
    train_wav_files, train_output_files, train_labels = _get_files(train_path, train_output_path, size)
    train_embedd = _create_embeddings(train_wav_files, train_output_files, ap, model, use_cuda)

    signatures, names = _get_signatures(train_wav_files)
    for sig, name in (zip(signatures, names)):
        create_plots.plot_embeddings_continuous(train_embedd, plot_path, sig, f"{title}_{name}")

def _get_signatures(wav_files):
    all_sig = []
    for wav in tqdm(wav_files):
        all_sig.append(attack_signatures.apply_all_signature_one_result(wav))
    result_lists = list(map(list, zip(*all_sig)))
    # result_lists = list(map(list, zip(*[attack_signatures.apply_all_signature_one_result(wav) for wav in wav_files])))

    return result_lists, attack_signatures.get_all_names()

# ---------
def plot_based_on_signatures(train_path, train_output_path, plot_path, title, size=1000):
    all_signature_names = attack_signatures.get_all_names_one_result()
    train_wav_files, _, train_labels = _get_files(train_path, train_output_path, size)
    
    for name in tqdm(all_signature_names):
        sig_function_1 = attack_signatures.get_signature_by_name(name)
        for other_name in all_signature_names:
            sig_function_2 = attack_signatures.get_signature_by_name(other_name)
            _plot_based_on_two_signatures(train_wav_files, train_labels, plot_path, title, sig_function_1, name, sig_function_2, other_name)

def _plot_based_on_two_signatures(train_wav_files, train_labels, plot_path, title, signature_1, signature_1_name, signature_2, signature_2_name):    
    train_embedd_x = []
    train_embedd_y = []
    for wav in train_wav_files:
        train_embedd_x.append(signature_1(wav))
        train_embedd_y.append(signature_2(wav))

    create_plots.just_plot(train_embedd_x, train_embedd_y, plot_path, train_labels, f"{title}_{signature_1_name}_{signature_2_name}")



if __name__ == '__main__':
    # compute_embeddings()                  # just for plotting embeddings
    # start_assigning_labels()              # for knn

    # label_based_on_signatures(  '/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_cm_protocols/',
    #                             '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2019_LA/',
    #                             '/home/franzi/masterarbeit/speaker_encoder_models/own_lstm_asvspoof/lstm_trim_silence-November-12-2021_02+43PM-debug/config.json',
    #                             '/home/franzi/masterarbeit/speaker_encoder_models/own_lstm_asvspoof/lstm_trim_silence-November-12-2021_02+43PM-debug/best_model.pth.tar',
    #                             True, '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2019_LA/plot/',
    #                             'signture_label')
    # label_based_on_signatures(  '/opt/franzi/datasets/ASVspoof2021_LA_eval',
    #                             '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2021_LA_eval/',
    #                             '/home/franzi/masterarbeit/speaker_encoder_models/own_lstm_asvspoof/lstm_trim_silence-November-12-2021_02+43PM-debug/config.json',
    #                             '/home/franzi/masterarbeit/speaker_encoder_models/own_lstm_asvspoof/lstm_trim_silence-November-12-2021_02+43PM-debug/best_model.pth.tar',
    #                             True, '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2021_LA_eval/plot/',
    #                             'signture_label')

    plot_based_on_signatures(   '/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_cm_protocols/',
                                '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2019_LA/',
                                '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2019_LA/plot/',
                                'signture_embedd')
    # plot_based_on_signatures(   '/opt/franzi/datasets/ASVspoof2021_LA_eval/',
    #                             '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2021_LA_eval/',
    #                             '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2021_LA_eval/plot/',
    #                             'signture_embedd')

