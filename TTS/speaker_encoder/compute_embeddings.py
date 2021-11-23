import argparse
import glob
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from TTS.bin.train_encoder import main
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


def plot(embeds, locations, plot_path, labels=None, title=None):
    import umap
    from bokeh.io import save, show
    from bokeh.models import (BoxZoomTool, ColumnDataSource, HoverTool,
                              OpenURL, ResetTool, TapTool)
    from bokeh.palettes import Category10
    from bokeh.plotting import figure
    from bokeh.transform import factor_cmap

    model = umap.UMAP()
    projection = model.fit_transform(embeds)
    if labels != None and labels != []:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    desc=locations,
                    label=labels
                )
            )
    else:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    desc=locations
                )
            )

    hover = HoverTool(
            tooltips=[
                ("file", "@desc"),
                ("speaker", "@label"),
            ]
        )

    # optionally consider adding these to the tooltips if you want additional detail
    # for the coordinates: ("(x,y)", "($x, $y)"),
    # for the index of the embedding / wav file: ("index", "$index"),

    if labels != None and labels != []:
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        pal = Category10[pal_size]

    p = figure(plot_width=600, plot_height=400, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])

    if labels != None and labels != []:
        p.circle('x', 'y',  source=source_wav_stems, color=factor_cmap('label', palette=pal, factors=factors),)
    else:
        p.circle('x', 'y',  source=source_wav_stems)

    # url = "http://localhost:8000/@desc"
    # taptool = p.select(type=TapTool)
    # taptool.callback = OpenURL(url=url)

    # show(p)
    
    if title: filename = f"plot_{title}.html"
    else: filename = "plot.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')

# if number == None: use all files
def compute(number=1000):

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
        idx = random.sample(range(len(labels)), number)
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
        plot([all_embedds[s_id][0] for s_id in sample_ids], [output_files[s_id] for s_id in sample_ids], args.plot_path, labels=labels, title=args.title)

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



if __name__ == '__main__':
    compute()

