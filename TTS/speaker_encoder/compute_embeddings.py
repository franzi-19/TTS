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
    if labels != None:
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

    if labels != None:
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        pal = Category10[pal_size]

    p = figure(plot_width=600, plot_height=400, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])

    if labels != None:
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
    args = parser.parse_args()


    c = load_config(args.config_path)
    ap = AudioProcessor(**c['audio'])

    data_path = args.data_path
    split_ext = os.path.splitext(data_path)
    sep = args.separator

    if len(split_ext) > 0 and split_ext[1].lower() == '.csv':
        # Parse CSV
        print(f'CSV file: {data_path}')
        with open(data_path) as f:
            wav_path = os.path.join(os.path.dirname(data_path), 'wavs')
            wav_files = []
            print(f'Separator is: {sep}')
            for line in f:
                components = line.split(sep)
                if len(components) != 2:
                    print("Invalid line")
                    continue
                wav_file = os.path.join(wav_path, components[0] + '.wav')
                #print(f'wav_file: {wav_file}')
                if os.path.exists(wav_file):
                    wav_files.append(wav_file)
        print(f'Count of wavs imported: {len(wav_files)}')
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
        wav_files = random.sample(wav_files, number)

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
        
        all_embedds.append(embedd)

    if args.plot_path != None:
        sample_ids = random.sample(range(len(all_embedds)), len(all_embedds))
        plot([all_embedds[s_id][0] for s_id in sample_ids], [output_files[s_id] for s_id in sample_ids], args.plot_path, title=args.title)


if __name__ == '__main__':
    compute()

