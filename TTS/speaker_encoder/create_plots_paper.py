import statistics

import numpy as np
import pandas as pd
import TTS.speaker_encoder.attack_signatures as attack_signatures
import TTS.speaker_encoder.compute_embeddings as ce
import TTS.speaker_encoder.create_plots as create_plots
from tqdm import tqdm

size = None # = None to use all wav files

# 4.3
def plot_asv19_attack_signatures(asv19_path, asv19_output_path, plot_path):
    asv19_wav_files, asv19_output_files, asv19_labels, gender = ce._get_files(asv19_path, asv19_output_path, size)

    all_signature_names = attack_signatures.get_all_names_one_result()
    all_signature_names.append('gender')
    gender = ce._asssign_gender_id(gender)

    embeds = []
    table = {}
    for name in tqdm(all_signature_names):
        sig_function = attack_signatures.get_signature_by_name(name)
        embed = []
        for idx, wav in enumerate(asv19_wav_files):
            if name == 'gender': 
                embed.append(gender[idx])
            else: 
                embed.append(sig_function(wav))
        
        embed = ce.normalize_points(embed) # TODO move to the end
        embeds.append(embed)
        centroid, mean_distance = _calculate_centroid_and_distance(plot_path + 'asv19_attack_signatures_table.tex', embed, asv19_labels)
        table[name] = [centroid, mean_distance]

    _table_to_latex(table, ['attack signatures', 'centroid', 'mean distance'], plot_path + 'asv19_attack_signatures_table.tex')
    create_plots.plot_embeddings(np.transpose(embeds), None, plot_path, asv19_labels, filename='asv19_attack_signatures_plot.png')

def _calculate_centroid_and_distance(latex_filepath, embeds, labels):
    cluster = {key: [] for key in labels}
    for idx, point in enumerate(embeds):
        cluster[labels[idx]].append(point)

    centroids, mean_distances = [], []
    for cl in cluster:
        centroid, mean_distance = ce.calculate_mean_distance(cluster[cl])
        centroids.append(centroid)
        mean_distances.append(mean_distance)

    return statistics.mean(centroids), statistics.mean(mean_distances)


def _table_to_latex(input, header, latex_filepath):
    df = pd.DataFrame(data=input)
    df.columns = [v.replace('_', ' ') for v in list(df)] # make the attack IDs look nice
    df = df.T # attack ID -> row names
    df.columns = header
    open(latex_filepath, 'a').close()
    df.to_latex(latex_filepath)

# 4.4.1
def plot_random_split(config_path, model_path, use_cuda, asv19_path, asv19_output_path): # idea: load model, get the file id for the used test split from a during training created save file -> load them and proceed as normal
    model, ap = ce._load_model(config_path, model_path, use_cuda)
    # TODO

# 4.4.1
def plot_4_speaker_split():
    NotImplemented

# 4.4.2
def plot_asv19_asv21(config_path, model_path, use_cuda, asv19_path, asv19_output_path, asv21_path, asv21_output_path, plot_path, title):
    model, ap = ce._load_model(config_path, model_path, use_cuda)

    asv19_wav_files, asv19_output_files, asv19_labels, _ = ce._get_files(asv19_path, asv19_output_path, size)
    asv21_wav_files, asv21_output_files, _, _ = ce._get_files(asv21_path, asv21_output_path, int(size*0.5))

    asv19_embedd = ce._create_embeddings(asv19_wav_files, asv19_output_files, ap, model, use_cuda)
    asv21_embedd = ce._create_embeddings(asv21_wav_files, asv21_output_files, ap, model, use_cuda)

    asv21_label = ['unknown']* len(asv21_embedd)

    create_plots.plot_two_sets(asv19_embedd, asv19_labels, asv21_embedd, asv21_label, asv19_wav_files, asv21_wav_files, plot_path, title)

# 4.4.2
def calculate_table_asv21_sig_metric(config_path, model_path, use_cuda, asv21_path, asv21_output_path, k=5):
    model, ap = ce._load_model(config_path, model_path, use_cuda)

    asv21_wav_files, asv21_output_files, _, gender = ce._get_files(asv21_path, asv21_output_path, size)
    asv21_embedd = ce._create_embeddings(asv21_wav_files, asv21_output_files, ap, model, use_cuda)

    signature_labels, names = ce._get_signature_labels(asv21_wav_files, gender)
    result_dict = {}
    for sig_labels, name in zip(signature_labels, names):
        labels = ce.normalize_points(sig_labels)
        coll_min, coll_max = [], []
        for idx, current in enumerate(asv21_embedd):
            found_min, found_max = ce.calculate_range(current, labels[idx], asv21_embedd, labels, k)
            coll_min.append(found_min)
            coll_max.append(found_max)
        result_dict[name] = (sum(coll_min)/len(coll_min), sum(coll_max)/len(coll_max))

if __name__ == '__main__':
    plot_asv19_attack_signatures(   '/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_cm_protocols/',
                                    '/home/franzi/masterarbeit/embeddings/lstm_November-12-2021_trim_silence/ASVspoof2019_LA/',
                                    '/home/franzi/masterarbeit/TTS/TTS/speaker_encoder/plots_paper/')
