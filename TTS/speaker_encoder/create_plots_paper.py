import csv

import numpy as np
import pandas as pd
import TTS.speaker_encoder.attack_signatures as attack_signatures
import TTS.speaker_encoder.compute_embeddings as ce
import TTS.speaker_encoder.create_plots as create_plots
from tqdm import tqdm

SIZE = 10000 # = None to use all wav files

model = 'own_lstm_asvspoof/lstm_trim_silence-November-12-2021_02+43PM-debug/'
random_split_model = 'own_lstm_asvspoof/asv19_random_10_percent-January-31-2022_01+21PM-debug/'
random_split_csv = 'random_10_percent_test_filenames.csv'
speaker_split_model = 'own_lstm_asvspoof/asv19_split_4_speaker_with_csv-February-02-2022_05+19PM-debug/'
speaker_split_csv = 'split_4_speaker_test_filenames.csv'

base_model_path = '/home/franzi/masterarbeit/speaker_encoder_models/'
base_embedding_path = '/home/franzi/masterarbeit/embeddings/'
model_name = 'best_model.pth.tar'
config_name = 'config.json'

asv19_name = 'ASVspoof2019_LA/'
asv21_name = 'ASVspoof2021_LA_eval/'

MODEL_PATH = base_model_path + model + model_name
MODEL_CONFIG = base_model_path + model + config_name

RANDOM_SPLIT_MODEL_PATH = base_model_path + random_split_model + model_name
RANDOM_SPLIT_MODEL_CONFIG = base_model_path + random_split_model + config_name
RANDOM_SPLIT_CSV = base_model_path + random_split_model + random_split_csv

SPEAKER_SPLIT_MODEL_PATH = base_model_path + speaker_split_model + model_name
SPEAKER_SPLIT_MODEL_CONFIG = base_model_path + speaker_split_model + config_name
SPEAKER_SPLIT_CSV = base_model_path + speaker_split_model + speaker_split_csv

ASV19_PATH = '/opt/franzi/datasets/DS/LA/ASVspoof2019_LA_cm_protocols/'
ASV19_OUTPUT_PATH_MODEL = base_embedding_path + model + asv19_name
ASV19_OUTPUT_PATH_RANDOM_SPLIT = base_embedding_path + random_split_model + asv19_name
ASV19_OUTPUT_PATH_SPEAKER_SPLIT = base_embedding_path + speaker_split_model + asv19_name
ASV21_PATH = '/opt/franzi/datasets/ASVspoof2021_LA_eval'
ASV21_OUTPUT_PATH_MODEL = base_embedding_path + model + asv21_name

USE_CUDA = True
PLOT_PATH = '/home/franzi/masterarbeit/TTS/TTS/speaker_encoder/plots_paper/'


# 4.3
# TODO: embedd contains NAN
def plot_asv19_attack_signatures(): # 36 hours for all asv19
    asv19_wav_files, _, asv19_labels, gender = ce._get_files(ASV19_PATH, ASV19_OUTPUT_PATH_MODEL, SIZE)

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
        
        embed = ce.normalize_points(embed)
        embeds.append(embed)
        centroid, mean_distance = _calculate_centroid_and_distance(embed, asv19_labels)
        table[name] = [centroid, mean_distance]

    _table_to_latex(table, ['centroid', 'mean distance'], PLOT_PATH + 'asv19_attack_signatures_table.tex')
    create_plots.plot_embeddings(np.transpose(embeds), PLOT_PATH, asv19_labels, filename='asv19_attack_signatures_plot.png')

# mdcc/variance
def _calculate_centroid_and_distance(embeds, labels):
    cluster = {key: [] for key in labels}
    for idx, point in enumerate(embeds):
        cluster[labels[idx]].append(point)

    centroids, mean_distances = [], []
    for cl in cluster:
        centroid, mean_distance = ce.calculate_mean_distance(cluster[cl])
        centroids.append(centroid)
        mean_distances.append(mean_distance)

    return np.nanmean(centroids), np.nanmean(mean_distances)

def _table_to_latex(input, header, latex_filepath):
    df = pd.DataFrame(data=input)
    df.columns = [v.replace('_', ' ') for v in list(df)]
    df.sort_index(axis=1, inplace=True) 
    df = df.applymap(lambda x: round(x,2))
    df = df.T # signature -> row names
    df.columns = header

    avg = df.mean(axis=0)
    df = df.append(pd.DataFrame([[round(avg[header[0]],2),round(avg[header[1]],2)]], columns=header, index=['average']))

    open(latex_filepath, 'a').close()
    df.to_latex(latex_filepath)
    print(f'saved to {latex_filepath}')

# 4.4.1
def plot_split(model_config, model_path, filenames_csv_path, filename, asv19_output_path):
    model, ap = ce._load_model(model_config, model_path, USE_CUDA)
    
    test_wav_files = _get_files_from_csv(filenames_csv_path)

    asv19_wav_files, asv19_output_files, asv19_labels, _ = ce._get_files(ASV19_PATH, asv19_output_path, None)
    test_output_files, test_labels = _match_with_whole_set(test_wav_files, asv19_wav_files, asv19_output_files, asv19_labels)

    test_embeds = ce._create_embeddings(test_wav_files, test_output_files, ap, model, USE_CUDA)

    test_embeds = ce.normalize_points(test_embeds)
    centroid, mean_distance = _calculate_centroid_and_distance(test_embeds, test_labels)
    _write_to_csv(PLOT_PATH + filename + '.csv', ['centroid', 'mean_distance'], [centroid, mean_distance])

    create_plots.plot_embeddings(test_embeds, PLOT_PATH, test_labels, filename=filename+'.png')

def _get_files_from_csv(csv_file):
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f, delimiter=",")
        return next(reader) 

def _match_with_whole_set(wav_files, set_wav_files, set_output_files, set_labels):
    labels, output_files = [], []
    for wav in wav_files:
        idx = set_wav_files.index(wav)
        output_files.append(set_output_files[idx])
        labels.append(set_labels[idx])

    print(f'found labels: {list(set(labels))}')
    return output_files, labels

def _write_to_csv(csv_file_path, header, content):
    with open(csv_file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow(content)

# 4.4.2
def plot_asv19_asv21():
    model, ap = ce._load_model(MODEL_CONFIG, MODEL_PATH, USE_CUDA)

    asv19_wav_files, asv19_output_files, asv19_labels, _ = ce._get_files(ASV19_PATH, ASV19_OUTPUT_PATH_MODEL, SIZE)
    asv21_wav_files, asv21_output_files, _, _ = ce._get_files(ASV21_PATH, ASV21_OUTPUT_PATH_MODEL, SIZE)

    asv19_embedd = ce._create_embeddings(asv19_wav_files, asv19_output_files, ap, model, USE_CUDA)
    asv21_embedd = ce._create_embeddings(asv21_wav_files, asv21_output_files, ap, model, USE_CUDA)

    asv21_label = ['unknown']* len(asv21_embedd)

    create_plots.plot_two_sets(asv19_embedd, asv19_labels, asv21_embedd, asv21_label, PLOT_PATH, 'asv19_asv21.png')

# 4.4.2
def calculate_table_asv21_sig_metric(k=5):
    model, ap = ce._load_model(MODEL_CONFIG, MODEL_PATH, USE_CUDA)

    asv21_wav_files, asv21_output_files, _, gender = ce._get_files(ASV21_PATH, ASV21_OUTPUT_PATH_MODEL, SIZE)
    asv21_embedd = ce._create_embeddings(asv21_wav_files, asv21_output_files, ap, model, USE_CUDA)

    signature_labels, names = ce._get_signature_labels(asv21_wav_files, gender)
    result_dict = {}
    for sig_labels, name in tqdm(zip(signature_labels, names)):
        labels = ce.normalize_points(sig_labels)

        print(f'{name}: {np.nanmax(sig_labels)} {np.nanmin(sig_labels)} {np.nanmean(sig_labels)} {np.nanmedian(sig_labels)}')

        coll_min, coll_max = [], []
        for idx, current in enumerate(asv21_embedd):
            found_min, found_max = ce.calculate_range(current, labels[idx], asv21_embedd, labels, k)
            coll_min.append(found_min)
            coll_max.append(found_max)
        result_dict[name] = [np.nanmean(coll_min), np.nanmean(coll_max)]

    _table_to_latex(result_dict, [f'min label for k={k}', f'max label for k={k}'],  PLOT_PATH + 'asv21_signatures_metric_table.tex')

if __name__ == '__main__':
    # plot_asv19_attack_signatures()
    # plot_asv19_asv21()
    calculate_table_asv21_sig_metric() 
    # plot_split(RANDOM_SPLIT_MODEL_CONFIG, RANDOM_SPLIT_MODEL_PATH, RANDOM_SPLIT_CSV, 'asv19_random_10_split', ASV19_OUTPUT_PATH_RANDOM_SPLIT)
    # plot_split(SPEAKER_SPLIT_MODEL_CONFIG, SPEAKER_SPLIT_MODEL_PATH, SPEAKER_SPLIT_CSV, 'asv19_4_speaker_split_3', ASV19_OUTPUT_PATH_SPEAKER_SPLIT)
