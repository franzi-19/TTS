import pandas as pd
from tqdm import tqdm
import TTS.speaker_encoder.attack_signatures as attack_signatures
import TTS.speaker_encoder.compute_embeddings as ce
import TTS.speaker_encoder.create_plots as create_plots

size = 100 # = None to use all wav files

# 4.3
def plot_asv19_attack_signatures(asv19_path, asv19_output_path, plot_path, latex_filepath):
    asv19_wav_files, asv19_output_files, asv19_labels, gender = ce._get_files(asv19_path, asv19_output_path, size)

    all_signature_names = attack_signatures.get_all_names_one_result()
    all_signature_names.append('gender')
    
    embeds = []
    for idx, wav in tqdm(enumerate(asv19_wav_files)):
        embed = []
        for name in all_signature_names:
            sig_function = attack_signatures.get_signature_by_name(name)
            if name == 'gender': embed.append(gender[idx]) # TODO: assign gender a float number, see normalize_points()
            else: embed.append(sig_function(wav))
        embeds.append(embed)
        
    # TODO: continue testing here
    embeds = ce.normalize_points(embeds)

    _calculate_centroid_and_distance(latex_filepath, embeds, asv19_labels)

    create_plots.plot_embeddings(embeds, None, plot_path, asv19_labels, "")

def _calculate_centroid_and_distance(latex_filepath, embeds, labels):
    cluster = {key: [] for key in labels}
    for idx, point in enumerate(embeds):
        cluster[labels[idx]].append(point)

    table = {}
    for cl in cluster:
        centroid, mean_distance = ce.calculate_mean_distance(cluster[cl])
        table[cl] = [centroid, mean_distance]

    _table_to_latex(table, ['attack ID', 'centroid', 'mean distance'], latex_filepath)

def _table_to_latex(input, header, latex_filepath):
    df = pd.DataFrame(data=input).T
    df.columns = header
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
                                    '/home/franzi/masterarbeit/TTS/speaker_encoder/plots_paper/',
                                    '/home/franzi/masterarbeit/TTS/speaker_encoder/plots_paper/asv19_attack_signatures_table.tex')
