### Speaker Encoder
This is an implementation of https://arxiv.org/abs/1710.10467. This model can be used for voice and speaker embedding.

With the code here you can generate d-vectors for both multi-speaker and single-speaker TTS datasets, then visualise and explore them along with the associated audio files in an interactive chart.

Below is an example showing embedding results of various speakers. You can generate the same plot with the provided notebook as demonstrated in [this video](https://youtu.be/KW3oO7JVa7Q).

![](umap.png)

<!-- Download a pretrained model from [Released Models](https://github.com/mozilla/TTS/wiki/Released-Models) page.

To run the code, you need to follow the same flow as in TTS.

- Define 'config.json' for your needs. Note that, audio parameters should match your TTS model.
- Example training call ```python speaker_encoder/train.py --config_path speaker_encoder/config.json --data_path ~/Data/Libri-TTS/train-clean-360```
- Generate embedding vectors ```python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth.tar model/config/path/config.json dataset/path/ output_path``` . This code parses all .wav files at the given dataset path and generates the same folder structure under the output path with the generated embedding files.
- Watch training on Tensorboard as in TTS -->

# How to generate embeddings
## Set it up
- change to TTS: ```cd TTS```
- for python 3.9 eg. do: ```pipenv install -r requirements.txt```
- Download a pretrained model from [Released Models](https://github.com/mozilla/TTS/wiki/Released-Models) page
    - eg. best_model.pth.tar and its config from https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3 (currently the only one that works)
- Download all datasets to want to create embeddings for
- Define 'config.json' for your needs. Note that, audio parameters should match your TTS model.
    - adjust dataset paths in your own dataset config file to specify for which datasets the embeddings should be created
    - make sure that both config.json files include "model": "glow_tts",

## Generate embeddings
- start your virtual enviroment
- make sure that your *desired dataset* is mentioned in your config.json
- this code parses all .wav files at given dataset paths from your config and generates a json file with one embedding per found .wav file:
     ```
    python -m TTS.bin.compute_embeddings --use_cuda true \
    model/path/best_model.pth.tar \
    model/path/config.json \
    path/to/your/own/config/config.json \
    folder/where/to/store/the/embeddings
    ```
    <!-- ```
    python -m TTS.bin.compute_embeddings --use_cuda true \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/speaker_encoder_models/own_lstm/train_model_lstm-August-31-2021_02+03PM-d479d32d/best_model.pth.tar \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/speaker_encoder_models/own_lstm/train_model_lstm-August-31-2021_02+03PM-d479d32d/config.json \
    TTS/speaker_encoder/configs/own_config_to_create_embed.json \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/embeddings/asvspoof_19/own_lstm/
    ``` -->


## Plot the embeddings
- for python 3.9: ```pipenv install -r requirements.txt``` and change to the virtual enviroment
- ```pipenv install ipykernel, bokeh```
- ```jupyter notebook PlotUmapLibriTTS.ipynb``` or 'open with jupyter notebook'
- run the notebook

## Training
- make sure that for a model *model_name* there exists a config file *model_name_config.py* in TTS/speaker_encoder/configs
- mention that model in your own config file with "model": "*model_name*" and the dataset you want to train with
- Example training call:
    ```
    python -m TTS.bin.train_encoder \
    --config_path /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/TTS/TTS/speaker_encoder/configs/own_config_to_train_model.json
    ```
