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
    - eg. best_model.pth.tar and its config from https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3
- Download all datasets to want to create embeddings for
- Define 'config.json' for your needs. Note that, audio parameters should match your TTS model.
    - adjust dataset paths in your own dataset config file to specify for which datasets the embeddings should be created
    - make sure that both config.json files include "model": "glow_tts",

## Generate embeddings
- This code parses all .wav files at the given dataset path and generates a config file with a embeddings:
     ```
    python -m TTS.bin.compute_embeddings --use_cuda true \
    model/path/best_model.pth.tar \
    model/path/config.json \
    path/to/your/own/dataset/config/config.json \
    folder/where/to/store/the/embeddings
    ```
    <!-- ```
    python -m TTS.bin.compute_embeddings --use_cuda true \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/speaker_encoder_model_mueller91/best_model.pth.tar \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/speaker_encoder_model_mueller91/config.json \
    TTS/speaker_encoder/configs/own_config.json \
    /run/media/franzi/ssd/Without_Backup/Uni_wb/Masterarbeit/embeddings/
    ``` -->


## Plot the embeddings
- for python 3.9: ```pipenv install -r requirements.txt```
- ```pipenv install ipykernel, bokeh```
- ```jupyter notebook PlotUmapLibriTTS.ipynb``` or 'open with jupyter notebook'
- start the notebook

## Training
TODO
- Example training call:
    ```
    python TTS/speaker_encoder/train.py --config_path speaker_encoder/config.json --data_path ~/Data/Libri-TTS/train-clean-360
    ```
