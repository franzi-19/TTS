### Speaker Encoder

This is an implementation of https://arxiv.org/abs/1710.10467. This model can be used for voice and speaker embedding. In the current version a lstm model is used

With the code here you can generate d-vectors for both multi-speaker and single-speaker TTS datasets, then visualise and explore them along with the associated audio files in an interactive chart.

Below is an example showing embedding results of various speakers. You can generate the same plot with the provided notebook as demonstrated in [this video](https://youtu.be/KW3oO7JVa7Q).

![](umap.png)

<!-- Download a pretrained model from [Released Models](https://github.com/mozilla/TTS/wiki/Released-Models) page. -->

## Installation
- Change to TTS: cd TTS
- For python 3.9 create and start a virtual enviroment eg. with: pipenv install -r requirements.txt
    - additional packages for plotting the embeddings: bokeh
- Download a pretrained model from Released Models page if needed
    - eg. best_model.pth.tar and its config from https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3 (currently the only one that works)
- Download all datasets you want to train with and/or create embeddings for and save them all in the same parent folder

## Usage
- Start your virtual enviroment
- Define 'TTS/speaker_encoder/config.json' for your needs. Note that, audio parameters should match your TTS model
    - make sure that your desired dataset is mentioned
- Example training call:
    ```
    python -m TTS.bin.train_encoder \
    --config_path TTS/speaker_encoder/config.json
    ```
- Resume training:
    ```
    python -m TTS.bin.train_encoder \
    --restore_path path/to/best_model.pth.tar \
    --config_path TTS/speaker_encoder/config.json
    ```
- Only generate embedding vectors:
    - This code parses all .wav or .flac files at the given dataset path and generates the same folder structure under the output path with the generated embedding files.
    ```
    python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth.tar model/config/path/config.json dataset/path/ output_path
    ``` 
- Generate and plot the embeddings:
    ```
    python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth.tar model/config/path/config.json dataset/path/ output_path --plot_path folder/where/to/save/plot
    ``` 
- Watch training on Tensorboard as in TTS
