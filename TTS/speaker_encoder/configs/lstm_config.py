from dataclasses import dataclass, field
from TTS.speaker_encoder.speaker_encoder_config import SpeakerEncoderConfig

@dataclass
class LstmConfig(SpeakerEncoderConfig):
    model: str = "lstm"
    model_params: dict = field(
        default_factory=lambda: {
            "model_name" : "lstm",
            "input_dim" : 80, 
            "proj_dim" : 256, 
            "lstm_dim" : 768,
            "num_lstm_layers" : 3,
            "use_lstm_with_projection" : True, 
        }
    )

    voice_len: int = 1 # number of seconds for each training instance
    num_utters_per_speaker: int = 10 # 200 # number of used utterances for each speaker each batch
    num_speakers_in_batch: int = 32 # 64 # 7 #  number of speaker in each batch
    skip_speakers: bool = True # skip speakers with samples less than "num_utters_per_speaker"
    storage: dict = field(
        default_factory=lambda:{
        "sample_from_storage_p": 0, # the probability with which we'll sample from the DataSet in-memory storage
        "storage_size": 35 # the size of the in-memory storage with respect to a single batch
        }
    )

    # training params
    max_train_step: int = 1000  # end training when number of training steps reaches this value.
    loss: str = "angleproto" # "ge2e", "angleproto", "softmaxproto"
    lr_decay: bool = True # if true, Noam learning rate decaying is applied through training
    warmup_steps: int =  400 # Noam decay steps to increase the learning rate from 0 to "lr"
    lr: int = 0.01 # Initial learning rate. If Noam decay is active, maximum learning rate
    grad_clip: int = 0.0  # 0.0 meaning no gradient clipping

    # logging params
    steps_plot_stats: int = 10
    print_step: int = 100
    save_step: int = 500

    audio_augmentation: dict = field(
        default_factory=lambda:{
        # "p": 0.5, #  propability of apply this method, 0 is disable rir and additive noise augmentation
        # "rir":{
        #     "rir_path": "/workspace/store/ecasanova/ComParE/RIRS_NOISES/simulated_rirs/",
        #     "conv_mode": "full"
        # },
        # "additive":{
        #     "sounds_path": "/workspace/store/ecasanova/ComParE/musan/",
        #     // list of each of the directories in your data augmentation, if a directory is in "sounds_path" but is not listed here it will be ignored
        #     "speech":{
        #         "min_snr_in_db": 13,
        #         "max_snr_in_db": 20,
        #         "min_num_noises": 2,
        #         "max_num_noises": 3
        #         },
        #     "noise":{
        #         "min_snr_in_db": 0,
        #         "max_snr_in_db": 15,
        #         "min_num_noises": 1,
        #         "max_num_noises": 1
        #         },
        #     "music":{
        #         "min_snr_in_db": 5,
        #         "max_snr_in_db": 15,
        #         "min_num_noises": 1,
        #         "max_num_noises": 1
        #         }
        # },
        # add a gaussian noise to the data in order to increase robustness
        # //add a gaussian noise to the data in order to increase robustness
        # "gaussian":{ // as the insertion of Gaussian noise is quick to be calculated, we added it after loading the wav file, this way, even audios that were reused with the cache can receive this noise
        #     "p": 0.5, // propability of apply this method, 0 is disable
        #     "min_amplitude": 0.0,
        #     "max_amplitude": 1e-5    
        # }
        }
    )
