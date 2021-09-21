from dataclasses import dataclass, field
from TTS.speaker_encoder.speaker_encoder_config import SpeakerEncoderConfig

# TODO

@dataclass
class ResnetConfig(SpeakerEncoderConfig):
    model: str = "resnet"
    model_params: dict = field(
        default_factory=lambda: {
            "model_name" : "resnet",
            "input_dim": 64,
            "proj_dim": 512,
            "layers": [3, 4, 6, 3],
            "num_filters": [32, 64, 128, 256],
            "encoder_type": "ASP",
            "log_input": False,
        }
    )

    voice_len: int = 2 # number of seconds for each training instance
    num_utters_per_speaker: int = 2 # ?
    num_speakers_in_batch: int = 10 # Batch size for training
    skip_speakers: bool = True # skip speakers with samples less than "num_utters_per_speaker"
    storage: dict = field(
        default_factory=lambda:{
        "sample_from_storage_p": 0, # the probability with which we'll sample from the DataSet in-memory storage
        "storage_size": 35 # the size of the in-memory storage with respect to a single batch
        }
    )

    loss: str = "ge2e"
    lr_decay: bool = True # if true, Noam learning rate decaying is applied through training
    warmup_steps: int =  4000 # Noam decay steps to increase the learning rate from 0 to "lr"
    lr: int = 0.0001 # Initial learning rate. If Noam decay is active, maximum learning rate

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