import queue
import random
import tempfile
from pathlib import Path

import g711
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ap, meta_data, voice_len=1.6, num_speakers_in_batch=64,
                 storage_size=1, sample_from_storage_p=0.5, additive_noise=0,
                 num_utter_per_speaker=10, skip_speakers=False, feature_type='mfcc', 
                 use_caching=False, cache_path=None, dataset_folder=None, verbose=False, train=True,
                 codecs=None, prob=0.0):
        """
        Args:
            ap (TTS.tts.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            seq_len (int): voice segment length in seconds.
            verbose (bool): print diagnostic information.
            TODO
        """
        self.items = meta_data # items from preprocess functions
        self.sample_rate = ap.sample_rate
        self.hop_length = ap.hop_length
        self.voice_len = voice_len
        self.seq_len = int(voice_len * self.sample_rate)
        self.num_speakers_in_batch = num_speakers_in_batch
        self.num_utter_per_speaker = num_utter_per_speaker
        self.skip_speakers = skip_speakers
        self.feature_type = feature_type
        self.ap = ap
        self.use_caching = use_caching
        self.cache_path = cache_path
        self.dataset_folder = dataset_folder
        self.verbose = verbose
        self.__parse_items()
        self.storage = queue.Queue(maxsize=storage_size*num_speakers_in_batch)
        self.sample_from_storage_p = float(sample_from_storage_p)
        self.additive_noise = float(additive_noise)
        self.codecs = codecs
        self.prob = prob
        if self.verbose:
            print(f"\n > DataLoader Initialization for {'Training' if train else 'Testing'}")
            print(f" | > Number of found Speakers: {len(self.speakers)}")
            if num_speakers_in_batch <= len(self.speakers):
                print(f" | > Speakers per Batch: {num_speakers_in_batch}")
            else:
                print(f" | > Speakers per Batch: {len(self.speakers)} (adjusted because specified number was too high)")
            if not use_caching:
                print(f" | > Storage Size: {self.storage.maxsize} speakers, each with {num_utter_per_speaker} utters")
                print(f" | > Sample_from_storage_p : {self.sample_from_storage_p}")
            print(f" | > Noise added : {self.additive_noise}")
            print(f" | > Number of Items in the Dataset : {len(self.items)}")
            print(f" | > Sequence Length: {self.seq_len}")
            if use_caching:
                print(f" | > Cache Path: {self.cache_path}")
            else:
                print(f" | > No caching used")
            print("")

    def load_wav(self, filename_to_load, codec=None):
        """
        always returns a wav file
        if codecs are given, it applies given codec first and then returns the modified wav
        """
        if codec == None:
            audio = self.ap.load_wav(filename_to_load, sr=self.ap.sample_rate)
        else:
            audio = self._apply_codec(codec, filename_to_load)
        return audio

    # TODO: maybe do it like the different dataset functions
    # TODO: check if intermediates are saved at right
    def _apply_codec(self, codec, filename_to_load):
        possible_codecs = ["opus", "gsm", "ulaw", "g722", "alaw", "wav"]
        assert codec in possible_codecs, f"Codec {codec} needs to be one of these: {possible_codecs}"

        with tempfile.TemporaryDirectory() as dirpath:
            intermediate_1 = Path(dirpath) / f"{codec}.{codec}"
            intermediate_2 = Path(dirpath) / f"{Path(filename_to_load).stem}_{codec}.wav"
            frame_rate = {"opus":16000, "gsm":8000, "g722":16000, "wav":16000, "ulaw":8000, "alaw":8000}
            
            if codec in ["opus", "gsm", "g722", "wav"]:
                sound = AudioSegment.from_file(filename_to_load)
                sound = sound.set_frame_rate(frame_rate[codec])
                sound.export(intermediate_1, format=codec)
                sound = AudioSegment.from_file(intermediate_1, codec=codec)
                sound = sound.set_frame_rate(frame_rate[codec])
                sound.export(intermediate_2, format="wav")

            elif codec == "ulaw": 
                wav = self.ap.load_wav(filename_to_load, sr=self.ap.sample_rate)
                ulaw = g711.encode_ulaw(wav)
                decoded = g711.decode_ulaw(ulaw)
                sf.write(intermediate_2, decoded, frame_rate[codec])

            try:
                if codec == "alaw":
                    wav = self.ap.load_wav(filename_to_load, sr=self.ap.sample_rate)
                    alaw = g711.encode_alaw(wav)
                    decoded = g711.encode_alaw(alaw)
                    # sf.write(intermediate_2, decoded, frame_rate[codec])
            except Exception:
                print("alaw not working for ", filename_to_load)
                codec = "wav"
                intermediate_1 = Path(dirpath) / f"{codec}.{codec}"
                intermediate_2 = Path(dirpath) / f"{Path(filename_to_load).stem}_{codec}.wav"
                sound = AudioSegment.from_file(filename_to_load, format="wav")
                sound.export(intermediate_1, format=codec)
                sound = AudioSegment.from_file(intermediate_1, codec=codec)
                sound = sound.set_frame_rate(frame_rate[codec])
                sound.export(intermediate_2, format="wav")
            
            assert intermediate_2.exists(), f"Something went wrong while applying the codecs, file {intermediate_2} should exist"
            audio =  self.ap.load_wav(intermediate_2, sr=self.ap.sample_rate)
            
        return audio

    def _select_codec(self, codecs, prob):
        # select codec
        if isinstance(codecs, list):
            codec = random.sample(codecs, 1)[0]
        else:
            codec = codecs

        # if to apply codec or just use normal wav
        if random.choices([0,1], [1.0-prob, prob], k=1)[0]:
            return codec
        else:
            return None

    # not used
    # def load_data(self, idx):
    #     text, wav_file, speaker_name = self.items[idx]
    #     wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)
    #     mel = self.ap.melspectrogram(wav).astype("float32")
    #     # sample seq_len

    #     assert text.size > 0, self.items[idx][1]
    #     assert wav.size > 0, self.items[idx][1]

    #     sample = {
    #         "mel": mel,
    #         "item_idx": self.items[idx][1],
    #         "speaker_name": speaker_name,
    #     }
    #     return sample

    def __parse_items(self):
        """
        matches speaker with their utterances for self.speaker_to_utters, remove speaker with not enough utterances if skip_speakers == True
        """
        self.speaker_to_utters = {}
        for i in self.items:
            path_ = i[1]
            speaker_ = i[2]
            if speaker_ in self.speaker_to_utters.keys():
                self.speaker_to_utters[speaker_].append(path_)
            else:
                self.speaker_to_utters[speaker_] = [path_, ]

        if self.skip_speakers:
            self.speaker_to_utters = {k: v for (k, v) in self.speaker_to_utters.items() if
                                      len(v) >= self.num_utter_per_speaker}
            assert self.speaker_to_utters != {}, f"Every speaker has less than 'num_utter_per_speaker', pleace specify a lower number"


        self.speakers = [k for (k, v) in self.speaker_to_utters.items()]

    def __len__(self):
        """
        return large number to avoid reaching the end of the dataset
        """
        return int(1e10)

    def __sample_speaker(self):
        """
        returns a random speaker (with some of their utterances)
        """
        speaker = random.sample(self.speakers, 1)[0]
        return speaker

    # used for non-caching
    def __sample_wavs(self, speaker):
        """
        returns list of wavs of length 'num_utter_per_speaker', labels = speaker
        if speaker has not enough utterances another speaker will be sampled
        """
        count = 0
        wavs = []
        labels = []
        while count < self.num_utter_per_speaker:
            utter = random.sample(self.speaker_to_utters[speaker], 1)[0]
            wav = self.load_wav(utter)

            if wav.shape[0] - self.seq_len > 0: # wav is long enough
                count += 1
            else: # remove too short utterances, if no utterances left draw another speaker
                self.speaker_to_utters[speaker].remove(utter)
                if len(self.speaker_to_utters[speaker]) == 0:
                    self.speakers.remove(speaker)
                    speaker = self.__sample_speaker()
                continue

            wavs.append(wav)
            labels.append(speaker)
        return wavs, labels
            
    # returns the next speaker
    def __getitem__(self, idx):
        speaker = self.__sample_speaker()
        return speaker

    # batch: batch of speaker
    # returns features and labels
    def collate_fn(self, batch):
        labels = []
        feats = []
        found_speaker = []
        for speaker in batch:
            while speaker in found_speaker: # speaker already used -> sample another one
                speaker = self.__sample_speaker()
            found_speaker.append(speaker)

            if not self.use_caching:
                feats_, labels_ = self.collate_without_caching(speaker)
            else:
                feats_, labels_ = self.collate_with_caching(speaker)
            labels.append(labels_)
            feats.extend(feats_)

        feats = torch.stack(feats) # [250,40,138] for no caching, 
        return feats.transpose(1, 2), labels

    def collate_with_caching(self, speaker):
        count = 0
        labels_ = []
        feats_ = []
        while count < self.num_utter_per_speaker:
            utter = random.sample(self.speaker_to_utters[speaker], 1)[0]
            codec = self._select_codec(self.codecs, self.prob)
            save_path = self.get_save_path(utter, codec)
            if save_path.exists():
                feat = self.load_np(save_path)  # 40, 138
            else:
                wav = self.load_wav(utter, codec) # [23915,]
                if wav.shape[0] < self.seq_len: # remove too short utterances, if no utterances left draw another speaker
                    self.speaker_to_utters[speaker].remove(utter)
                    if len(self.speaker_to_utters[speaker]) == 0:
                        self.speakers.remove(speaker)
                        speaker = self.__sample_speaker()
                    continue
                
                feat = self._add_gaussian_noise(wav)  # [23915,]

                if self.feature_type == 'mfcc':
                    feat = self.ap.melspectrogram(wav) # [40, 202]
                
                np.save(file=save_path, arr=feat)
            feat = self._select_subset(feat)
            feat = torch.FloatTensor(feat) 

            count += 1
            feats_.append(feat)
            labels_.append(speaker)
            
        return feats_, labels_

    def get_save_path(self, wav_path, codec):
        filename = f"{Path(wav_path).stem}_{self.feature_type}_{codec}.npy"

        parent_folder = Path(wav_path).parents[0]
        parent_folder = parent_folder.relative_to(self.dataset_folder)
        parent_folder = self.cache_path / parent_folder 
        parent_folder.mkdir(parents=True, exist_ok=True)

        save_path = parent_folder / Path(filename)
        return save_path

    def collate_without_caching(self, speaker):
        if random.random() < self.sample_from_storage_p and self.storage.full():
            # sample from storage (if full), ignoring the speaker
            wavs_, labels_ = random.choice(self.storage.queue)
        else:
            # don't sample from storage, but from HDD
            wavs_, labels_ = self.__sample_wavs(speaker)
            # if storage is full, remove an item
            if self.storage.full():
                _ = self.storage.get_nowait()
            # put the newly loaded item into storage
            self.storage.put_nowait((wavs_, labels_))

        wavs_ = [self._add_gaussian_noise(wav) for wav in wavs_]

        # get a random subset of each of the wavs and convert to MFCC.
        offsets_ = [random.randint(0, wav.shape[0] - self.seq_len) for wav in wavs_]

        # use selected feature type
        if self.feature_type == 'mfcc':
            mels_ = [self.ap.melspectrogram(wavs_[i][offsets_[i]: offsets_[i] + self.seq_len]) for i in range(len(wavs_))] # [40,138]
            feats_ = [torch.FloatTensor(mel) for mel in mels_]
        else: # TODO: not working
            subsets_ = [wavs_[i][offsets_[i]: offsets_[i] + self.seq_len] for i in range(len(wavs_))]
            feats_ = [torch.FloatTensor(subset) for subset in subsets_]

        return feats_, labels_

    def _add_gaussian_noise(self, wav):
        """
        add random gaussian noise
        """
        if self.additive_noise > 0:
            noise_ = np.random.normal(0, self.additive_noise, size=len(wav))
            wav_ = wav + noise_
        return wav_

    def _select_subset(self, feat):
        """
        get a random subset
        """
        if self.feature_type == 'mfcc':
            seq_len = self.seq_len // self.hop_length # calculate mfcc size based on specifed audio_len in config
            offset = random.randint(0, feat.shape[1] - seq_len)
            return feat[:,offset: offset + seq_len]
        else:
            offset = random.randint(0, feat.shape[0] - self.seq_len)
            print(offset)
            return feat[offset: offset + self.seq_len]

    @staticmethod
    def load_np(filepath):
        try:
            data = np.load(filepath).astype('float32')
        except ValueError as e: # propably a corrupted numpy file
            raise ValueError(f"ValueError while loading {filepath}")
        return data
    