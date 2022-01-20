import sys

import librosa
import parselmouth
from pydub import AudioSegment


# https://librosa.org/doc/main/generated/librosa.pyin.html
def get_fundamental_frequency_librosa(wav_path): # (array([ nan,  nan,  nan,  nan,    nan,  nan,  nan,  nan,    nan,  nan,  nan, 225.14225624,   231.73942791, 237.15608027, 239.91170119, 237.15608027,   233.08188076, 225.14225624, 270.85177094,  nan,    nan, 223.84553226, 197.13312122, 190.41804342,   181.81906999, 176.64303401, 173.60841241, 170.6259238 ,    nan, 105.03257643,  81.93226047,  82.40688923,    82.88426748,  82.40688923,  82.40688923,  84.8215954 ,    87.8128225 ,  90.909535  ,  nan,  nan,    nan,  nan,  nan,  nan,    nan,  nan, 191.52112393, 190.41804342,   187.14677541, 182.87233715,  nan, 202.90956259,   191.52112393, 187.14677541, 179.73069986, 178.6955272 ,   180.77186921, 180.77186921, 181.81906999, 181.81906999,   180.77186921, 184.99721136,  nan,  nan,    nan,  nan, 183.93170582, 178.6955272 ,   173.60841241, 170.6259238 , 166.72882232, 164.81377846,   162.92073075, 161.98237639, 161.04942655, 161.04942655,   161.98237639, 162.92073075, 163.86452094, 162.92073075,   162.92073075, 162.92073075, 163.86452094,  nan,    nan,  nan,  nan,  nan,    nan,  nan,  nan,  nan,    nan,  nan,  nan,  nan,    nan,  nan,  nan,  nan]),     array([False, False, False, False, False, False, False, False, False,   False, False,  True,  True,  True,  True,  True,  True,  True,    True, False, False,  True,  True,  True,  True,  True,  True,    True, False,  True,  True,  True,  True,  True,  True,  True,    True,  True, False, False, False, False, False, False, False,   False,  True,  True,  True,  True, False,  True,  True,  True,    True,  True,  True,  True,  True,  True,  True,  True, False,   False, False, False,  True,  True,  True,  True,  True,  True,    True,  True,  True,  True,  True,  True,  True,  True,  True,    True,  True, False, False, False, False, False, False, False,   False, False, False, False, False, False, False, False, False,   False]),    array([0.01000123, 0.01      , 0.01      , 0.01000005, 0.01003777,   0.01001876, 0.03010686, 0.09203757, 0.01      , 0.01000052,   0.01082446, 0.61594989, 0.75716013, 0.75716013, 0.82666856,   0.68608281, 0.61594989, 0.24075439, 0.01000123, 0.01      ,   0.01000002, 0.01034906, 0.01025905, 0.01005297, 0.02294422,   0.24075439, 0.32333606, 0.32333606, 0.01082446, 0.01      ,   0.01005297, 0.01646759, 0.32333606, 0.27960517, 0.2065068 ,   0.09203757, 0.06608151, 0.01046753, 0.01000052, 0.01398021,   0.01108579, 0.01034906, 0.01310026, 0.01508559, 0.01002672,   0.01000187, 0.0560714 , 0.61594989, 0.75716013, 0.0560714 ,   0.02616803, 0.01185351, 0.12793709, 0.94593021, 0.75716013,   0.54852505, 0.89114449, 0.98487898, 0.98487898, 0.94593021,   0.82666856, 0.06608151, 0.01240312, 0.01185351, 0.01035246,   0.01      , 0.01010197, 0.03489743, 0.48498681, 0.68608281,   0.82666856, 0.75716013, 0.94593021, 0.98487898, 0.98487898,   0.98487898, 0.98487898, 0.98487898, 0.94593021, 0.98487898,   1.        , 0.89114449, 0.48498681, 0.01005297, 0.01508559,   0.02294422, 0.04069733, 0.01      , 0.01000419, 0.01000903,   0.01005297, 0.01019108, 0.01046753, 0.01001307, 0.01000002,   0.01014005, 0.1504646 , 0.01      , 0.01      , 0.01000002]))
    wav, _ = librosa.load(wav_path)
    fundamental_freq, bool_flags, prob_voiced = librosa.pyin(wav, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'))
    return fundamental_freq, bool_flags, prob_voiced

# https://librosa.org/doc/main/generated/librosa.feature.spectral_flatness.html
def measure_noise_likeness_librosa(wav_path): # [[3.7553793e-04 3.0664928e-04 8.0501213e-06 5.7700845e-07 2.3003638e-07 2.8720575e-07 1.0714828e-06 1.4431788e-06 2.0147532e-07 1.8071314e-07 1.7122144e-07 1.0818940e-07 6.7840197e-08 6.7763132e-08 7.6435356e-08 9.3441422e-08 2.9103197e-07 6.6904786e-06 2.9306048e-05 1.3326178e-05 6.5162993e-07 1.3972469e-07 9.0489330e-08 8.8215579e-08 7.0659020e-08 7.6277644e-08 1.2095609e-07 4.9281419e-07 6.6454745e-06 4.3192770e-07 2.2496036e-07 2.2962249e-07 2.6260258e-07 3.2724532e-07 3.6340938e-07 6.8472912e-07 7.8960056e-06 3.5884361e-05 3.7452595e-05 4.5983310e-05 5.8105968e-05 4.7055488e-05 3.6743644e-05 2.6517184e-05 7.6005887e-07 2.2495252e-07 1.6059994e-07 2.2590629e-07 1.1908542e-06 4.3925238e-06 5.7601682e-07 2.1895099e-07 1.6720399e-07 2.3806639e-07 5.4014930e-07 3.3141450e-07 1.6315430e-07 9.4504081e-08 1.0257747e-07 1.9859989e-07 1.8571145e-06 3.0523202e-06 2.3858443e-06 4.0670629e-06 1.3747416e-06 3.7010247e-07 2.9327913e-07 2.5682380e-07 2.8280485e-07 2.8320721e-07 2.5409693e-07 1.8268835e-07 1.3030179e-07 8.1324607e-08 6.0526901e-08 5.5649981e-08 6.4158897e-08 9.1651785e-08 9.7472359e-08 9.9112938e-08 1.4807705e-07 5.1185106e-07 5.4064158e-06 7.4308705e-06 7.7882387e-06 1.0173372e-05 8.6184345e-06 9.0409839e-07 5.5211171e-07 5.6716402e-07 6.7232838e-07 9.5115763e-07 1.6485910e-06 2.4031019e-06 1.5350441e-06 2.5724971e-06 3.1507909e-05 7.6383119e-05 1.9946076e-04 5.6522287e-04]]
    wav, _ = librosa.load(wav_path)
    return librosa.feature.spectral_flatness(wav)

# same as voiced unvoiced rate
# def get_all_silence_length_rate(sound):
#     from pydub import AudioSegment
#     from pydub.silence import detect_silence

#     ranges = detect_silence(sound, min_silence_len = 20, silence_thresh = -30) 

#     print("total: ", sound.duration_seconds)
#     print('ranges:', ranges)

#     silence = sum([range[1] - range[0] for range in ranges])

#     print("silence:", silence)
#     print("len audio", len(sound))
    
#     return silence / len(sound)

# https://github.com/mueller91/crawler_tts/blob/master/audio_selection/features_from_audio.py#L224
def get_voiced_unvoiced_rate(wav_path):   # 0.5286343612334802
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()

    voiced_frames = parselmouth.praat.call(pitch, "Count voiced frames")
    total_frames = parselmouth.praat.call(pitch, "Get number of frames")
    vuv_rate = voiced_frames / total_frames

    return vuv_rate

# https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html
# https://www.fon.hum.uva.nl/praat/download_linux.html
# https://github.com/drfeinberg/PraatScripts
# def get_pitch_info(snd): # 175.22250009963122, 39.23073677537671, 81.35743208724539, 240.92413524792198, 273.58519194246514
#     pitch = snd.to_pitch()

#     pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "Hertz")
#     pitch_std = parselmouth.praat.call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")
#     pitch_min = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "None")
#     pitch_max = parselmouth.praat.call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "None")
#     pitch_mas = parselmouth.praat.call(pitch, "Get mean absolute slope", "Hertz")

#     return pitch_mean, pitch_std, pitch_min, pitch_max, pitch_mas

def get_pitch_mean(wav_path): # 175.22250009963122
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "Hertz")

def get_pitch_std(wav_path): # 39.23073677537671
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")

def get_pitch_min(wav_path): # 81.35743208724539
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "None")

def get_pitch_max(wav_path): # 240.92413524792198
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "None")

def get_pitch_mas(wav_path): # 273.58519194246514
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get mean absolute slope", "Hertz")


# TODO find tmin, tmax
def get_intensity_info(audio_path): 
    snd = parselmouth.Sound(str(audio_path))
    intensity = snd.to_intensity()

    intensity_min = parselmouth.praat.call(intensity, "Get minimum", tmin, tmax, "Parabolic")
    intensity_max = parselmouth.praat.call(intensity, "Get maximum", tmin, tmax, "Parabolic")
    intensity_mean = parselmouth.praat.call(intensity, "Get mean", 0.0, 0.0, "dB")
    intensity_std = parselmouth.praat.call(intensity, "Get standard deviation", 0.0, 0.0)

    return intensity_min, intensity_max, intensity_mean, intensity_std

def get_power(wav_path): # 0.021697740538486117
    snd = parselmouth.Sound(str(wav_path))
    return snd.get_power()

def get_energy(wav_path): # 0.04990480323851807
    snd = parselmouth.Sound(str(wav_path))
    return snd.get_energy()

def get_jitter(wav_path): # 0.020679496393133184
    snd = parselmouth.Sound(str(wav_path))
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    return parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

def get_shimmer(wav_path): # 0.13407159808861868
    snd = parselmouth.Sound(str(wav_path))
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    return parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

# def get_harmonity_info(snd): # 11.343795497737961, 6.498392039266402
#     harmonicity = snd.to_harmonicity()
#     hnr_mean = parselmouth.praat.call(harmonicity, "Get mean", 0.0, 0.0)
#     hnr_std = parselmouth.praat.call(harmonicity, "Get standard deviation", 0.0, 0.0)

#     return hnr_mean, hnr_std

def get_hnr_mean(wav_path): # 11.343795497737961
    snd = parselmouth.Sound(str(wav_path))
    harmonicity = snd.to_harmonicity()
    return parselmouth.praat.call(harmonicity, "Get mean", 0.0, 0.0)

def get_hnr_std(wav_path): # 6.498392039266402
    snd = parselmouth.Sound(str(wav_path))
    harmonicity = snd.to_harmonicity()
    return parselmouth.praat.call(harmonicity, "Get mean", 0.0, 0.0)

# def get_volume(sound): # 32767, -16.637252347739015, -0.0002650763603796191, 2.3
#     from pydub import AudioSegment
 
#     peak_amplitude = sound.max  # highest amplitude
#     loudness = sound.dBFS       # loudness in dBFS (Decibels relative to full scale, db relative to the maximum possible loudness).
#     max_loudness = sound.max_dBFS   # highest amplitude in dBFS (relative to the highest possible amplitude value)
#     duration = sound.duration_seconds # duration in seconds
#     return peak_amplitude, loudness, max_loudness, duration

def get_peak_amplitude(wav_path): # highest amplitude, 32767
    sound = AudioSegment.from_file(wav_path)
    return sound.max

def get_loudness(wav_path): # loudness in dBFS (Decibels relative to full scale, db relative to the maximum possible loudness), -16.637252347739015
    sound = AudioSegment.from_file(wav_path)
    return sound.dBFS

def get_max_loudness(wav_path): # highest amplitude in dBFS (relative to the highest possible amplitude value), -0.0002650763603796191
    sound = AudioSegment.from_file(wav_path)
    return sound.max_dBFS

def get_duration(wav_path): # duration in seconds, 2.3
    sound = AudioSegment.from_file(wav_path)
    return sound.duration_seconds

def get_lead_trail_silence_rate(wav_path): # 0.040914916691314175
    wav, _ = librosa.load(wav_path)
    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    silence = librosa.get_duration(wav) - librosa.get_duration(trimmed)
    return silence / librosa.get_duration(wav)

def get_gender(wav_path): # dummy method
    return 0


def apply_all_signature_one_result(wav_path):
    # lead_trail_silence = get_lead_trail_silence_rate(wav_path)
    peak_amplitude = get_peak_amplitude(wav_path)
    loudness = get_loudness(wav_path)
    max_loudness = get_max_loudness(wav_path)
    duration = get_duration(wav_path)

    hnr_mean = get_hnr_mean(wav_path)
    hnr_std = get_hnr_std(wav_path)
    shim = get_shimmer(wav_path)
    jit = get_jitter(wav_path)
    en = get_energy(wav_path)
    pow = get_power(wav_path)
    pitch_mean = get_pitch_mean(wav_path)
    pitch_std = get_pitch_std(wav_path)
    pitch_min = get_pitch_min(wav_path)
    pitch_max = get_pitch_max(wav_path)
    pitch_mas = get_pitch_mas(wav_path)
    vuv_r = get_voiced_unvoiced_rate(wav_path)

    import math
    if math.isnan(peak_amplitude) or math.isnan(loudness) or math.isnan(max_loudness) or math.isnan(duration) or math.isnan(hnr_mean) or math.isnan(hnr_std) or math.isnan(shim) or math.isnan(jit) or math.isnan(en) or math.isnan(pow) or math.isnan(pitch_mean) or math.isnan(pitch_std) or math.isnan(pitch_min) or math.isnan(pitch_max) or math.isnan(pitch_mas) or math.isnan(vuv_r):
        print(wav_path) # /opt/franzi/datasets/ASVspoof2021_LA_eval/flac/LA_E_8762729.flac

    return peak_amplitude, loudness, max_loudness, duration, hnr_mean, hnr_std, shim, jit, en, pow, pitch_mean, pitch_std, pitch_min, pitch_max, pitch_mas, vuv_r

def get_all_names_one_result():
    return ['peak_amplitude', 'loudness', 'max_loudness', 'duration', 'hnr_mean', 'hnr_std', 'shimmer', 'jitter', 'energy', 'power', 'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_mas', 'voiced_unvoiced_rate']

def get_signature_by_name(name):
    """Returns the respective preprocessing function."""
    name = f"get_{name}"
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


if __name__ == '__main__':
    path = "test_audio/LA_E_9999987.flac"
    print(apply_all_signature_one_result(path))


