from pydub import AudioSegment as asg
from pydub.silence import split_on_silence
import os
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import librosa
import re

# Splitting data
def split_numbers(data_path, silence_len=50, detect_threshold=-30):
    """
    :param path: Path of sound file.
    :param silence_len: (in ms) Minimum length of a silence to be used for a
    split.
    :param detect_threshold: (in dBFS) Anything quieter than this will be considered
    silence. default=-16
    :return: Dictionary of numbers split apart by key. 10 components are
    expected.
    """

    if str(data_path)[7] == "h":
        silence_len = 200
        detect_threshold = -50

    all_utterances = asg.from_file(data_path)
    counter = 0
    separate_utterances = split_on_silence(all_utterances,
                                           min_silence_len=silence_len,
                                           silence_thresh=detect_threshold)

    split_len = len(separate_utterances)

    while split_len != 10 and counter < 100:
        if len(separate_utterances) < 10:
            detect_threshold -= 0.15
        else:
            detect_threshold += 0.15

        separate_utterances = split_on_silence(all_utterances,
                                               min_silence_len=silence_len,
                                               silence_thresh=detect_threshold
                                               )
        counter += 1

        if counter == 1:
            print(f"Trying different thresholds for {data_path}...", end="")
            print(f"silence length: {silence_len}")
            print(f"detect threshold: {detect_threshold}")

    if split_len != 10:
        raise ValueError(f"Separation of sounds made {split_len} components "
                         f"for file {data_path}. "
                         "Try adjusting silence_len and/or detect_threshold "
                         "to accomplish better splitting.")

    if counter > 0:
        print("Threshold found!")

    sequence_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    return dict(zip(sequence_labels, separate_utterances))


def save_splits(sound_dict, split_folder):
    """
    :param sound_dict: Dictionary of sound splits. Keys should represent the
    number split.
    :param split_folder: Top level folder to save split folders
    :return: None. Function used for side effects (saving files to folders)
    """
    split_len = {k: list(range(0, len(v))) for k, v in sound_dict.items()}

    # Create split paths if they do not exist
    for key in sound_dict.keys():
        if not os.path.isdir(split_folder):
            os.mkdir(split_folder)
        if not os.path.isdir(os.path.join(split_folder, key)):
            os.mkdir(os.path.join(split_folder, key))

    for key, num_sequence in split_len.items():
        for num in num_sequence:
            sound_dict[key][num].export(
                os.path.join(split_folder, key, f"number_{key}_{num}.wav"),
                format="wav"
            )

# Augmentation
def plot_waveform(signal, sampling_freq):
    time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)
    plt.plot(time_axis, signal, color="black")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Utterance")
    plt.show()


def stretch(data, rate=1):
    data = data.astype(float)  # time_stretch() accepts floating format
    data = librosa.effects.time_stretch(data, rate)  # stretching

    # Scale it to 16-bit integer values
    scaling_factor = np.power(2, 15) - 1
    data_normalized = data / np.max(np.abs(data))
    data_scaled = np.int16(data_normalized * scaling_factor)

    return data_scaled


def augment(data_path):
    """
    :param data_path: Top level path to sound files.
    :return: Dictionary of augmented sound files.
    """

    p = re.compile("\\w*$")
    m = p.search(data_path).group()

    augment_dict = {m:
                        {
                            "sampling_freq": None,
                            "signal_wn_scaled": None,
                            "signal_roll": None,
                            "signal_stretch_original": None,
                            "sound_stretch80": None
                        }
                    }

    sample_freq_list = list()
    signal_wn_scaled_list = list()
    signal_roll_list = list()
    signal_stretch_original_list = list()
    sound_stretch80_list = list()

    pathlist = [os.path.join(data_path, path) for path in os.listdir(data_path)]

    for path in pathlist:
        if path == f"new_data/{m}/.DS_Store":
            continue
        sampling_freq, signal = wavfile.read(path)
        wn = np.random.randn(len(signal))
        signal_wn = signal + 1000 * wn  # add white noise to signal
        sample_freq_list.append(sampling_freq)

        # Scale it to 16-bit integer values
        scaling_factor = np.power(2, 15) - 1
        signal_wn_normalized = signal_wn / np.max(np.abs(signal_wn))
        signal_wn_scaled = np.int16(signal_wn_normalized * scaling_factor)
        signal_wn_scaled_list.append(signal_wn_scaled)

        signal_roll = np.roll(signal, shift=1000)
        signal_roll_list.append(signal_roll)

        sound_original = asg(data=signal, sample_width=2,
                             frame_rate=sampling_freq,
                             channels=1)
        signal_stretch_80 = stretch(signal, 0.8)

        sound_stretch80 = asg(signal_stretch_80, sample_width=2, frame_rate=int(
            sampling_freq / 0.8), channels=1)
        signal_stretch_original_list.append(sound_original)
        sound_stretch80_list.append(sound_stretch80)

        augment_dict[m]["sampling_freq"] = sample_freq_list
        augment_dict[m]["signal_wn_scaled"] = signal_wn_scaled_list
        augment_dict[m]["signal_roll"] = signal_roll_list
        augment_dict[m]["signal_stretch_original"] = signal_stretch_original_list
        augment_dict[m]["sound_stretch80"] = sound_stretch80_list

    return augment_dict


def save_augments(augment_dict, split_folder):
    wave_file_to_writes = ["signal_roll", "signal_wn_scaled"]

    for k, v in augment_dict.items():
        for k2, v2 in v.items():
            counter = 0
            for list_element in v2:
                path = os.path.join(split_folder, k, f"number_{k}_{k2}_{counter}.wav")
                if k2 is "sampling_freq":
                    continue
                elif k2 in wave_file_to_writes:
                    wavfile.write(path, 8000, list_element)
                    counter += 1
                else:
                    list_element.export(path, format="wav")
                    counter += 1

