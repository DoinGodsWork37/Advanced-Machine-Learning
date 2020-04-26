import warnings
from python_speech_features import mfcc
from sklearn.externals import joblib
from pydub import AudioSegment as asg
from pydub.silence import split_on_silence
import os
import scipy.io.wavfile as wavfile
import itertools
import csv
import prep_data

os.chdir(os.path.join(os.getcwd(), "Speech-Recognition/project/"))

num_models = joblib.load("saved_num_models.pkl")

paths = os.listdir("test_data")
paths = [os.path.join("test_data", p) for p in paths]

splits = list(map(prep_data.split_numbers, paths,
                  itertools.repeat(205, len(paths)),
                  itertools.repeat(-70, len(paths))))

# 205, -70


splits_dict = {k: [d[k] for d in splits] for k in splits[0]}

prep_data.save_splits(splits_dict, "test_data_split")


def score_one_word(trained_model, test_file_path):
    # trained_model: ModelHMM object with trained model
    # test_file_path: path to wav file

    sampling_freq, signal = wavfile.read(test_file_path)
    num_cep_coeff = trained_model[2]

    # Extract features
    # Default values:
    # winlen=0.025, winstep=0.01, nfilt=26, nfft=512,
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        features_mfcc = mfcc(signal, sampling_freq, numcep=num_cep_coeff)

        # Calculate log likelihood
        word_score = trained_model[0].compute_score(features_mfcc)
    return word_score


def recognize_words(test_files_folder):
    results = []

    for filename in os.listdir(test_files_folder):

        # Get the name of the subfolder
        current_file = os.path.join(test_files_folder, filename)

        # Extract true label from file name
        start_index = current_file.find('/') + 1
        end_index = current_file.rfind('15')
        true_label = current_file[start_index:end_index]
        print('\n\nTrue label: ', true_label)

        max_score = -float('inf')
        output_label = None

        # Loop through vocabulary models
        for item in num_models:
            trained_model, label = item
            current_score = score_one_word(trained_model, current_file)
            if current_score > max_score:
                max_score = current_score
                output_label = label
            print('\nNext model')
            print('Current label, score: ', label, current_score)
            print('Output label, max score: ', output_label, max_score)
        results.append((true_label, output_label))

    return results


def test_final(test_files_folder):
    filenames_all = []
    phone_numbers_all = []
    # loop for each of the test file
    for filename in sorted(os.listdir(test_files_folder),
                           key=lambda x: int(os.path.splitext(x)[0])):
        # only if it is wave file
        if filename.endswith(".wav") and "_" not in filename:
            # full file path
            current_file = os.path.join(test_files_folder, filename)
            # print("Looping for file ", current_file)

            # In this file, take out all 10 utterances
            all_utterances = asg.from_file(current_file)
            silence_len = 205  # (in ms) minimum length of a silence to be used for a split
            thresh = -70  # (in dBFS) anything quieter than this will be considered silence. default=-16
            separate_utterances = split_on_silence(all_utterances,
                                                   min_silence_len=silence_len,
                                                   silence_thresh=thresh
                                                   )
            # fail-safe
            if len(separate_utterances) != 10:
                thresh = -55  # (in dBFS) anything quieter than this will be considered silence. default=-16
                separate_utterances = split_on_silence(all_utterances,
                                                       min_silence_len=silence_len,
                                                       silence_thresh=thresh
                                                       )
            if len(separate_utterances) != 10:
                raise ValueError("Error in ", filename)

            # string to hold the detected phone number
            output = ""
            # for each of the numeral in this file
            for numeral in separate_utterances:
                # apply all 10 models to this numeral
                numeral_path = "test_attempt" + filename[:-4] + "_delete.wav"
                numeral.export(numeral_path, format="wav")
                max_score = -float("inf")
                output_label = None
                for item in num_models:
                    trained_model, label = item
                    current_score = score_one_word(trained_model, numeral_path)
                    if current_score > max_score:
                        max_score = current_score
                        output_label = label
                output += output_label
                os.remove(numeral_path)
            filenames_all.append(filename)
            phone_numbers_all.append(output)
    return filenames_all, phone_numbers_all


filenames_all, phone_numbers_all = test_final("test_data")

with open("project_submission13.csv", mode="w") as file:
    file_writer = csv.writer(file, delimiter=",", quotechar='"',
                             quoting=csv.QUOTE_MINIMAL)
    for a, b in zip(filenames_all, phone_numbers_all):
        file_writer.writerow([a, b])




