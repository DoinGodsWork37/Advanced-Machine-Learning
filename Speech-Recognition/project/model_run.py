import os
import warnings
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm      # hidden Markov model
from python_speech_features import mfcc
from sklearn.externals import joblib

class ModelHMM(object):
    def __init__(self, num_components=8, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter
    # Define the covariance type and the type of HMM:
        self.cov_type = "diag"
        self.model_name = "GaussianHMM"
    # Initialize the variable in which we will store the models for each word:
        self.models = []
    # Define the model using the specified parameters:
        self.model = hmm.GaussianHMM(n_components=self.n_components,
                                     covariance_type=self.cov_type, n_iter=self.n_iter)

    # Define a method to train the model
    # 'training_data' is a 2D numpy array where each row has length of number of mfcc coefficients
    def train(self, training_data):
        np.seterr(all="ignore")
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)

    # Define a method to compute log likelihood score for input features
    def compute_score(self, input_data):
        return self.model.score(input_data)  # model.score returns log likelihood of sample input_data


def build_one_model(input_folder, num_states, num_cep_coeff):
    # input_folder: path to the folder containing training wav files with the word
    # num_states: number of hidden states in HMM
    # num_cep_coeff: number of MFCC features extracted from each time window

    X = np.array([])  # features

    training_files = [x for x in os.listdir(input_folder) if x.endswith(".wav")]

    for filename in training_files:
        # Extract the current filepath and read the file
        filepath = os.path.join(input_folder, filename)
        sampling_freq, signal = wavfile.read(filepath)

        # Extract features
        # Default values:
        # winlen=0.025, winstep=0.01, nfilt=26, nfft=512,
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_mfcc = mfcc(signal, sampling_freq, numcep=num_cep_coeff)

        # Append features to the variable X
        if len(X) == 0:
            X = features_mfcc
        else:
            X = np.append(X, features_mfcc, axis=0)

    # Initiate HMM model object
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = ModelHMM(num_components=num_states)

    # Train HMM model, calculate likelihood of the sample by the trained model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model.train(X)
        model_score = model.compute_score(X)

    return model, model_score, num_cep_coeff


def build_models(input_folder):
    # input_folder contains subfolders with samples of words in wav files

    # Initialize the variable to store all the models
    speech_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):

        # Get name of subfolder
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
            continue

        # Extract label
        label = subfolder[subfolder.rfind("/") + 1:]

        # Fit model for label
        model = build_one_model(subfolder, num_states=17, num_cep_coeff=17)

        # Add the model to the list
        speech_models.append((model, label))

        # Reset model variable
        model = None

    return speech_models

num_models = build_models("new_data/")

joblib.dump(num_models, "saved_num_models.pkl")

