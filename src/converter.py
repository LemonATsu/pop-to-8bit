import os
import numpy as np
import librosa
import scipy.io as spio
from scipy.signal import hamming
from nmf import nmf
from pyin import pYIN
from svs import svs 

class 8BitConverter(object):

    def __init__(self, wave, fs=44100., v_centered=True):

