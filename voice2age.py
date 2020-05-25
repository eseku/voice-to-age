import numpy as np
import sys
import os
import sox
import preprocess
import scipy.io.wavfile as wav
import numpy as np


from mfcc import *
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from collections import Counter


def compute_mel_log(file_name):
    print (file_name)
    rate, data = wav.read(file_name);
    mfcc = MFCC(nfilt = 40, ncep = 13, samprate = rate,
                wlen = 0.0256, frate = 100,
                lowerf=133.33334, upperf=6855.4976)
    mel_log = mfcc.sig2logspec(data)
    return mel_log


if __name__ == "__main__":
    argv = sys.argv[1:]
     
    if(len(argv) != 1):
        print("Usage: python voice2age.py <full path to wav file>")
        sys.exit()
    else:
        model = load_model("age_range_classifier.h5")
        tfm = sox.Transformer()
         
        tfm.convert(samplerate=16000)
        tfm.build(argv[0],'downsampled.wav')
         
        mel_log = compute_mel_log('downsampled.wav')
        os.remove('downsampled.wav')
        
        preprocessed_x = preprocess.preprocess(mel_log,9)
         
        age = model.predict(preprocessed_x)
        print(age)
