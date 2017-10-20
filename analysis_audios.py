
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sys import argv
import os
import glob

for filename in glob.glob('audio/*.wav'):
    [fs, x] = wavfile.read(filename)
    # compute the spectrogram
    f, t, Sxx = signal.spectrogram(x, fs)
# plots them
    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.plot(Sxx)
    plt.show(block=True)
    plt.interactive(False)
