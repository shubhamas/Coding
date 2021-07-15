import numpy as np
import scipy
from scipy.io import wavfile
import glob
from numpy.fft import fft, fftshift

path = glob.glob("E:/Coding/ml/q2/*.wav")
for files in path:
    sr, data = wavfile.read(files)

print(sr)
print(data)
print(data.shape)

from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# read audio samples
for files in path:
    input_data = read(files)
    # print(input_data)
    audio = input_data[1]
    # print(input_data[1].shape)

    # plot the first 1024 samples
        # plt.plot(audio[0:1024])
        # # label the axes
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # # set the title  
        # plt.title("Sample Wav")
        # # display the plot
        # plt.show()

    # Applying hamming window 

    # window = np.hamming()  
    # print(audio[0:399])
    for i in range(0,298):
        audio  =  audio[0:400]         
        window = np.hamming(400)
        audio = audio * window
        A = fft(window, 256)
        A = A[0:128] 
        A_mag = np.abs(A)
        log_mag = np.log10(A_mag)
        # print(A)
        print(A.shape)

















