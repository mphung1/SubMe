import os

from scipy.io import wavfile
import noisereduce as nr


def clean_wav(file):
    ext = os.path.splitext(file)[1]
    ext = ext.replace('.', '')
    if ext == "wav":
        rate, data = wavfile.read(file)
        reduced_noise = nr.reduce_noise(y=data, sr=16000,
                                        freq_mask_smooth_hz=500,
                                        chunk_size=600000,
                                        padding=30000,
                                        n_fft=1024,
                                        prop_decrease=1.0,
                                        thresh_n_mult_nonstationary=2,
                                        sigmoid_slope_nonstationary=10,
                                        n_std_thresh_stationary=1.5, )
        file = os.path.splitext(file)[0] + '1.' + ext
        wavfile.write(file, rate, reduced_noise)
    else:
        print("Only WAV cleanup is supported")
    return file
