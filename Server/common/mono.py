import os

from pydub import AudioSegment


def convert_to_mono(file):
    ext = os.path.splitext(file)[1]
    ext = ext.replace('.', '')
    if ext == "wav":
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        file = os.path.splitext(file)[0]+'1.'+ext
        sound.export(file, format=ext)
    else:
        print("Only WAV stereo is supported")
    return file
