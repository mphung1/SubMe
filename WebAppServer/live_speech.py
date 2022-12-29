# Import Module
from tkinter import *
from threading import *

import os
from itertools import chain, repeat

from dotenv import load_dotenv

import torch
import pyaudio
import asyncio
import wave

from common import config
from common import helpers
from common.dataset import SingleAudioDataset, get_data_loader
from common.features import FilterbankFeatures

load_dotenv()
modelName = os.getenv('model')
model = None
if modelName == 'quartznet':
    from QuartzNet.model import GreedyCTCDecoder, QuartzNet
elif modelName == 'jasper':
    from Jasper.model import GreedyCTCDecoder, Jasper

model_config = "./configs/" + modelName + ".yaml"
ckpt = "./pretrained/" + modelName + ".pt"

device = torch.device(os.getenv('device') if torch.cuda.is_available() else "cpu")

cfg = config.load(model_config)

symbols = helpers.add_ctc_blank(cfg['labels'])

assert not cfg['input_val']['audio_dataset'].get('pad_to_max_duration', False)

_, features_kw = config.input(cfg, 'val')
feat_proc = FilterbankFeatures(**features_kw)

if modelName == 'quartznet':
    model = QuartzNet(encoder_kw=config.encoder(cfg),
                      decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
elif modelName == 'jasper':
    model = Jasper(encoder_kw=config.encoder(cfg),
                   decoder_kw=config.decoder(cfg, n_classes=len(symbols)))

if ckpt is not None:
    checkpoint = torch.load(ckpt, map_location="cpu")
    key = 'ema_state_dict'
    state_dict = checkpoint[key]
    model.load_state_dict(state_dict, strict=True)

model.to(device)
model.eval()

if feat_proc is not None:
    feat_proc.to(device)
    feat_proc.eval()

p = pyaudio.PyAudio()

stream = p.open(
    rate=96000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,  # input stream flag
)

greedy_decoder = GreedyCTCDecoder()


async def send_receive():
    global label

    async def send():
        while True:
            data = stream.read(3 * 96000)
            output_filename = './live.wav'
            wav_file = wave.open(output_filename, 'w')
            wav_file.setnchannels(1)  # number of channels
            wav_file.setsampwidth(2)  # sample width in bytes
            wav_file.setframerate(96000)
            wav_file.writeframes(data)
            await receive()

        return True

    async def receive():
        global label
        result_str = infer()
        label["text"] = result_str

    await asyncio.gather(send(), receive())


def infer():
    with torch.no_grad():
        file = './live.wav'
        dataset = SingleAudioDataset(file)
        data_loader = chain.from_iterable(repeat(get_data_loader(dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 drop_last=False)))
        for batch in data_loader:
            batch = [t.to(device, non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch
            feats, feat_lens = feat_proc(audio, audio_lens)

            if model.encoder.use_conv_masks:
                log_probs, log_prob_lens = model(feats, feat_lens)
            else:
                log_probs = model(feats, feat_lens)

            preds = greedy_decoder(log_probs)
            pred = helpers.gather_predictions([preds], symbols)
            break
        # communicate the results
        return pred[0]


# Create Object
root = Tk()
label = Label(root, text="Welcome to ASR-AI Walker Module", font='Aerial 18')
label.pack()
# Set geometry
root.geometry("400x400")

# use threading
def threading():
    # Call work function
    t1 = Thread(target=work)
    t1.start()

# work function
def work():
    while True:
        asyncio.run(send_receive())

# Create Button
Button(root, text="Start Transcribing", command=threading).pack()

# Execute Tkinter
root.mainloop()
