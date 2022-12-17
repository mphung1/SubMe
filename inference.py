import os
import time
from itertools import chain, repeat

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from common import config
from common import helpers
from common.clean_audio import clean_wav
from common.dataset import (get_data_loader, SingleAudioDataset)
from common.features import FilterbankFeatures
from common.helpers import print_once
from common.mono import convert_to_mono

load_dotenv()


def infer(modelName='quartznet'):

    model = None
    if modelName == 'quartznet':
        from QuartzNet.model import GreedyCTCDecoder, QuartzNet
    elif modelName == 'jasper':
        from Jasper.model import GreedyCTCDecoder, Jasper

    model_config = "./configs/" + modelName + ".yaml"
    transcribe_wav = './uploads/live.wav'
    transcribe_wav = convert_to_mono(transcribe_wav)
    ckpt = "./pretrained/" + modelName + ".pt"

    device = torch.device(os.getenv('device') if torch.cuda.is_available() else "cpu")

    cfg = config.load(model_config)

    symbols = helpers.add_ctc_blank(cfg['labels'])

    assert not cfg['input_val']['audio_dataset'].get('pad_to_max_duration', False)

    dataset = SingleAudioDataset(transcribe_wav)

    data_loader = get_data_loader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)

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

    agg = {'txts': [], 'preds': [], 'logits': []}

    looped_loader = chain.from_iterable(repeat(data_loader))
    greedy_decoder = GreedyCTCDecoder()
    start_time, end_time = 0, 0

    steps = len(data_loader)
    with torch.no_grad():
        for it, batch in enumerate(tqdm(looped_loader, initial=1, total=steps)):
            start_time = time.time()
            batch = [t.to(device, non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch
            feats, feat_lens = feat_proc(audio, audio_lens)

            if model.encoder.use_conv_masks:
                log_probs, log_prob_lens = model(feats, feat_lens)
            else:
                log_probs = model(feats, feat_lens)

            preds = greedy_decoder(log_probs)

            if txt is not None:
                agg['txts'] += helpers.gather_transcripts([txt], [txt_lens],
                                                          symbols)
            agg['preds'] += helpers.gather_predictions([preds], symbols)
            agg['logits'].append(log_probs)

            if it + 1 == steps:
                end_time = time.time()
                break
        latency = end_time - start_time
        # communicate the results
        for idx, p in enumerate(agg['preds']):
            return p, latency

