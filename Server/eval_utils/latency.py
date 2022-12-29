import os
import time
from itertools import chain, repeat

from dotenv import load_dotenv

import torch
from tqdm import tqdm

from common import config, helpers
from common.dataset import SingleAudioDataset, get_data_loader
from common.features import FilterbankFeatures


def get_latency():
    load_dotenv()
    modelName = os.getenv('model')
    model = None
    device = torch.device(os.getenv('device') if torch.cuda.is_available() else "cpu")
    if modelName == 'quartznet':
        from QuartzNet.model import QuartzNet
    elif modelName == 'jasper':
        from Jasper.model import Jasper

    model_config = "./configs/" + modelName + ".yaml"
    input_wav = './test.flac'
    cfg = config.load(model_config)
    symbols = helpers.add_ctc_blank(cfg['labels'])
    _, features_kw = config.input(cfg, 'val')
    feat_proc = FilterbankFeatures(**features_kw)
    macs, params, start_time, end_time = None, None, 0, 0
    num_runs, num_warmup_runs = 15, 10
    if modelName == 'quartznet':
        model = QuartzNet(encoder_kw=config.encoder(cfg),
                          decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    elif modelName == 'jasper':
        model = Jasper(encoder_kw=config.encoder(cfg),
                       decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    model.to(device)
    model.eval()

    if feat_proc is not None:
        feat_proc.to(device)
        feat_proc.eval()
    dataset = SingleAudioDataset(input_wav)
    data_loader = get_data_loader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)
    looped_loader = chain.from_iterable(repeat(data_loader))
    steps = len(data_loader)
    with torch.no_grad():
        for it, batch in enumerate(tqdm(looped_loader, initial=1, total=steps)):
            batch = [t.to(device, non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch
            feats, feat_lens = feat_proc(audio, audio_lens)
            for i in range(num_runs):
                if i == num_warmup_runs:
                    start_time = time.time()
                if model.encoder.use_conv_masks:
                    log_probs, log_prob_lens = model(feats, feat_lens)
                else:
                    log_probs = model(feats, feat_lens)

            end_time = time.time()
            break
    total_forward = end_time - start_time
    actual_num_runs = num_runs - num_warmup_runs
    latency = total_forward / actual_num_runs
    return latency
