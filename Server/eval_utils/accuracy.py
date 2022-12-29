import os
from itertools import chain, repeat

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from tqdm import tqdm

from common import helpers, config
from common.audio import audio_from_file
from common.clean_audio import clean_wav
from common.dataset import get_data_loader, getListOfFiles, FilelistDataset, SingleAudioDataset
from common.features import FilterbankFeatures


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=True, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=True, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=True, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=True, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def fill_zero(i):
    if len(str(i)) < 4:
        i = '0' + i
        i = fill_zero(i)
    return str(i)


def get_accuracy():
    load_dotenv()
    modelName = os.getenv('model')
    model = None
    device = torch.device(os.getenv('device') if torch.cuda.is_available() else "cpu")
    if modelName == 'quartznet':
        from QuartzNet.model import GreedyCTCDecoder, QuartzNet
    elif modelName == 'jasper':
        from Jasper.model import GreedyCTCDecoder, Jasper

    model_config = "./configs/" + modelName + ".yaml"
    cfg = config.load(model_config)
    symbols = helpers.add_ctc_blank(cfg['labels'])
    _, features_kw = config.input(cfg, 'val')
    feat_proc = FilterbankFeatures(**features_kw)
    wert, cert = 0, 0
    if modelName == 'quartznet':
        model = QuartzNet(encoder_kw=config.encoder(cfg),
                          decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    elif modelName == 'jasper':
        model = Jasper(encoder_kw=config.encoder(cfg),
                       decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    ckpt = "./pretrained/" + modelName + ".pt"
    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location="cpu")
        key = 'ema_state_dict'
        state_dict = checkpoint[key]
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    greedy_decoder = GreedyCTCDecoder()
    if feat_proc is not None:
        feat_proc.to(device)
        feat_proc.eval()
    dataset = torchaudio.datasets.LIBRISPEECH("./datasets/", url="dev-clean", download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    steps = len(dataloader)
    with torch.no_grad():
        for it, batch in enumerate(tqdm(dataloader, initial=1, total=steps)):
            _, _, txt, c, u, i = batch
            fname = './datasets/LibriSpeech/dev-clean/' + str(c.item()) + '/' + str(u.item()) + '/' + str(c.item()) + \
                    '-' + str(u.item()) + '-' + fill_zero(str(i.item())) + '.flac'
            sf = SingleAudioDataset(fname)
            al = get_data_loader(sf,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False)
            ll = chain.from_iterable(repeat(al))
            with torch.no_grad():
                for idt, batch_d in enumerate(tqdm(ll, initial=1, total=len(al))):
                    batch_d = [t.to(device, non_blocking=True) for t in batch_d]
                    audio, audio_lens, _, _ = batch_d
                    feats, feat_lens = feat_proc(audio, audio_lens)
                    if model.encoder.use_conv_masks:
                        log_probs, log_prob_lens = model(feats, feat_lens)
                    else:
                        log_probs = model(feats, feat_lens)

                    preds = greedy_decoder(log_probs)
                    pred = helpers.gather_predictions([preds], symbols)
                    pred = ' '.join(pred)
                    txt = txt[0]
                    print(pred)
                    print(txt)
                    wert += wer(pred, txt)
                    cert += cer(pred, txt)
                    break
            if it + 1 == steps:
                break
    return wert / steps, cert / steps


def get_raw_accuracy():
    load_dotenv()
    modelName = os.getenv('model')
    model = None
    device = torch.device(os.getenv('device') if torch.cuda.is_available() else "cpu")
    if modelName == 'quartznet':
        from QuartzNet.model import GreedyCTCDecoder, QuartzNet
    elif modelName == 'jasper':
        from Jasper.model import GreedyCTCDecoder, Jasper

    model_config = "./configs/" + modelName + ".yaml"
    cfg = config.load(model_config)
    symbols = helpers.add_ctc_blank(cfg['labels'])
    _, features_kw = config.input(cfg, 'val')
    feat_proc = FilterbankFeatures(**features_kw)
    wert, cert = 0, 0
    if modelName == 'quartznet':
        model = QuartzNet(encoder_kw=config.encoder(cfg),
                          decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    elif modelName == 'jasper':
        model = Jasper(encoder_kw=config.encoder(cfg),
                       decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    ckpt = "./pretrained/" + modelName + ".pt"
    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location="cpu")
        key = 'ema_state_dict'
        state_dict = checkpoint[key]
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    greedy_decoder = GreedyCTCDecoder()
    if feat_proc is not None:
        feat_proc.to(device)
        feat_proc.eval()
    dataset = []
    with open('./datasets/BSSpeech/transcript.txt') as f:
        dataset = f.readlines()
    steps = len(dataset)
    with torch.no_grad():
        for it, batch in enumerate(tqdm(dataset, initial=1, total=steps)):
            file, txt = batch.split(", ")
            fname = './datasets/BSSpeech/' + file
            fname = clean_wav(fname)
            sf = SingleAudioDataset(fname)
            al = get_data_loader(sf,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False)
            ll = chain.from_iterable(repeat(al))
            with torch.no_grad():
                for idt, batch_d in enumerate(tqdm(ll, initial=1, total=len(al))):
                    batch_d = [t.to(device, non_blocking=True) for t in batch_d]
                    audio, audio_lens, _, _ = batch_d
                    feats, feat_lens = feat_proc(audio, audio_lens)
                    if model.encoder.use_conv_masks:
                        log_probs, log_prob_lens = model(feats, feat_lens)
                    else:
                        log_probs = model(feats, feat_lens)

                    preds = greedy_decoder(log_probs)
                    pred = helpers.gather_predictions([preds], symbols)
                    pred = ' '.join(pred)
                    print(pred)
                    print(txt)
                    wert += wer(pred, txt)
                    cert += cer(pred, txt)
                    break
            if it + 1 == steps:
                break
    return wert / steps, cert / steps
