import os
from dotenv import load_dotenv

import os
import time

import torch
import torch.distributed as dist
from contextlib import suppress as empty_context

from common import helpers
from common.dataset import AudioDataset, get_data_loader
from common.features import FilterbankFeatures
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.optimizers import lr_policy, Novograd
from common.utils import BenchmarkStats
from common import config

load_dotenv()
modelName = os.getenv('model')
if modelName == 'quartznet':
    from QuartzNet.model import CTCLossNM, GreedyCTCDecoder, QuartzNet
elif modelName == 'jasper':
    from Jasper.model import CTCLossNM, GreedyCTCDecoder, Jasper


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, labels, model,
             ema_model, ctc_loss, greedy_decoder, use_amp):
    for model, subset in [(model, 'dev'), (ema_model, 'dev_ema')]:
        if model is None:
            continue

        model.eval()
        start_time = time.time()
        agg = {'losses': [], 'preds': [], 'txts': []}

        for batch in val_loader:
            batch = [t.cuda(non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch
            feat, feat_lens = val_feat_proc(audio, audio_lens)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs, enc_lens = model(feat, feat_lens)
                loss = ctc_loss(log_probs, txt, enc_lens, txt_lens)
                pred = greedy_decoder(log_probs)

            agg['losses'] += helpers.gather_losses([loss])
            agg['preds'] += helpers.gather_predictions([pred], labels)
            agg['txts'] += helpers.gather_transcripts([txt], [txt_lens], labels)

        wer, loss = process_evaluation_epoch(agg)
        print(f'{epoch}, {step} -- loss: {loss}, wer: {100.0 * wer}, took: {time.time() - start_time}')
        model.train()
    return wer


def main():
    assert (torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    prediction_frequency = 100
    log_frequency = 25
    eval_frequency = 50
    save_frequency = 25
    keep_milestones = [50, 100, 150]
    epochs = 200
    grad_accumulation = 1
    model_config = "./configs/" + modelName + ".yaml"

    world_size = 1

    cfg = config.load(model_config)

    symbols = helpers.add_ctc_blank(cfg['labels'])

    batch_size = 16
    dataset_dir = "datasets/LibriSpeech"
    train_manifests = "datasets/LibriSpeech/librispeech-train-clean-500-wav.json"
    val_manifests = "datasets/LibriSpeech/librispeech-test-clean-wav.json"

    print_once('Setting up datasets...')

    train_dataset_kw, train_features_kw = config.input(cfg, 'train')
    train_dataset = AudioDataset(dataset_dir,
                                 train_manifests,
                                 symbols,
                                 **train_dataset_kw)
    train_loader = get_data_loader(train_dataset,
                                   batch_size,
                                   shuffle=True,
                                   num_workers=4)
    train_feat_proc = FilterbankFeatures(**train_features_kw)

    val_dataset_kw, val_features_kw = config.input(cfg, 'val')
    val_dataset = AudioDataset(dataset_dir,
                               val_manifests,
                               symbols,
                               **val_dataset_kw)
    val_loader = get_data_loader(val_dataset,
                                 batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False)
    val_feat_proc = FilterbankFeatures(**val_features_kw)

    dur = train_dataset.duration / 3600
    dur_f = train_dataset.duration_filtered / 3600
    nsampl = len(train_dataset)
    print_once(f'Training samples: {nsampl} ({dur:.1f}h, '
               f'filtered {dur_f:.1f}h)')

    if train_feat_proc is not None:
        train_feat_proc.cuda()
    if val_feat_proc is not None:
        val_feat_proc.cuda()

    steps_per_epoch = len(train_loader) // grad_accumulation
    model = None
    # set up the model
    if modelName == 'quartznet':
        model = QuartzNet(encoder_kw=config.encoder(cfg),
                          decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    elif modelName == 'jasper':
        model = Jasper(encoder_kw=config.encoder(cfg),
                       decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    model.cuda()
    ctc_loss = CTCLossNM(n_classes=len(symbols))
    greedy_decoder = GreedyCTCDecoder()

    print_once(f'Model size: {num_weights(model) / 10 ** 6:.1f}M params\n')

    # optimization
    kw = {'lr': 1e-3, 'weight_decay': 1e-3}
    optimizer = Novograd(model.parameters(), **kw)

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    adjust_lr = lambda step, epoch, optimizer: lr_policy(
        step, epoch, 1e-3, optimizer, steps_per_epoch=steps_per_epoch,
        warmup_epochs=0, hold_epochs=0,
        num_epochs=epochs, policy='exponential', min_lr=1e-5,
        exp_gamma=0.99)

    ema_model = None

    # load checkpoint
    meta = {'best_wer': 10 ** 6, 'start_epoch': 0}
    checkpointer = Checkpointer('./pretrained', modelName,
                                keep_milestones)

    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()
    torch.cuda.empty_cache()

    bmark_stats = BenchmarkStats()

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        epoch_utts = 0
        epoch_loss = 0
        accumulated_batches = 0
        epoch_start_time = time.time()
        epoch_eval_time = 0

        for batch in train_loader:

            if accumulated_batches == 0:
                step_loss = 0
                step_utts = 0
                step_start_time = time.time()

            batch = [t.cuda(non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch
            feat, feat_lens = train_feat_proc(audio, audio_lens)

            # Use context manager to prevent redundant accumulation of gradients
            if accumulated_batches + 1 < grad_accumulation:
                ctx = model.no_sync()
            else:
                ctx = empty_context()

            with ctx:
                with torch.cuda.amp.autocast(enabled=False):
                    log_probs, enc_lens = model(feat, feat_lens)

                    loss = ctc_loss(log_probs, txt, enc_lens, txt_lens)
                    loss /= grad_accumulation

                reduced_loss = loss

                if torch.isnan(reduced_loss).any():
                    print_once(f'WARNING: loss is NaN; skipping update')
                    continue
                else:
                    step_loss += reduced_loss.item()
                    step_utts += batch[0].size(0) * world_size
                    epoch_utts += batch[0].size(0) * world_size
                    accumulated_batches += 1

                    scaler.scale(loss).backward()

            if accumulated_batches % grad_accumulation == 0:
                epoch_loss += step_loss
                scaler.step(optimizer)
                scaler.update()

                adjust_lr(step, epoch, optimizer)
                optimizer.zero_grad()

                if step % log_frequency == 0:
                    preds = greedy_decoder(log_probs)
                    wer, pred_utt, ref = greedy_wer(preds, txt, txt_lens, symbols)

                    if step % prediction_frequency == 0:
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')

                    step_time = time.time() - step_start_time
                    print(f'Train {epoch} -- loss: {step_loss}, wer: {100.0 * wer}, throughput: {step_utts / step_time}'
                          f', took: {step_time}, lrate: {optimizer.param_groups[0]["lr"]}')

                step_start_time = time.time()

                if step % eval_frequency == 0:
                    tik = time.time()
                    wer = evaluate(epoch, step, val_loader, val_feat_proc,
                                   symbols, model, ema_model, ctc_loss,
                                   greedy_decoder, False)

                    if wer < best_wer and epoch >= 380:
                        checkpointer.save(model, ema_model, optimizer, scaler,
                                          epoch, step, best_wer, is_best=True)
                        best_wer = wer
                    epoch_eval_time += time.time() - tik

                step += 1
                accumulated_batches = 0
                # end of step

            # DALI iterator need to be exhausted;
            # if not using DALI, simulate drop_last=True with grad accumulation
            if step > steps_per_epoch * epoch:
                break

        epoch_time = time.time() - epoch_start_time
        epoch_loss /= steps_per_epoch
        print(f'{epoch} -- train_avg: throughput: {epoch_utts / epoch_time}, took: {epoch_time}, loss: {epoch_loss}')
        bmark_stats.update(epoch_utts, epoch_time, epoch_loss)

        if epoch % save_frequency == 0 or epoch in keep_milestones:
            checkpointer.save(model, ema_model, optimizer, scaler, epoch, step,
                              best_wer)

        if 0 <= epoch - start_epoch:
            print_once(f'Finished after {epoch} epochs.')
            break
        # end of epoch

    print(f'train_avg: {bmark_stats.get(1)}')

    evaluate(None, step, val_loader, val_feat_proc, symbols, model,
             ema_model, ctc_loss, greedy_decoder, False)

    if epoch == epochs:
        checkpointer.save(model, ema_model, optimizer, scaler, epoch, step,
                          best_wer)


if __name__ == "__main__":
    main()
