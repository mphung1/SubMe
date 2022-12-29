from thop import clever_format

from eval_utils import flops, latency, accuracy
import os

flops, params = flops.get_flops()
print(str(clever_format([flops, params], "%.3f")))
latency = latency.get_latency()
flops = flops / latency
flops = clever_format([flops], "%.3f")
ac = accuracy.get_accuracy()
raw_accuracy = accuracy.get_raw_accuracy()
print('FLOPs: ' + str(flops) + ' per second \nLatency: ' + str(latency * 1000) + 'ms')
print('LibriSpeech WER: ' + str(ac[0] * 100) + '% \nLibriSpeech CER: ' + str(ac[1] * 100) + '%')
print('Raw WER: ' + str(raw_accuracy[0] * 100) + '% \nRaw CER: ' + str(raw_accuracy[1] * 100) + '%')

