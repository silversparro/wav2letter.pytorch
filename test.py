import argparse
import time
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from decoder import GreedyDecoder
import torch
from data.data_loader import SpectrogramDataset, AudioDataLoader
from model import WaveToLetter
import torch.quantization
# from torch.quantization import QuantStub, DeQuantStub
np.random.seed(123456)

parser = argparse.ArgumentParser(description='Wav2Letter transcription')
parser.add_argument('--model-path', default='/media/yoda/gargantua/data_pb/models/agent/wave2Letter39WERWithGreedyDecoder/deepspeech2_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', default=True, action="store_true", help='Use cuda to test model')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/media/evaData/clientData/pb/speechRecognition/trainingFiles/agent/valSubSet.csv')
parser.add_argument('--batch-size', default=10, type=int, help='Batch size for training')
parser.add_argument('--fuse-layers', default=True, action="store_true"
					, help='if True then combine all the CONV-BN layer to increase the speed of network. W/o Decreasing the accuracy.')
parser.add_argument('--num-workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', default=False, action="store_true", help="print out decoded output and error of each sample")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm-path', default="", type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.75, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1.0, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm-workers', default=4, type=int, help='Number of LM processes to use')
parser.add_argument('--mixPrec', default=False,dest='mixPrec', action='store_true', help='Use mix prec for inference even if it was not avail for training.')

parser.add_argument('--usePCEN', default=True,dest='usePcen', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

if __name__ == '__main__':
	torch.set_grad_enabled(False)
	device = torch.device("cuda" if args.cuda else "cpu")
	# device = torch.device("cpu")
	model = WaveToLetter.load_model(args.model_path, cuda=args.cuda)
	if args.fuse_layers:
		model.module.convertTensorType()
	model = model.to(device)
	model.eval()
	avgTime = []
	labels = WaveToLetter.get_labels(model)
	audio_conf = WaveToLetter.get_audio_conf(model)
	# model.module.fuse_model()
	# model.qconfig = torch.quantization.default_qconfig
	# torch.quantization.prepare(model, inplace=True)
	if args.decoder == "beam":
		from decoder import BeamCTCDecoder

		decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
								 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
								 beam_width=args.beam_width, num_processes=args.lm_workers)
	elif args.decoder == "greedy":
		decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
	else:
		decoder = None
	target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
	test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
									  normalize=True)
	test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
								  num_workers=args.num_workers)
	total_cer, total_wer = 0, 0
	output_data = []
	for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
		inputs, targets, input_percentages, target_sizes, inputfilePaths,inputsMags = data

		inputsMags = Variable(inputsMags, volatile=True)
		inputs = Variable(inputs, volatile=True)
		# if model.mixed_precision:
		# inputs = inputs.half()
		# unflatten targets
		split_targets = []
		offset = 0
		for size in target_sizes:
			split_targets.append(targets[offset:offset + size])
			offset += size

		if args.cuda:
			inputs = inputs.to(device)
			inputsMags = inputsMags.to(device)
		beforeInferenceTime = time.time()
		if args.usePcen:
			out = model(inputsMags)
		else:
			out = model(inputs)
		afterInferenceTime = time.time()
		# out = out.transpose(0, 1)  # TxNxH
		seq_length = out.size(1)
		sizes = input_percentages.mul_(int(seq_length)).int()

		if decoder is None:
			# add output to data array, and continue
			output_data.append((out.data.cpu().numpy(), sizes.numpy()))
			continue
		beforeDecoderTime = time.time()
		avgTime.append(afterInferenceTime - beforeInferenceTime)
		try:
			decoded_output, _, = decoder.decode(out.data, sizes)
		except Exception as e:
			continue
		target_strings = target_decoder.convert_to_strings(split_targets)
		wer, cer = 0, 0
		afterDecoderTime = time.time()
		
		print ('inferenceTime Total {}, only decodingTime {}, model outputTime {}'
				''.format((afterDecoderTime-beforeInferenceTime),
						 (afterDecoderTime-beforeDecoderTime),(afterInferenceTime-beforeInferenceTime)))
		for x in range(len(target_strings)):
			transcript, reference = decoded_output[x][0], target_strings[x][0]
			wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
			cer_inst = decoder.cer(transcript, reference) / float(len(reference))
			wer += wer_inst
			cer += cer_inst
			if args.verbose:
				print("Ref:", reference.lower())
				print("Hyp:", transcript.lower())
				print("WER:", wer_inst, "CER:", cer_inst, "\n")
		total_cer += cer
		total_wer += wer
	temp = (i+1)*args.batch_size
	if args.verbose:
		print("average_wer: ", total_wer/temp,"average_cer:", total_cer/temp)

	if decoder is not None:
		wer = total_wer / len(test_loader.dataset)
		cer = total_cer / len(test_loader.dataset)

		print('Test Summary \t'
			  'Average WER {wer:.3f}\t'
			  'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
	else:
		np.save(args.output_path, output_data)
	avgTimeVal = float(sum(avgTime)) / len(avgTime)
	print ('avg time to run inference is {}'.format(avgTimeVal))