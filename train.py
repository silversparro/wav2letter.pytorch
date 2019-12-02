import argparse
import errno
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from torch.autograd import Variable
from tqdm import tqdm
from warpctc_pytorch import CTCLoss
from apex import amp
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.distributed import DistributedDataParallel
from decoder import GreedyDecoder
from model import WaveToLetter
import Levenshtein as Lev
from wav2letter.criterion import ASGLoss,CriterionScaleMode

parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='~/data/train.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='~/data/validation.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=0, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--peak-normalization',dest='peak_normalization', default=False, action='store_true', help='Apply peak normalization while training and validation')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True,dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--usePcen', default=True,dest='pcen', action='store_true', help='Use pcen features')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--mixPrec',dest='mixPrec',default=False,action='store_true', help='use mix precision for training')
parser.add_argument('--reg-scale', dest='reg_scale', default=0.9, type=float, help='L2 regularizationScale')
parser.add_argument('--momentum', default=0.90, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.2, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', default=True,dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=5000, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', default=True,dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/w2lOnMozillaDataAftr118Epch', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--seed', default=1234 )
parser.add_argument('--id', default='Wav2Letter training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='~/models/wave2Letter', help='Location to save epoch models')
parser.add_argument('--model-path', default='~/models/wave2Letter/wav2Letter_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', default=False,dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', default=False ,dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.9, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.1,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.7,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def werCalc(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

def cerCalc(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def poly_lr_scheduler(init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return 0
    lr = init_lr*(1 - float(iter)/float(max_iter))**power
    return lr


if __name__ == '__main__':
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mixPrec and not args.cuda:
        raise ValueError('If using mixed precision training, CUDA must be enabled!')
    args.distributed = args.world_size > 1
    main_proc = True
    # device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None
    optim_state = None
    if args.visdom and main_proc:
        from visdom import Visdom

        viz = Visdom()
        opts = dict(title=args.id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        viz_window = None
        epochs = torch.arange(1, args.epochs + 1)
    if args.tensorboard and main_proc:
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Tensorboard log directory already exists.')
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        raise
            else:
                raise
        from tensorboardX import SummaryWriter

        tensorboard_writer = SummaryWriter(args.log_dir)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise
    # criterion = CTCLoss()

    avg_loss, start_epoch, start_iter = 0, 0, 0
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = WaveToLetter.load_model_package(package)
        audio_conf = WaveToLetter.get_audio_conf(model)
        labels = WaveToLetter.get_labels(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)

        if args.noise_dir is not None:
            model = WaveToLetter.setAudioConfKey(model,'noise_dir',args.noise_dir)
            model = WaveToLetter.setAudioConfKey(model,'noise_prob',args.noise_prob)
            model = WaveToLetter.setAudioConfKey(model,'noise_max',args.noise_max)
            model = WaveToLetter.setAudioConfKey(model,'noise_min',args.noise_min)

        if not args.finetune:  # Don't want to restart training
            # if args.cuda:
            #     model.cuda()
            optim_state = package['optim_dict']
            # optimizer.load_state_dict(package['optim_dict'])

            # Temporary fix for pytorch #2830 & #1442 while pull request #3658 in not incorporated in a release
            # TODO : remove when a new release of pytorch include pull request #3658
            # if args.cuda:
            #     for state in optimizer.state.values():
            #         for k, v in state.items():
            #             if torch.is_tensor(v):
            #                 state[k] = v.cuda()

            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            loss_results, cer_results, wer_results = package['loss_results'], package[
                'cer_results'], package['wer_results']
            if main_proc and args.visdom and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
                x_axis = epochs[0:start_epoch]
                y_axis = torch.stack(
                    (loss_results[0:start_epoch], wer_results[0:start_epoch], cer_results[0:start_epoch]),
                    dim=1)
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            if main_proc and args.tensorboard and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
                for i in range(start_epoch):
                    values = {
                        'Avg Train Loss': loss_results[i],
                        'Avg WER': wer_results[i],
                        'Avg CER': cer_results[i]
                    }
                    tensorboard_writer.add_scalars(args.id, values, i + 1)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        model = WaveToLetter(labels=labels,
                           audio_conf=audio_conf,sample_rate=args.sample_rate,window_size=args.window_size,mixed_precision=args.mixPrec)
        # parameters = model.parameters()
        # optimizer = torch.optim.SGD(parameters, lr=args.lr,
        #                             momentum=args.momentum, nesterov=True)

    decoder = GreedyDecoder(labels)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, peak_normalization=args.peak_normalization, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, peak_normalization=args.peak_normalization, augment=False)

    criterion = ASGLoss(len(labels), scale_mode=CriterionScaleMode.TARGET_SZ_SQRT).to(device)
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)
    if args.distributed:
        model = DistributedDataParallel(model)
    # if args.cuda and not args.distributed:
    #     model = torch.nn.DataParallel(model).cuda()
    # elif args.cuda and args.distributed:
    #     model.cuda()
    #     model = DistributedDataParallel(model)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    if args.mixPrec:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    print(model)
    print("Number of parameters: %d" % WaveToLetter.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    globalStep = 0
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes, inputFilePaths, inputsMags = data
            grad = torch.ones(inputsMags.size()[0], dtype=torch.float16, device=device)
            globalStep +=1
            if not args.pcen:
                inputsMags = inputs
            # measure data loading time
            data_time.update(time.time() - end)
            inputsMags = Variable(inputsMags, requires_grad=True)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            inputsMags = inputsMags.to(device)
            # if args.cuda:
            #     inputsMags = inputsMags.cuda()
                # targets = targets.cuda()
                # target_sizes= target_sizes.cuda()
            out = model(inputsMags)
            # out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            # out = out.cpu()
            loss = criterion(out, targets,target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_sum = loss.data.sum()
            inf = float("inf")

            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]


            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))
            # compute gradient
            optimizer.zero_grad()
            if args.mixPrec:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.clip_master_grads(args.max_norm)
            else:
                loss.backward(grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()
            # if args.cuda:
            #     torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            # if losses.val <0.01:
            #     out = out.transpose(0,1)
            #     decoded_output, _ = decoder.decode(out.data, sizes)
            #     for numFile,idxP in enumerate(decoded_output):
            #         print(idxP),inputFilePaths[numFile][1]
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/wav2Letter_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(WaveToLetter.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
                           file_path)
            del loss
            del out
            torch.cuda.empty_cache()

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        if (epoch+1)%1==0:
            print ("coming into test loop")
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, input_percentages, target_sizes, inputFilePaths, inputsMags = data
                if not args.pcen:
                    inputsMags = inputs

                inputsMags = Variable(inputsMags, volatile=True)
                inputsMags = inputsMags.to(device)


                # unflatten targets
                split_targets = []
                offset = 0
                target_strings = []
                for idxInp,size in enumerate(target_sizes):
                    target_strings.append(inputFilePaths[idxInp][1])
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                # if args.cuda:
                #     inputsMags = inputsMags.cuda()

                out = model(inputsMags)  # NxTxH
                seq_length = out.size(1)
                sizes = input_percentages.mul_(int(seq_length)).int()

                decoded_output, _ = decoder.decode(out.data, sizes)
                # target_strings = decoder.convert_to_strings(split_targets)

                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x]
                    print ('transcript : {}, reference :{} , filePath : {}'.format(transcript,reference,inputFilePaths[x][0]))
                    # print 'reference : {}'.format(reference)
                    try:
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))
                    except Exception as e:
                        print ('encountered exception {}'.format(e))
                total_cer += cer
                total_wer += wer

                if args.cuda:
                    torch.cuda.synchronize()
                del out
                torch.cuda.empty_cache()
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        if args.visdom and main_proc:
            x_axis = epochs[0:epoch + 1]
            y_axis = torch.stack((loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
            if viz_window is None:
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=viz_window,
                    update='replace',
                )
        if args.tensorboard and main_proc:
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
        if args.checkpoint and main_proc:
            file_path = '%s/wav2Letter_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(WaveToLetter.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                       file_path)
        if (epoch + 1) % 1 == 0:
        #     # anneal lr

            optim_state = optimizer.state_dict()
            lrToUse = optim_state['param_groups'][0]['lr'] / args.learning_anneal
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
            optim_state['param_groups'][0]['lr'] = lrToUse
            optimizer.load_state_dict(optim_state)
            print('Learning rate annealed to: {lr:.15f}'.format(lr=optim_state['param_groups'][0]['lr']))

        # if (best_wer is None or best_wer > wer) and main_proc:
        if main_proc:
            # print("Found better validated model, saving to %s" % args.model_path)
            torch.save(WaveToLetter.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results)
                       , args.model_path)

            best_wer = wer

        avg_loss = 0

