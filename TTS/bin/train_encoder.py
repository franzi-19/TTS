#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

from TTS.speaker_encoder.dataset import MyDataset
from TTS.speaker_encoder.losses import GE2ELoss, AngleProtoLoss
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.speaker_encoder.utils import check_config_speaker_encoder
from TTS.speaker_encoder.visuals import plot_embeddings
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.utils.io import save_best_model
from TTS.utils.generic_utils import (
    create_experiment_folder, get_git_branch, remove_experiment_folder,
    set_init_dict)
from TTS.utils.io import copy_config_file, load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import count_parameters
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import NoamLR, check_update

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap, is_val=False, verbose=False, train=True):
    if train:
        dataset = MyDataset(ap,
                            meta_data_train,
                            voice_len=c.dataset_settings["voice_len"],
                            num_utter_per_speaker=c.dataset_settings["num_utters_per_speaker_train"],
                            num_speakers_in_batch=c.dataset_settings["num_speakers_in_batch_train"],
                            skip_speakers=c.dataset_settings["skip_speakers"],
                            storage_size=c.storage["storage_size"],
                            sample_from_storage_p=c.storage["sample_from_storage_p"],
                            additive_noise=c.storage["additive_noise"],
                            feature_type=c.dataset_settings["feature_type"],
                            use_caching=c.dataset_settings["use_caching"],
                            cache_path=c.dataset_settings["cache_path"],
                            dataset_folder=c.dataset_settings["dataset_folder"],
                            verbose=verbose,
                            train=train,
                            codecs=c.compression["codecs"], 
                            prob=c.compression["prob"],
                            trim_silence=c.dataset_settings["trim_silence"])
        c.dataset_settings["num_speakers_in_batch_train"] = len(dataset.speakers) # if number of speakers per batch was adjusted

    else:
        dataset = MyDataset(ap,
                            meta_data_test,
                            voice_len=c.dataset_settings["voice_len"],
                            num_utter_per_speaker=c.dataset_settings["num_utters_per_speaker_test"],
                            num_speakers_in_batch=c.dataset_settings["num_speakers_in_batch_test"],
                            skip_speakers=c.dataset_settings["skip_speakers"],
                            storage_size=c.storage["storage_size"],
                            sample_from_storage_p=c.storage["sample_from_storage_p"],
                            additive_noise=c.storage["additive_noise"],
                            feature_type=c.dataset_settings["feature_type"],
                            use_caching=c.dataset_settings["use_caching"],
                            cache_path=c.dataset_settings["cache_path"],
                            dataset_folder=c.dataset_settings["dataset_folder"],
                            verbose=verbose,
                            train=train,
                            trim_silence=c.dataset_settings["trim_silence"])
        c.dataset_settings["num_speakers_in_batch_test"] = len(dataset.speakers) # if number of speakers per batch was adjusted

    loader = DataLoader(dataset,
                        batch_size=len(dataset.speakers),
                        shuffle=False,
                        num_workers=c.num_loader_workers,
                        collate_fn=dataset.collate_fn)
    return loader

def test(model, batch, criterion, global_step, max_steps):
    model.eval()
    start_time = time.time()

    data = batch
    inputs = data[0]
    labels = data[1]

    # dispatch data to GPU
    if use_cuda:
        inputs = inputs.cuda(non_blocking=True)

    # forward pass model
    outputs = model(inputs)

    x=outputs.view(c.dataset_settings["num_speakers_in_batch_test"],
                        outputs.shape[0] // c.dataset_settings["num_speakers_in_batch_test"], -1)
    loss = criterion(
        outputs.view(c.dataset_settings["num_speakers_in_batch_test"],
                        outputs.shape[0] // c.dataset_settings["num_speakers_in_batch_test"], -1))

    step_time = time.time() - start_time

    _plot_to_tensorboard(global_step, c, tb_logger, outputs, labels, loss.item(), None, None, step_time, None, False)
    _print_to_console(global_step, c, loss.item(), None, None, step_time, None, None, None, max_steps, False)


def train(model, criterion, optimizer, scheduler, ap, global_step, max_steps, best_loss=float('inf')):
    data_loader_train = setup_loader(ap, verbose=True)
    if meta_data_test != []:
        data_loader_test_iter = iter(setup_loader(ap, verbose=True, train=False))

    epoch_time = 0
    avg_loss = 0
    avg_loader_time = 0
    end_time = time.time()
    for _, data in enumerate(data_loader_train):
        model.train()
        if global_step >= max_steps and max_steps != 0:
            break

        start_time = time.time()

        # setup input data
        inputs = data[0] # [500, 100, 40]
        labels = data[1] # [5, 100]
        loader_time = time.time() - end_time
        global_step += 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            inputs = inputs.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

        # forward pass model
        outputs = model(inputs) # [500, 256]

        # loss computation
        loss = criterion(
            outputs.view(c.dataset_settings["num_speakers_in_batch_train"],
                         outputs.shape[0] // c.dataset_settings["num_speakers_in_batch_train"], -1))
        loss.backward()

        grad_norm, _ = check_update(model, c.grad_clip)

        optimizer.step()

        step_time = time.time() - start_time
        epoch_time += step_time

        # Averaged Loss and Averaged Loader Time
        avg_loss = 0.01 * loss.item() \
                   + 0.99 * avg_loss if avg_loss != 0 else loss.item()
        avg_loader_time = 1/c.num_loader_workers * loader_time + \
                          (c.num_loader_workers-1) / c.num_loader_workers * avg_loader_time if avg_loader_time != 0 else loader_time
        current_lr = optimizer.param_groups[0]['lr']

        _plot_to_tensorboard(global_step, c, tb_logger, outputs, labels, avg_loss, current_lr, grad_norm, step_time, avg_loader_time)
        _print_to_console(global_step, c, loss.item(), avg_loss, grad_norm, step_time, loader_time, avg_loader_time, current_lr, max_steps)

        # save best model
        best_loss = save_best_model(model, optimizer, criterion, avg_loss, best_loss,
                                    OUT_PATH, global_step)

        # run model on test set
        if global_step % c.steps_test == 0 and meta_data_test != []:
            test(model, next(data_loader_test_iter), criterion, global_step, max_steps)

        end_time = time.time()

    return avg_loss, global_step

def _plot_to_tensorboard(global_step, c, tb_logger, outputs, labels, avg_loss, current_lr, grad_norm, step_time, avg_loader_time, train=True):
    if train:
        if global_step % c.steps_plot_train == 0:
            train_stats = {
                "loss": avg_loss,
                "lr": current_lr,
                "grad_norm": grad_norm,
                "step_time": step_time,
                "avg_loader_time": avg_loader_time
            }
            tb_logger.tb_train_epoch_stats(global_step, train_stats)
            figures = {
                "UMAP Plot": plot_embeddings(outputs.detach().cpu().numpy(),
                                                c.dataset_settings["num_utters_per_speaker_train"], labels),
            }
            tb_logger.tb_train_figures(global_step, figures)
    else:
        train_stats = {
            "loss": avg_loss,
            "step_time": step_time
        }
        tb_logger.tb_test_epoch_stats(global_step, train_stats)
        embedd = outputs.detach().cpu().numpy()
        figures = {
            "UMAP Plot": plot_embeddings(embedd,
                                            c.dataset_settings["num_utters_per_speaker_test"], labels, max_utter=len(embedd)),
        }
        tb_logger.tb_test_figures(global_step, figures)

def _print_to_console(global_step, c, loss, avg_loss, grad_norm, step_time, loader_time, avg_loader_time, current_lr, max_steps, train=True):
    if train:
        if global_step % c.steps_print_train == 0:
            if max_steps == 0: max_steps = 'Infinity'
            print(
                " > Train: Step:{}/{}  Loss:{:.5f}  AvgLoss:{:.5f}  GradNorm:{:.5f}  "
                "StepTime:{:.2f}  LoaderTime:{:.2f}  AvGLoaderTime:{:.2f}  LR:{:.6f} \n".format(
                    global_step, max_steps, loss, avg_loss, grad_norm, step_time,
                    loader_time, avg_loader_time, current_lr),
                flush=True)
    else:
        print(
            " > Test: Step:{}/{}  Loss:{:.5f}  StepTime:{:.2f} \n".format(
                global_step, max_steps, loss, step_time), flush=True)

def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train
    global meta_data_test

    ap = AudioProcessor(**c.audio)
    model = SpeakerEncoder(input_dim=c.model['input_dim'],
                           proj_dim=c.model['proj_dim'],
                           lstm_dim=c.model['lstm_dim'],
                           num_lstm_layers=c.model['num_lstm_layers'],
                           use_lstm_with_projection=c.model['use_lstm_with_projection'])
    optimizer = RAdam(model.parameters(), lr=c.lr)

    if c.loss == "ge2e":
        criterion = GE2ELoss(loss_method='softmax')
    elif c.loss == "angleproto":
        criterion = AngleProtoLoss()
    else:
        raise Exception("The %s  not is a loss supported" % c.loss)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            print(f" > Restoring Model from {args.restore_path}...")
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            criterion.load_state_dict(checkpoint['criterion'])
        except KeyError:
            print(" > Failed -> Partial model Initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        args.restore_step = checkpoint['step']
        loss = checkpoint['loss']
        print(f" > Model restored from step {checkpoint['step']} with loss {loss}", flush=True)

    else:
        args.restore_step = 0
        loss = float('inf')

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if c.lr_decay:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    # pylint: disable=redefined-outer-name
    meta_data_train, meta_data_test = load_meta_data(c.datasets, c.dataset_settings['dataset_folder'], c.dataset_settings['split_train_data'])

    global_step = args.restore_step
    _, global_step = train(model, criterion, optimizer, scheduler, ap,
                           global_step, c.max_steps, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=True,
                        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument('--output_path',
                        type=str,
                        help='path for training outputs.',
                        default='')
    parser.add_argument('--output_folder',
                        type=str,
                        default='',
                        help='folder name for training outputs.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    check_config_speaker_encoder(c)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    new_fields = {}
    if args.restore_path:
        new_fields["restore_path"] = args.restore_path
    new_fields["github_branch"] = get_git_branch()
    copy_config_file(args.config_path, os.path.join(OUT_PATH, 'config.json'),
                     new_fields)

    LOG_DIR = OUT_PATH
    tb_logger = TensorboardLogger(LOG_DIR, model_name='Speaker_Encoder')

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
