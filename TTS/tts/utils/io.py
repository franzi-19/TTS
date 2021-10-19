import os
import torch
import datetime
import pickle as pickle_tts
import fsspec
from typing import Any

from TTS.utils.io import RenamingUnpickler


def load_checkpoint(model, checkpoint_path, amp=None, use_cuda=False):
    try:
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'), pickle_module=pickle_tts)
    model.load_state_dict(state['model'])
    if amp and 'amp' in state:
        amp.load_state_dict(state['amp'])
    if use_cuda:
        model.cuda()
    # set model stepsize
    if hasattr(model.decoder, 'r'):
        model.decoder.set_r(state['r'])
    return model, state


def save_model(model, optimizer, current_step, epoch, r, output_path, amp_state_dict=None, **kwargs):
    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': r
    }
    if amp_state_dict:
        state['amp'] = amp_state_dict
    state.update(kwargs)
    torch.save(state, output_path)


def save_checkpoint(model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    file_name = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print(" > CHECKPOINT : {}".format(checkpoint_path))
    save_model(model, optimizer, current_step, epoch, r, checkpoint_path, **kwargs)


# def save_best_model(target_loss, best_loss, model, optimizer, current_step, epoch, r, output_folder, **kwargs):
#     if target_loss < best_loss:
#         file_name = 'best_model.pth.tar'
#         checkpoint_path = os.path.join(output_folder, file_name)
#         print(" >> BEST MODEL : {}".format(checkpoint_path))
#         save_model(model, optimizer, current_step, epoch, r, checkpoint_path, model_loss=target_loss, **kwargs)
#         best_loss = target_loss
#     return best_loss

def save_best_model(model, optimizer, criterion, model_loss, best_loss, out_path, current_step):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            "model": new_state_dict,
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "step": current_step,
            "loss": model_loss,
            "date": datetime.date.today().strftime("%B %d, %Y"),
        }
        best_loss = model_loss
        bestmodel_path = "best_model.pth.tar"
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("> BEST MODEL ({0:.5f}) : {1:} \n".format(model_loss, bestmodel_path))
        save_fsspec(state, bestmodel_path)
    return best_loss

def save_fsspec(state: Any, path: str, **kwargs):
    """Like torch.save but can save to other locations (e.g. s3:// , gs://).
    Args:
        state: State object to save
        path: Any path or url supported by fsspec.
        **kwargs: Keyword arguments forwarded to torch.save.
    """
    with fsspec.open(path, "wb") as f:
        torch.save(state, f, **kwargs)
