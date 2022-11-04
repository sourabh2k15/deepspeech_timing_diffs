import numpy as np
import torch
from absl import app
from absl import logging
from pytorch_utils import pytorch_init, pytorch_setup
from model import DeepspeechEncoderDecoder, DeepspeechConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch import nn
from torch.nn import init
import jax
import enum
import time

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
MAX_INPUT_LENGTH = 320000

class ParameterType(enum.Enum):
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM = 3
  EMBEDDING = 4

class ShapeTuple:
  def __init__(self, shape_tuple):
    self.shape_tuple = shape_tuple

def pytorch_param_shapes(model):
  return {k: ShapeTuple(v.shape) for k, v in model.named_parameters()}


def initialize(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
    init.xavier_uniform_(m.weight)
    if m.bias is not None:
      init.constant_(m.bias, 0)
  elif isinstance(m, nn.MultiheadAttention):
    init.xavier_uniform_(m.in_proj_weight)
  for i in m.children():
    initialize(i)


def init_model_fn(rng):
    """Deepspeech model init function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    torch.random.manual_seed(rng[0])
    model = DeepspeechEncoderDecoder(
        DeepspeechConfig(
            feed_forward_dropout_rate=0.1,
            input_dropout_rate=0.1)).eval()
    # Run model once to initialize lazy layers.
    t = MAX_INPUT_LENGTH
    wave = torch.randn((2, t))
    pad = torch.zeros_like(wave)
    _ = model(wave, pad)
    initialize(model)

    param_shapes = pytorch_param_shapes(model)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model

def pytorch_cosine_warmup(optimizer):
    warmup = LinearLR(
        optimizer,
        start_factor=1e-10,
        end_factor=0.02,
        total_iters=5000)
    cosine_steps = max(60000 - 5000, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine_decay],
        milestones=[5000])


def init_optimizer_state(model_params):
    """Creates an AdamW optimizer and a learning rate schedule."""

    epsilon = (1e-8)
    optimizer_state = {
        'optimizer':
            torch.optim.AdamW(
                model_params.parameters(),
                lr=0.02,
                betas=(0.98, 0.99),
                eps=epsilon,
                weight_decay=0.0)
    }

    optimizer_state['scheduler'] = pytorch_cosine_warmup(optimizer_state['optimizer'])
    return optimizer_state

def update_params(model_params, batch, optimizer_state, step, grad_clip):
    current_model = model_params
    optimizer_state['optimizer'].zero_grad()
    
    current_model.train()
    inputs, input_paddings = batch['inputs']
    targets, target_paddings = batch['targets']

    logits, logits_paddings = current_model(inputs.to(DEVICE),
                                    input_paddings.to(DEVICE))

    logprobs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.einsum('bh->b', 1 - logits_paddings).long()
    target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')

    per_seq_loss = ctc_loss(
        logprobs.permute(1, 0, 2),
        targets.long(),
        input_lengths,
        target_lengths).sum()

    l = target_lengths.sum().to(per_seq_loss)
    if USE_PYTORCH_DDP:
      dist.all_reduce(per_seq_loss)
      dist.all_reduce(l)

    loss = per_seq_loss / max(l, 1)
    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            current_model.parameters(), max_norm=grad_clip)

    optimizer_state['optimizer'].step()
    optimizer_state['scheduler'].step()

    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      total_norm = torch.norm(
          torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)

    logging.info('{}) loss = {}, grad_norm = {}'.format(step, loss.item(), total_norm.item()))
    return optimizer_state, current_model, None

def main(_):
    pytorch_init(USE_PYTORCH_DDP, RANK)

     # Loading 1 real batch 
    sharded_padded_batch = np.load('../sharded_padded_batch.npz')

    inputs, input_paddings = sharded_padded_batch['inputs']
    targets, target_paddings = sharded_padded_batch['targets']

    inputs = inputs.reshape(256, -1)[RANK*32: (RANK + 1)*32, :]
    input_paddings = input_paddings.reshape(256, -1)[RANK*32: (RANK + 1)*32, :]
    targets = targets.reshape(256, -1)[RANK*32: (RANK + 1)*32, :]
    target_paddings = target_paddings.reshape(256, -1)[RANK*32: (RANK + 1)*32, :]

    sharded_padded_batch = {
        'inputs': (torch.from_numpy(inputs), torch.from_numpy(input_paddings)),
        'targets': (torch.from_numpy(targets), torch.from_numpy(target_paddings))
    }

    logging.info('loaded librispeech sharded padded batch')
    # print('inputs shape = ', inputs.shape)
    # print('input paddings shape = ', input_paddings.shape)
    # print('targets shape = ', targets.shape)
    # print('target_paddings shape = ', input_paddings.shape)

    
    # Initializing optimizer and LR schedule 
    rng = jax.random.PRNGKey(0)
    model = init_model_fn(rng)
    optimizer_state = init_optimizer_state(model)
    num_training_steps = 100

    start_time = time.time()
    for step in range(num_training_steps):
        optimizer_state, model, _ = update_params(model, sharded_padded_batch, optimizer_state, step, 5.0)

    end_time = time.time()
    print('PyTorch program execution took %s seconds' % (end_time - start_time))

    if USE_PYTORCH_DDP:
        # Cleanup.
        dist.destroy_process_group()

if __name__ == '__main__':
    app.run(main)