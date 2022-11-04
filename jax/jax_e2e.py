import model
import jax
import numpy as np
import functools
import jax.numpy as jnp 
import flax
import flax.linen as nn
from flax import jax_utils
import optax
from absl import logging
import jax.lax as lax
import time

from absl import app
from absl import flags
from absl import logging

_GRAD_CLIP_EPS = 1e-6

def load_batch():
    # Loading 1 real batch 
    sharded_padded_batch = np.load('sharded_padded_batch.npz')

    inputs, input_paddings = sharded_padded_batch['inputs']
    targets, target_paddings = sharded_padded_batch['targets']

    print('loaded librispeech sharded padded batch')
    print('inputs shape = ', inputs.shape)
    print('input paddings shape = ', input_paddings.shape)
    print('targets shape = ', targets.shape)
    print('target_paddings shape = ', target_paddings.shape)

    sharded_padded_batch = {
        'inputs': (inputs, input_paddings),
        'targets': (targets, target_paddings)
    }

    return sharded_padded_batch

# Initing optimizer and LR schedule
def jax_cosine_warmup():
  # Create learning rate schedule.
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=0.02,
      transition_steps=5000)
  cosine_steps = max(60000 - 5000, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=0.02, decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[500])
  return schedule_fn

def init_optimizer_state(params):
  """Creates an AdamW optimizer and a learning rate schedule."""

  lr_schedule_fn = jax_cosine_warmup()

  # Create optimizer.
  epsilon = (1e-8)
  opt_init_fn, opt_update_fn = optax.adamw(
      learning_rate=lr_schedule_fn,
      b1=0.98,
      b2=0.99,
      eps=epsilon,
      weight_decay=0.0)
  optimizer_state = opt_init_fn(params)

  return jax_utils.replicate(optimizer_state), opt_update_fn


# Defining pmapped update step
@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1))
def pmapped_train_step(model_class,
                       opt_update_fn,
                       batch_stats,
                       optimizer_state,
                       params,
                       batch,
                       rng,
                       grad_clip):

  def _loss_fn(params):
    """Loss function used for training."""
    inputs, input_paddings = batch['inputs']
    targets, target_paddings = batch['targets']

    (logits, logit_paddings), new_batch_stats = model_class.apply(
        {'params': params, 'batch_stats': batch_stats},
        inputs,
        input_paddings,
        train=True,
        rngs={'dropout' : rng},
        mutable=['batch_stats'])

    logprobs = nn.log_softmax(logits)
    per_seq_loss = optax.ctc_loss(logprobs,
                                 logit_paddings,
                                 targets,
                                 target_paddings)
    normalizer = jnp.sum(1 - target_paddings)
    normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)

    return normalized_loss, new_batch_stats

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, new_batch_stats), grad = grad_fn(params)
  (loss, grad) = lax.pmean((loss, grad), axis_name='batch')
  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               params)
  updated_params = optax.apply_updates(params, updates)
  return new_optimizer_state, updated_params, new_batch_stats, jnp.mean(loss), jnp.mean(grad_norm)


def main(_):
    sharded_padded_batch = load_batch()
    
    # Initing model
    config = model.DeepspeechConfig()
    model_class = model.Deepspeech(config)
    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng, 2)

    model_init_fn = jax.jit(functools.partial(model_class.init, train=False))
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    print('Initializing model.')
    vars = model_init_fn({'params': params_rng, 'dropout': dropout_rng}, *fake_input_batch)
    batch_stats, params = vars.pop('params')

    print('Initializing optimizer')
    replicated_optimizer_state, opt_update_fn = init_optimizer_state(params)
    replicated_params = jax_utils.replicate(params)
    replicated_batch_stats = jax_utils.replicate(params)

    # Starting Training to measure time: 

    num_training_steps = 10
    grad_clip=5.0

    print('Starting training')
    print('JAX local device count = ', jax.local_device_count())
    start_time = time.time()
    for step in range(num_training_steps):
        (replicated_optimizer_state, 
        replicated_params, 
        replicated_batch_stats, 
        loss, 
        grad_norm) = pmapped_train_step(
            model_class,
            opt_update_fn,
            replicated_batch_stats,
            replicated_optimizer_state,
            replicated_params,
            sharded_padded_batch,
            rng,
            grad_clip)
        
        print('{}) loss = {} grad_norm = {}'.format(step, loss[0], grad_norm[0]))

    end_time = time.time()
    print('JAX program execution took %s seconds' % (end_time - start_time))

if __name__ == '__main__':
    app.run(main)