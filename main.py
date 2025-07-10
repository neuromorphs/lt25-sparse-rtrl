import os
from datetime import datetime
import random
import pickle
from functools import partial
import argparse

from tqdm import tqdm
import ipdb
from ipdb import launch_ipdb_on_exception
import numpy as np
import yaml
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import haliax

import jaxpruner
import ml_collections

import wandb

from dataloaders.dataloading import create_speechcommands35_classification_dataset, create_toy_classification_dataset
# from data import get_data, dataloader
from models import RNN, CellType

from typing import Any, Tuple

def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps

@eqx.filter_jit
def loss_fn(y, pred_y):
    one_hot_label = jax.nn.one_hot(y, num_classes=pred_y.shape[1])
    l =  -np.sum(one_hot_label * jnp.log(pred_y))
    # # Trains with respect to binary cross-entropy
    # l = -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    return l, l

@eqx.filter_jit
def compute_loss_and_outputs(model, x, y):
    pred_y, outs = jax.vmap(model)(x)
    # jax.debug.print("{pred_y}", pred_y=pred_y)
    ## Trains with respect to binary cross-entropy
    loss, _ = loss_fn(y, pred_y)
    # jax.debug.print("{loss}", loss=loss)
    return loss, outs

@eqx.filter_value_and_grad(has_aux=True)
def compute_loss_and_grads(model, x, y):
    return compute_loss_and_outputs(model, x, y)

@eqx.filter_jit
def compute_influence_matrix(carry, inp):
    M_prev = carry
    J, bar_M = inp
    M = jnp.einsum('bkl,blij->bkij', J, M_prev) + bar_M
    return M, M

@eqx.filter_jit
def make_step_bptt(full_model, x, y, opt_state, optim, cell_type, prune):
    params, static = eqx.partition(full_model, eqx.is_inexact_array)
    trainable_params = haliax.state_dict.to_state_dict(params)

    (loss, outs), grads = compute_loss_and_grads(full_model, x, y)

    # updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    grads = haliax.state_dict.to_state_dict(eqx.filter(grads, eqx.is_inexact_array))
    updates, opt_state = optim.update(grads, opt_state, trainable_params)
    trainable_params = eqx.apply_updates(trainable_params, updates)
    # For jax pruner
    if prune:
        trainable_params = pruner.post_gradient_update(trainable_params, opt_state)
    ## jax pruner
    params = haliax.state_dict.from_state_dict(params, trainable_params)
    full_model = eqx.combine(params, static)

    time_state_sparsity = None
    if cell_type in [CellType.EqxGRU]:
        time_state_sparsity = jnp.mean(outs == 0., axis=2)
    elif cell_type in [CellType.EGRU]:
        time_state_sparsity = jnp.mean(outs[0] == 0., axis=2)
    # loss, full_model, opt_state, outs, sparsity

    return loss, full_model, opt_state, outs, (time_state_sparsity, time_state_sparsity)

@eqx.filter_jit
def make_step_rtrl(full_model, x, y, opt_state, optim, cell_type, prune):

    params, static = eqx.partition(full_model, eqx.is_inexact_array)
    trainable_params = haliax.state_dict.to_state_dict(params)
    # lmm = lambda mm: mm(x), mm(x)

    ## full_model: x -> pred_y, outs
    ## For EGRU, outs = (h, c, o, i, (Jh, barMh), (Jc, barMc), (Jo, barMo), (Ji, barMi))
    ## For eqx.GRU, outs = (h)
    # @eqx.filter_grad
    @eqx.filter_jit
    def fma(params, static, xx):
        mm = eqx.combine(params, static)
        pred_y, outs = mm(xx)
        return (pred_y, outs), (pred_y, outs)

    # @eqx.filter_grad
    @eqx.filter_jit
    def jj(mm, xx):
        params, static = eqx.partition(mm, eqx.is_array)
        return jax.jacfwd(fma, has_aux=True)(params, static, xx)

    @eqx.filter_jit
    def ls(mm, hht, y):
        # lg, pred_y = jax.jacfwd(lin, has_aux=True)(lambda x: jax.nn.sigmoid(mm.linear(x)), hht)
        def _loss(lin):
            pred_y = jax.nn.sigmoid(lin(hht))
            l, _ = loss_fn(y, pred_y)
            return l, l

        lg, loss = jax.jacfwd(_loss, has_aux=True)(mm.linear)
        # ipdb.set_trace()
        return loss, (loss, lg)

    ## Jouts is M at (t-1)
    (_Jpred_y, Jouts), (_pred_y, outs) = jax.vmap(partial(jj, full_model))(x)
    # ipdb.set_trace()

    # jax.debug.print("Mean value of states: {m}", m=jnp.mean(outs[0]))
    # jax.debug.print("Percent zeros in states: {m}", m=jnp.mean(outs[0] == 0.))
    # jax.debug.print("Percent zeros in Ms: {m}", m=jnp.mean(jnp.isclose(Jouts[0].cell.weight_hh, 0.)))


    if cell_type in [CellType.EqxGRU]:
        final_state = outs[:, -1]
        time_state_sparsity = jnp.mean(outs == 0., axis=2)
        time_J_sparsity = None
    else:
        final_state = outs[0][:, -1]
        time_state_sparsity = jnp.mean(outs[0] == 0., axis=2)
        time_J_sparsity = jnp.mean(jnp.isclose(Jouts[0].cell.weight_hh, 0.), axis=(2, 3, 4))
    Jloss, (loss, lg) = jax.vmap(partial(jax.jacfwd(ls, has_aux=True, argnums=1), full_model))(final_state, y)

    def calc_grad(M):
        barC = Jloss
        bhgrads = jnp.einsum('bh,bh...->b...', barC, M[:, -1])
        return jnp.mean(bhgrads, axis=(0,))

    if cell_type in [CellType.EqxGRU]:
        cell_grads = jax.tree_util.tree_map(calc_grad, Jouts.cell)
    else:
        cell_grads = jax.tree_util.tree_map(calc_grad, Jouts[0].cell)
    lin_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=(0,)), lg)

    ## Test if grads are correct (Yes they are!!)
    loss = jnp.mean(loss)

    # jax.debug.print("Are the hh grads correct?: {a}", a=jnp.isclose(cell_grads.weight_hh, grads_.cell.weight_hh).all())
    # jax.debug.print("Are the lin grads correct?: {a}", a=jnp.isclose(lin_grads.weight, grads_.linear.weight).all())
    # jax.debug.print("Grads: {a}, {b}", a=grads.weight_hh, b=grads_.cell.weight_hh)
    # ipdb.set_trace()

    # wcg = eqx.tree_at(lambda m: m.cell, trainable_params, replace=cell_grads)
    # grads = eqx.tree_at(lambda m: m.linear, wcg, replace=lin_grads)
    grads = haliax.state_dict.to_state_dict(eqx.filter(cell_grads, eqx.is_inexact_array), prefix="cell") | \
        haliax.state_dict.to_state_dict(eqx.filter(lin_grads, eqx.is_inexact_array), prefix="linear")

    # (loss, outs), grads_ = compute_loss_and_grads(full_model, x, y)
    # jax.debug.print("Are the hh grads correct?: {a}", a=jnp.isclose(grads.cell.weight_hh, grads_.cell.weight_hh).all())
    # jax.debug.print("Are the lin grads correct?: {a}", a=jnp.isclose(grads.linear.weight, grads_.linear.weight).all())

    # updates, opt_state = optim.update(grads, opt_state, eqx.filter(full_model, eqx.is_array))
    updates, opt_state = optim.update(grads, opt_state, trainable_params)
    # ipdb.set_trace()
    trainable_params = eqx.apply_updates(trainable_params, updates)
    # ipdb.set_trace()

    # For jax pruner
    if prune:
        trainable_params = pruner.post_gradient_update(trainable_params, opt_state)
    ## jax pruner

    params = haliax.state_dict.from_state_dict(params, trainable_params)
    full_model = eqx.combine(params, static)

    return loss, full_model, opt_state, outs, (time_state_sparsity, time_J_sparsity)

    # return

    # # (loss, outs), grads = compute_loss_and_grads(full_model, x, y)
    # loss, outs = compute_loss_and_outputs(full_model, x, y)
    # ## For now, assume only last timestep contributes to loss. TODO fix later.
    # (new_h, new_c, new_o, new_i, jacs) = outs
    # # (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi) = jacs
    # updates, opt_state = optim.update(grads, opt_state)
    # full_model = eqx.apply_updates(full_model, updates)
    # return loss, full_model, opt_state, outs

def train(
        make_step,
        batch_size=50,
        learning_rate=3e-3,
        steps=10000,
        hidden_size=16,
        seed=5678,
        weight_sparsity=0.,
        disable_activity_sparsity=False,
        cell_type=CellType.EGRU,
        prune=False,
        pruner=None,
        use_wandb=False,
        dataset='speech',
        dataset_size = 10000,
        seq_len = 17,
):
    print(f"Seed: {seed}")
    if use_wandb:
        wandb.init(project="sparse-rtrl", entity="anands",
                   config=dict(seq_len=seq_len, batch_size=batch_size, learning_rate=learning_rate, steps=steps,
                               hidden_size=hidden_size, seed=seed, weight_sparsity=weight_sparsity,
                               disable_activity_sparsity=disable_activity_sparsity, cell_type=cell_type.value))

    if dataset == 'speech':
        cache_dir = './raw_datasets/speech_commands/0.0.3/SpeechCommands/processed_data'
        trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE = \
                create_speechcommands35_classification_dataset(cache_dir, bsz = batch_size, seed = seed)
    elif dataset == 'toy':
        trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE = \
            create_toy_classification_dataset(dataset_size, seq_len, bsz = batch_size, seed = seed)

    _, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)

    if cell_type in [CellType.EqxGRU]:
        model = RNN(cell_type=cell_type, in_size=IN_DIM, out_size=N_CLASSES, hidden_size=hidden_size, key=model_key)
    elif cell_type in [CellType.RNN]:
        model = RNN(cell_type=cell_type, in_size=IN_DIM, out_size=N_CLASSES, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=IN_DIM, out_size=N_CLASSES, hidden_size=hidden_size, key=model_key,
                    weight_sparsity=weight_sparsity, activity_sparse=(not disable_activity_sparsity))

    full_model = model

    optim = optax.adam(learning_rate)
    # For Jax pruner
    if prune:
        optim = pruner.wrap_optax(optim)
    ## End Jax pruner
    # ipdb.set_trace()

    params = eqx.filter(full_model, eqx.is_inexact_array)
    trainable_params = haliax.state_dict.to_state_dict(params)
    opt_state = optim.init(trainable_params)

    validation_accs = []
    cum_mean_state_density, cum_mean_M_density = 0., 0.
    # for step, (x, y) in zip(range(steps), trn_loader):
    n_epochs = 10
    for epoch in range(n_epochs):
        for step, batch in enumerate(tqdm(trn_loader)):
            x, y, integration_times = prep_batch(batch, SEQ_LENGTH, IN_DIM)

            loss, full_model, opt_state, outs, sparsity = make_step(full_model, x, y, opt_state, optim, cell_type, prune)

            # print(outs)
            loss = loss.item()
            mean_state_sparsity = jnp.mean(sparsity[0])
            mean_M_sparsity = jnp.mean(sparsity[1])
            cum_mean_state_density += (1 - mean_state_sparsity)
            cum_mean_M_density += (1 - mean_M_sparsity)
            data = dict(epoch=epoch, step=step, loss=loss, state_sparsity=sparsity[0], M_sparsity=sparsity[1],
                        mean_state_sparsity=mean_state_sparsity,
                        mean_M_sparsity=mean_M_sparsity,
                        cum_mean_state_density=cum_mean_state_density,
                        cum_mean_M_density=cum_mean_M_density,
                        mean_sq_M_sparsity=jnp.mean(sparsity[1] ** 2))
            if use_wandb:
                wandb.log(dict(train=data))

            if step % 10 == 0:
                print(f"epoch={epoch}, step={step}, loss={loss}")

        va = []
        for batch_val in val_loader:
            xs_val, ys_val, integration_times = prep_batch(batch_val, SEQ_LENGTH, IN_DIM)
            pred_ys, outs = jax.vmap(full_model)(xs_val)

            num_correct = jnp.sum(pred_ys.argmax(axis=1) == ys_val)
            # ipdb.set_trace()
            acc = (num_correct / len(xs_val)).item()
            va.append(acc)

        acc = np.mean(va)
        validation_accs.append(acc)
        if use_wandb:
            wandb.log(dict(validation=dict(step=step, accuracy=acc)))
        print(f"epoch={epoch}, step={step}, validation_accuracy={acc}")
        if prune:
            print(jaxpruner.summarize_sparsity(full_model, only_total_sparsity=True))
        if jnp.mean(jnp.array(validation_accs[-3:])) > 0.99:
            print("=================== Reached required accuracy")

    ta = []
    for batch_tst in tst_loader:
        xs_test, ys_test, integration_times = prep_batch(batch_val, SEQ_LENGTH, IN_DIM)
        pred_ys, outs = jax.vmap(full_model)(xs_test)

        num_correct = jnp.sum(pred_ys.argmax(axis=1) == ys_val)
        final_accuracy = (num_correct / len(xs_test)).item()
        ta.append(final_accuracy)

    final_accuracy = np.mean(ta)
    if use_wandb:
        wandb.log(dict(test=dict(accuracy=final_accuracy)))
    print(f"test_accuracy={final_accuracy}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--location', type=str, choices=['mac', 'desktop', 'jusuf', 'taurus'], default='mac')
    argparser.add_argument('--weight-sparsity', type=float, default=0.0)
    argparser.add_argument('--seed', type=int, default=5678)
    argparser.add_argument('--disable-activity-sparsity', action='store_true')
    argparser.add_argument('--wandb', action='store_true')
    argparser.add_argument('--prune', action='store_true')
    argparser.add_argument('--method', type=str, choices=['rtrl', 'bptt'], default='bptt')
    argparser.add_argument('--dataset', type=str, choices=['toy', 'speech'], default='toy')
    argparser.add_argument('--model', type=str, choices=['gru', 'egru'], default='egru')

    args = argparser.parse_args()

    if args.model == 'gru':
        cell_type = CellType.EqxGRU
    elif args.model == 'egru':
        cell_type = CellType.EGRU
    else:
        raise RuntimeError(f"Unknown model {cell_type}")

    config_dict = dict(seed=args.seed,
                       cell_type=cell_type, hidden_size=[64, 64],
                       weight_sparsity=args.weight_sparsity, disable_activity_sparsity=args.disable_activity_sparsity,
                       prune=args.prune,
                       use_wandb=args.wandb,
                       dataset=args.dataset,
                       )
    # config = ml_collections.ConfigDict(config_dict)

    pruner = None
    if args.prune:
        sparsity_config_dict = dict(
            algorithm='magnitude',
            update_freq=10,
            update_end_step=1000,
            update_start_step=200,
            sparsity=0.95,
            dist_type='erk',
        )
        sparsity_config = ml_collections.ConfigDict(sparsity_config_dict)
        # sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=0.8)
        # pruner = jaxpruner.MagnitudePruning(sparsity_distribution_fn=sparsity_distribution)
        pruner = jaxpruner.create_updater_from_config(sparsity_config)

    with launch_ipdb_on_exception():
        if args.method == 'rtrl':
            train(make_step=make_step_rtrl, pruner=pruner, **config_dict)  # All right, let's run the code.
        elif args.method == 'bptt':
            train(make_step=make_step_bptt, pruner=pruner, **config_dict)
        else:
            raise RuntimeError(f"Unknown method {args.method}")
