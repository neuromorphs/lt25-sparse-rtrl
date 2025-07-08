import os
from datetime import datetime
import random
import pickle
from functools import partial
import argparse

import ipdb
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

from simmanager import SimManager
from simrecorder import Recorder, ZarrDataStore

from data import get_data, dataloader
from models import EqxRNN, RNN, CellType


def record_dict(prefix, d):
    for k, v in d.items():
        recorder.record('{}/{}'.format(prefix, k), np.array(v))


def get_random_name(prefix='baseline'):
    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    randnum = str(random.randint(1e3, 1e5))
    sim_name = f"{prefix}-{randnum}-{datetime_suffix}"
    return sim_name


@eqx.filter_jit
def loss_fn(y, pred_y):
    # Trains with respect to binary cross-entropy
    l = -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
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


def train(
        dataset_size=10000,
        seq_len=16,
        batch_size=32,
        learning_rate=3e-3,
        steps=600,
        hidden_size=16,
        seed=5678,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
        prune=False,
):
    """
    Trains with BPTT
    """
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)

    xs, ys = get_data(dataset_size, seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key,
                    output_jac=False)

    full_model = model

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(full_model, x, y, opt_state):
        (loss, outs), grads = compute_loss_and_grads(full_model, x, y)
        # updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        updates, opt_state = optim.update(grads, opt_state)
        full_model = eqx.apply_updates(full_model, updates)
        return loss, full_model, opt_state, outs

    optim = optax.adam(learning_rate)

    opt_state = optim.init(full_model)

    for step, (x, y) in zip(range(steps), iter_data):
        loss, full_model, opt_state, outs = make_step(full_model, x, y, opt_state)
        # print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
        if prune:
            print(jaxpruner.summarize_sparsity(full_model, only_total_sparsity=True))

    pred_ys, outs = jax.vmap(full_model)(xs)

    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


def train_fwd_implicit(
        dataset_size=10000,
        seq_len=17,
        batch_size=32,
        learning_rate=3e-3,
        steps=1700,
        hidden_size=16,
        seed=5678,
        weight_sparsity=0.,
        disable_activity_sparsity=False,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
        prune=False,
        use_wandb=False,
        use_simmanager=False
):
    """
    Trains with RTRL
    """
    print(f"Seed: {seed}")
    if use_wandb:
        wandb.init(project="sparse-rtrl", entity="anands",
                   config=dict(seq_len=seq_len, batch_size=batch_size, learning_rate=learning_rate, steps=steps,
                               hidden_size=hidden_size, seed=seed, weight_sparsity=weight_sparsity,
                               disable_activity_sparsity=disable_activity_sparsity, cell_type=cell_type.value))

    data_key_train, data_key_val, data_key_test, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 5)

    xs, ys = get_data(dataset_size, seq_len, key=data_key_train)
    idx = jax.random.randint(data_key_val, (dataset_size,), 0, dataset_size)
    xs, ys = xs.take(idx, axis=0), ys.take(idx, axis=0)

    xs_train, xs_val, xs_test = xs[:int(dataset_size * 0.7)], xs[int(dataset_size * 0.7):int(dataset_size * 0.85)], \
                                    xs[int(dataset_size * 0.85):]
    ys_train, ys_val, ys_test = ys[:int(dataset_size * 0.7)], ys[int(dataset_size * 0.7):int(dataset_size * 0.85)], \
                                    ys[int(dataset_size * 0.85):]

    iter_data = dataloader((xs_train, ys_train), batch_size, key=loader_key)
    # ipdb.set_trace()

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    elif cell_type in [CellType.RNN]:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key,
                    weight_sparsity=weight_sparsity, activity_sparse=(not disable_activity_sparsity))

    full_model = model

    # @eqx.filter_jit
    # def compute_loss_and_outputs(model, x, y):
    #     pred_y, outs = jax.vmap(model)(x)
    #     # jax.debug.print("{pred_y}", pred_y=pred_y)
    #     ## Trains with respect to binary cross-entropy
    #     loss, _ = loss_fn(y, pred_y)
    #     # jax.debug.print("{loss}", loss=loss)
    #     return loss, outs

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(full_model, x, y, opt_state):

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

        time_state_sparsity = jnp.mean(outs[0] == 0., axis=2)
        time_J_sparsity = jnp.mean(jnp.isclose(Jouts[0].cell.weight_hh, 0.), axis=(2, 3, 4))

        if cell_type in [CellType.EqxGRU]:
            final_state = outs[:, -1]
        else:
            final_state = outs[0][:, -1]
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

    optim = optax.adam(learning_rate)
    # For Jax pruner
    if prune:
        sparsity_distribution = partial( jaxpruner.sparsity_distributions.uniform, sparsity=0.8 )
        pruner = jaxpruner.MagnitudePruning( sparsity_distribution_fn=sparsity_distribution )
        # sparsity_updater = jaxpruner.create_updater_from_config(config.sparsity_config)
        optim = pruner.wrap_optax(optim)
    ## End Jax pruner
    # ipdb.set_trace()

    params = eqx.filter(full_model, eqx.is_inexact_array)
    trainable_params = haliax.state_dict.to_state_dict(params)
    opt_state = optim.init(trainable_params)

    validation_accs = []
    cum_mean_state_density, cum_mean_M_density = 0., 0.
    for step, (x, y) in zip(range(steps), iter_data):
        loss, full_model, opt_state, outs, sparsity = make_step(full_model, x, y, opt_state)

        # print(outs)
        loss = loss.item()
        mean_state_sparsity = jnp.mean(sparsity[0])
        mean_M_sparsity = jnp.mean(sparsity[1])
        cum_mean_state_density += (1 - mean_state_sparsity)
        cum_mean_M_density += (1 - mean_M_sparsity)
        data = dict(step=step, loss=loss, state_sparsity=sparsity[0], M_sparsity=sparsity[1],
                    mean_state_sparsity=mean_state_sparsity,
                    mean_M_sparsity=mean_M_sparsity,
                    cum_mean_state_density=cum_mean_state_density,
                    cum_mean_M_density=cum_mean_M_density,
                    mean_sq_M_sparsity=jnp.mean(sparsity[1] ** 2))
        if use_simmanager:
            record_dict('train', data)
        if use_wandb:
            wandb.log(dict(train=data))

        if step % 10 == 0:
            print(f"step={step}, loss={loss}")
            if prune:
                print(jaxpruner.summarize_sparsity(full_model, only_total_sparsity=True))
        if step % 100 == 0:
            pred_ys, outs = jax.vmap(full_model)(xs_val)

            num_correct = jnp.sum((pred_ys > 0.5) == ys_val)
            acc = (num_correct / len(xs_val)).item()
            validation_accs.append(acc)
            if use_simmanager:
                record_dict('validation', dict(step=step, accuracy=acc))
            if use_wandb:
                wandb.log(dict(validation=dict(step=step, accuracy=acc)))
            print(f"step={step}, validation_accuracy={acc}")
            if jnp.mean(jnp.array(validation_accs[-3:])) > 0.999:
                print("=================== Reached required accuracy")
                # break

    pred_ys, outs = jax.vmap(full_model)(xs_test)

    num_correct = jnp.sum((pred_ys > 0.5) == ys_test)
    final_accuracy = (num_correct / len(xs_test)).item()
    if use_simmanager:
        record_dict('test', dict(accuracy=final_accuracy))
    if use_wandb:
        wandb.log(dict(test=dict(accuracy=final_accuracy)))
    print(f"test_accuracy={final_accuracy}")


def main(
        dataset_size=10000,
        seq_len=100,
        batch_size=32,
        hidden_size=16,
        seed=5678,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
):
    # raise RuntimeError("Doesn't work with internal linear layer yet")
    data_key_train, data_key_val, data_key_test, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 5)
    xs, ys = get_data(dataset_size, seq_len=seq_len, key=data_key_train)
    xs_val, ys_val = get_data(2500, seq_len=seq_len, key=data_key_val)
    xs_test, ys_test = get_data(2500, seq_len=seq_len, key=data_key_test)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    # linear = eqx.nn.Linear(out_features=1, in_features=hidden_size, key=model_key)

    full_model = model

    x, y = next(iter_data)
    if cell_type in [CellType.EGRU]:
        loss, (new_h, new_c, new_o, new_i, (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi)) = \
            compute_loss_and_outputs(full_model, x, y)
        states, Js, bar_Ms = new_c * new_o, Jc, bar_Mc
        print("EGRU")
    elif cell_type in [CellType.RNN]:
        loss, (states, bar_Ms, Js) = compute_loss_and_outputs(full_model, x, y)

    print("Shape bar_Ms & Js: ", jax.tree_util.tree_map(lambda bar_ms: bar_ms.shape, bar_Ms), Js.shape)

    # print(Js[:, 0])
    loss = loss.item()
    print(f"loss={loss}")

    Js_tr = jnp.transpose(Js, (1, 0, 2, 3))
    bar_Ms_tr = jax.tree_util.tree_map(lambda bar_ms: jnp.swapaxes(bar_ms, 0, 1), bar_Ms)

    print("Transpose shape bar_Ms & Js: ", jax.tree_util.tree_map(lambda bar_ms_tr: bar_ms_tr.shape, bar_Ms_tr),
          Js_tr.shape)

    bar_Ms_tr_w = bar_Ms_tr.weight_hh

    _, Ms_tr = lax.scan(compute_influence_matrix, jnp.zeros_like(bar_Ms_tr_w[0, :]), (Js_tr, bar_Ms_tr_w))

    Ms = jnp.transpose(Ms_tr, (1, 0, 2, 3, 4))

    print("Ms.shape: ", Ms.shape)

    print("Mean value of states: ", jnp.mean(states))
    print("Percent zeros in states: ", jnp.mean(states == 0.))

    print("Percent zeros in Js: ", jnp.mean(jnp.isclose(Js, 0.)))
    print("Percent zeros in bar_Ms: ", jnp.mean(jnp.isclose(bar_Ms.weight_hh, 0.)))

    print("Percent zeros in Ms: ", jnp.mean(jnp.isclose(Ms, 0.)))

    # pred_ys, _ = jax.vmap(model)(xs)
    # num_correct = jnp.sum((pred_ys > 0.5) == ys)
    # final_accuracy = (num_correct / dataset_size).item()
    # print(f"final_accuracy={final_accuracy}")

    fname = f'sparse-data-{cell_type.value}-{seq_len}.p'
    with open(fname, 'wb') as f:
        pickle.dump(dict(Ms=Ms, bar_Ms=bar_Ms, Js=Js, states=states), f)
    print(f"Saved in {fname}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--location', type=str, choices=['mac', 'desktop', 'jusuf', 'taurus'], default='mac')
    argparser.add_argument('--weight-sparsity', type=float, default=0.0)
    argparser.add_argument('--seed', type=int, default=5678)
    argparser.add_argument('--disable-activity-sparsity', action='store_true')
    argparser.add_argument('--wandb', action='store_true')
    argparser.add_argument('--simmanager', action='store_true')
    argparser.add_argument('--prune', action='store_true')
    args = argparser.parse_args()

    from ipdb import launch_ipdb_on_exception

    # with launch_ipdb_on_exception():
    #     # train_fwd()
    #     # main()  # All right, let's run the code.
    #     # train()  # All right, let's run the code.
    #     train_fwd_implicit()  # All right, let's run the code.

    # train_fwd_implicit(seed=args.seed, weight_sparsity=args.weight_sparsity, disable_activity_sparsity=args.disable_activity_sparsity,
    #         use_wandb=args.wandb)  # All right, let's run the code.
    config_dict = dict(seed=args.seed, weight_sparsity=args.weight_sparsity,
                       disable_activity_sparsity=args.disable_activity_sparsity, use_wandb=args.wandb,
                       use_simmanager=args.simmanager, prune=args.prune, cell_type=CellType.EGRU)
    config = ml_collections.ConfigDict(config_dict)

    sparsity_config_dict = dict(
        algorithm='magnitude',
        update_freq=10,
        update_end_step=1000,
        update_start_step=200,
        sparsity=0.95,
        dist_type='erk',
    )
    config.sparsity_config = ml_collections.ConfigDict(sparsity_config_dict)

    if not args.simmanager:
        with launch_ipdb_on_exception():
            train_fwd_implicit(**config_dict)  # All right, let's run the code.
    else:
        ## START DIR NAMES
        if args.location == 'mac':
            rroot = os.path.expanduser(os.path.join('~', 'output'))
            data_path = './data'
        elif args.location == 'desktop':
            rroot = os.path.join('/scratch', 'anand', 'output')
            data_path = os.path.join(rroot, 'DATA')
        elif args.location == 'jusuf':
            rroot = os.path.expandvars(os.path.join('$SCRATCH', 'output'))  # JUSUF
            data_path = os.path.expandvars(os.path.join('$PROJECT', 'DATA'))  # JUSUF
        elif args.location == 'taurus':  # TUD cluster
            rroot = os.path.join('/beegfs/ws/0/ansu260e-evnn-workspace', 'output')
            data_path = os.path.join(rroot, 'DATA')
        else:
            raise RuntimeError(f"Unknown location: {args.location}")
        ## END DIR NAMES

        print(rroot)
        root_dir = os.path.join(rroot, 'sparse-rtrl')
        os.makedirs(root_dir, exist_ok=True)
        sim_name = get_random_name()

        with SimManager(sim_name, root_dir, write_protect_dirs=False, tee_stdx_to='output.log') as simman:
            paths = simman.paths

            print("Results will be stored in ", paths.results_path)
            os.makedirs(os.path.join(paths.results_path, 'models'), exist_ok=True)

            with open(os.path.join(paths.data_path, 'config.yaml'), 'w') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            zarr_datastore = ZarrDataStore(os.path.join(paths.results_path, 'data.mdb'))
            recorder = Recorder(zarr_datastore)

            train_fwd_implicit(**config_dict)  # All right, let's run the code.

            recorder.close()

            # make_plots(paths.results_path, config.total_input_width)
            print("Results stored in ", paths.results_path)

# FIXME: Storate simmanager paths in wandb
