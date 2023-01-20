import pickle

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from data import get_data, dataloader
from models import EqxRNN, RNN, CellType


@eqx.filter_jit
def loss_fn(y, pred_y):
    # Trains with respect to binary cross-entropy
    l = -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    return l, l


@eqx.filter_jit
def compute_loss_and_outputs(model, linear, x, y):
    final_state, outs = jax.vmap(model)(x)
    pred_y = jax.vmap(linear)(final_state)
    # Trains with respect to binary cross-entropy
    loss, _ = loss_fn(y, pred_y)
    return loss, outs


@eqx.filter_jit
def compute_influence_matrix(carry, inp):
    M_prev = carry
    J, bar_M = inp
    M = jnp.einsum('bkl,blij->bkij', J, M_prev) + bar_M
    return M, M


def train_fwd_explicit(
        dataset_size=10000,
        seq_len=17,
        batch_size=32,
        learning_rate=3e-3,
        steps=600,
        hidden_size=16,
        seed=5678,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
):
    raise RuntimeError("Function not fully implemented")
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    linear = eqx.nn.Linear(out_features=1, in_features=hidden_size, key=model_key)

    params, static = eqx.partition(model, eqx.is_array)

    # import ipdb
    # ipdb.set_trace()

    @eqx.filter_jit
    @eqx.filter_grad
    def make_step(model, x, y, opt_state):
        final_state, outs = jax.vmap(model)(x)

        new_h, new_c, new_o, new_i, (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi) = outs
        states, Js, bar_Ms = new_h, Jh, bar_Mh


        Js_tr = jnp.transpose(Js, (1, 0, 2, 3))
        bar_Ms_tr = jax.tree_util.tree_map(lambda bar_ms: jnp.swapaxes(bar_ms, 0, 1), bar_Ms)

        print("Transpose shape bar_Ms & Js: ", jax.tree_util.tree_map(lambda bar_ms_tr: bar_ms_tr.shape, bar_Ms_tr), Js_tr.shape)

        import ipdb
        ipdb.set_trace()

        ## FIXME: Need to do map tree
        _, Ms_tr = lax.scan(compute_influence_matrix, jnp.zeros_like(bar_Ms[:, 0]), (Js_tr, bar_Ms_tr))

        Ms = jnp.transpose(Ms_tr, (1, 0, 2, 3, 4))
        ## End calculate M

        # Calculate loss and \do L(t)/\do a(t)
        # Need to apply linear readout here before loss, not inside
        fn = lambda a: jax.vmap(loss_fn)(y, jax.vmap(linear)(a))

        jac = jax.jacfwd(fn, argnums=(0,), has_aux=True)
        (bar_c, ), loss = jac(final_state)

        import ipdb
        ipdb.set_trace()

        nb = Ms.shape[0]
        nh = Ms.shape[2]
        grads = jnp.einsum('bk,bkp->bp', bar_c[:, -1], Ms[:, -1].reshape(nb, nh, 3 * nh ** 2))
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, outs

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state, outs = make_step(model, x, y, opt_state)
        # print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys, _ = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


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
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss_and_outputs(model, x, y):
        final_state, outs = jax.vmap(model)(x)
        pred_y, _ = lin(final_state)
        loss, _ = loss_fn(pred_y)
        # Trains with respect to binary cross-entropy
        return loss, outs

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        (loss, outs), grads = compute_loss_and_outputs(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, outs

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state, outs = make_step(model, x, y, opt_state)
        # print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys, _ = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


def main(
        dataset_size=10000,
        seq_len=100,
        batch_size=32,
        hidden_size=16,
        seed=5678,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, seq_len=seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    linear = eqx.nn.Linear(out_features=1, in_features=hidden_size, key=model_key)

    x, y = next(iter_data)
    if cell_type in [CellType.EGRU]:
        loss, (new_h, new_c, new_o, new_i, (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi)) = \
            compute_loss_and_outputs(model, linear, x, y)
        states, Js, bar_Ms = new_c * new_o, Jc, bar_Mc
        print("EGRU")
    elif cell_type in [CellType.RNN]:
        loss, (states, bar_Ms, Js) = compute_loss_and_outputs(model, x, y)

    print("Shape bar_Ms & Js: ", jax.tree_util.tree_map(lambda bar_ms: bar_ms.shape, bar_Ms), Js.shape)

    # print(Js[:, 0])
    loss = loss.item()
    print(f"loss={loss}")

    Js_tr = jnp.transpose(Js, (1, 0, 2, 3))
    bar_Ms_tr = jax.tree_util.tree_map(lambda bar_ms: jnp.swapaxes(bar_ms, 0, 1), bar_Ms)

    print("Transpose shape bar_Ms & Js: ", jax.tree_util.tree_map(lambda bar_ms_tr: bar_ms_tr.shape, bar_Ms_tr), Js_tr.shape)

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
    from ipdb import launch_ipdb_on_exception

    # with launch_ipdb_on_exception():
    #     train_fwd()
    # train()  # All right, let's run the code.
    with launch_ipdb_on_exception():
        main()  # All right, let's run the code.
