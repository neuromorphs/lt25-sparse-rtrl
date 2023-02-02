import pickle
from functools import partial

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax

import ipdb

from data import get_data, dataloader
from models import EqxRNN, RNN, CellType


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
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key, output_jac=False)

    full_model = model

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(full_model, x, y, opt_state):
        (loss, outs), grads = compute_loss_and_grads(full_model, x, y)
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

    pred_ys, outs = jax.vmap(full_model)(xs)

    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")

def train_fwd_implicit(
        dataset_size=10000,
        seq_len=17,
        batch_size=32,
        learning_rate=3e-3,
        steps=6000,
        hidden_size=16,
        seed=5678,
        # cell_type=CellType.EqxGRU,
        cell_type=CellType.EGRU,
):
    data_key_train, data_key_val, data_key_test, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 5)

    xs, ys = get_data(dataset_size, seq_len, key=data_key_train)
    idx = jax.random.randint(data_key_val, (dataset_size, ), 0, dataset_size)
    xs, ys = xs.take(idx, axis=0), ys.take(idx, axis=0)
    xs_train, xs_val, xs_test = xs[:int(dataset_size * 0.7)], xs[int(dataset_size * 0.7):int(dataset_size * 0.85)], xs[int(dataset_size * 0.85):]
    ys_train, ys_val, ys_test = ys[:int(dataset_size * 0.7)], ys[int(dataset_size * 0.7):int(dataset_size * 0.85)], ys[int(dataset_size * 0.85):]
    iter_data = dataloader((xs_train, ys_train), batch_size, key=loader_key)
    # ipdb.set_trace()

    if cell_type in [CellType.EqxGRU]:
        model = EqxRNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    else:
        model = RNN(cell_type=cell_type, in_size=2, out_size=1, hidden_size=hidden_size, key=model_key, output_jac=False)

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

        jax.debug.print("Mean value of states: {m}", m=jnp.mean(outs[0]))
        jax.debug.print("Percent zeros in states: {m}", m=jnp.mean(outs[0] == 0.))
        jax.debug.print("Percent zeros in Ms: {m}", m=jnp.mean(jnp.isclose(Jouts[0].cell.weight_hh, 0.)))

        if cell_type in [CellType.EqxGRU]:
            final_state = outs[:, -1]
        else:
            final_state = outs[0][:, -1]
        Jloss, (loss, lg) = jax.vmap(partial(jax.jacfwd(ls, has_aux=True, argnums=1), full_model))(final_state, y)

        def calc_grad(M):
            barC = Jloss
            bhgrads = jnp.einsum('bh,bh...->b...', barC, M[:, -1])
            return jnp.mean(bhgrads, axis=(0, ))

        if cell_type in [CellType.EqxGRU]:
            cell_grads = jax.tree_util.tree_map(calc_grad, Jouts.cell)
        else:
            cell_grads = jax.tree_util.tree_map(calc_grad, Jouts[0].cell)
        lin_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=(0, )), lg)

        ## Test if grads are correct (Yes they are!!)
        loss = jnp.mean(loss)


        # jax.debug.print("Are the hh grads correct?: {a}", a=jnp.isclose(cell_grads.weight_hh, grads_.cell.weight_hh).all())
        # jax.debug.print("Are the lin grads correct?: {a}", a=jnp.isclose(lin_grads.weight, grads_.linear.weight).all())
        # jax.debug.print("Grads: {a}, {b}", a=grads.weight_hh, b=grads_.cell.weight_hh)

        wcg = eqx.tree_at(lambda m: m.cell, full_model, replace=cell_grads)
        grads = eqx.tree_at(lambda m: m.linear, wcg, replace=lin_grads)

        # (loss, outs), grads_ = compute_loss_and_grads(full_model, x, y)
        # jax.debug.print("Are the hh grads correct?: {a}", a=jnp.isclose(grads.cell.weight_hh, grads_.cell.weight_hh).all())
        # jax.debug.print("Are the lin grads correct?: {a}", a=jnp.isclose(grads.linear.weight, grads_.linear.weight).all())

        updates, opt_state = optim.update(grads, opt_state)
        # ipdb.set_trace()
        full_model = eqx.apply_updates(full_model, updates)
        # ipdb.set_trace()

        return loss, full_model, opt_state, outs

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
    opt_state = optim.init(full_model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, full_model, opt_state, outs = make_step(full_model, x, y, opt_state)
        # print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
        if step % 100 == 0:
            pred_ys, outs = jax.vmap(full_model)(xs_val)

            num_correct = jnp.sum((pred_ys > 0.5) == ys_val)
            acc = (num_correct / len(xs_val)).item()
            print(f"step={step}, validation_accuracy={acc}")
            if acc > 0.99:
                break

    pred_ys, outs = jax.vmap(full_model)(xs_test)

    num_correct = jnp.sum((pred_ys > 0.5) == ys_test)
    final_accuracy = (num_correct / len(xs_test)).item()
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
    with launch_ipdb_on_exception():
        # main()  # All right, let's run the code.
        # train()  # All right, let's run the code.
        train_fwd_implicit()  # All right, let's run the code.
