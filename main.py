import pickle

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from data import get_data, dataloader
from models import RNN


def train(
        dataset_size=10000,
        batch_size=32,
        learning_rate=3e-3,
        steps=1000,
        hidden_size=16,
        seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss_and_outputs(model, x, y):
        pred_y, outs = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y)), outs

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
        print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


def main(
        dataset_size=10000,
        seq_len=100,
        batch_size=32,
        hidden_size=16,
        seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, seq_len=seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_jit
    def compute_loss_and_outputs(model, x, y):
        pred_y, outs = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y)), outs

    x, y = next(iter_data)
    loss, (states, bar_Ms, Js) = compute_loss_and_outputs(model, x, y)
    print("Shape states & bar_Ms & Js: ", states.shape, bar_Ms.shape, Js.shape)
    # print(Js[:, 0])
    loss = loss.item()
    print(f"loss={loss}")

    def compute_influence_matrix(carry, inp):
        M_prev = carry
        J, bar_M = inp
        M = jnp.einsum('bkl,blij->bkij', J, M_prev) + bar_M
        return M, M

    Js_tr = jnp.transpose(Js, (1, 0, 2, 3))
    bar_Ms_tr = jnp.transpose(bar_Ms, (1, 0, 2, 3, 4))

    print("Transpose shape bar_Ms & Js: ", bar_Ms_tr.shape, Js_tr.shape)

    _, Ms_tr = lax.scan(compute_influence_matrix, jnp.zeros_like(bar_Ms[:, 0]), (Js_tr, bar_Ms_tr))

    Ms = jnp.transpose(Ms_tr, (1, 0, 2, 3, 4))

    print("Ms.shape: ", Ms.shape)

    print("Mean value of states: ", jnp.mean(states))
    print("Percent zeros in states: ", jnp.mean(states == 0.))

    print("Percent zeros in Js: ", jnp.mean(jnp.isclose(Js, 0.)))
    print("Percent zeros in bar_Ms: ", jnp.mean(jnp.isclose(bar_Ms, 0.)))

    print("Percent zeros in Ms: ", jnp.mean(jnp.isclose(Ms, 0.)))

    # pred_ys, _ = jax.vmap(model)(xs)
    # num_correct = jnp.sum((pred_ys > 0.5) == ys)
    # final_accuracy = (num_correct / dataset_size).item()
    # print(f"final_accuracy={final_accuracy}")

    with open(f'sparse-data-{seq_len}.p', 'wb') as f:
        pickle.dump(dict(Ms=Ms, bar_Ms=bar_Ms, Js=Js, states=states), f)


if __name__ == '__main__':
    # train()  # All right, let's run the code.
    main()  # All right, let's run the code.
