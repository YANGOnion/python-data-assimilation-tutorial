
import torch
import numpy as np
from typing import Dict, Tuple
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
from sklearn.metrics import root_mean_squared_error


state_dim = 3
y_dim = 1
time_steps = 100
measure_noise_std = .5  # synthetic measurement error
model_noise_std = 2.0  # model noise


# Define the state transition model for observed data generation
def true_state_transition(states: torch.Tensor, rainfall: torch.Tensor, evap_coeff: torch.Tensor) -> torch.Tensor:
    '''
    :param states: (state_dim, )
    :param rainfall: (state_dim, )
    :param evap_coeff: (state_dim, )
    :return: (state_dim, )
    '''
    updated_states = states + rainfall - evap_coeff * states
    updated_states = torch.clamp(updated_states, 0, 50)
    return updated_states


# Define the state transition model with PyTorch
def state_transition(states: torch.Tensor, rainfall: torch.Tensor, evap_coeff: torch.Tensor) -> torch.Tensor:
    updated_states = states ** 0.99 * 1.01 + rainfall * 1.02 - evap_coeff * states
    updated_states = torch.clamp(updated_states, 0, 50)
    return updated_states


# Define the observation operator for observed data generation
def true_observation_operator(states: torch.Tensor, measure_noise_std: float) -> torch.Tensor:
    '''
    :param states: (state_dim, )
    :return: (y_dim, )
    '''
    # Non-linear relationship
    y = torch.sqrt(states.clamp(min=0)).sum().view(1)
    y += torch.randn(y_dim) * measure_noise_std
    return y


# Define the observation operator with PyTorch
def observation_operator(states: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(states.clamp(min=0)).sum().view(1)


# Generate synthetic data using PyTorch
def generate_data(time_steps: int, state_dim: int, measure_noise_std: float):
    # Generate synthetic rainfall data and evaporation coefficients
    rainfall_data = torch.rand((time_steps, state_dim)) * 20 - 10
    rainfall_data = torch.maximum(rainfall_data, torch.zeros_like(rainfall_data))
    evap_coef_data = torch.rand((state_dim,)) * 0.05 + 0.05
    evap_coef_data = evap_coef_data.unsqueeze(0).repeat(time_steps, 1)

    # Initialize variables
    true_state_list = [torch.rand(state_dim) * 20 + 20]  # Initial true state, random between 20 and 40
    sim_state_list = [torch.rand(state_dim) * 20 + 20]  # Initial simulated state, random between 20 and 40

    # Simulate true states and observations
    obs_y_list = []
    sim_y_list = []
    for t in range(time_steps):
        # Synthetic true states and simulated states
        true_state = true_state_transition(true_state_list[-1], rainfall_data[t], evap_coef_data[t])
        true_state_list.append(true_state)
        sim_state = state_transition(sim_state_list[-1], rainfall_data[t], evap_coef_data[t])
        sim_state_list.append(sim_state)

        # Synthetic observations and output simulations
        obs_y = true_observation_operator(true_state, measure_noise_std)
        obs_y_list.append(obs_y)
        sim_y = observation_operator(sim_state)
        sim_y_list.append(sim_y)

    # Convert lists to tensors
    true_states = torch.stack(true_state_list)
    sim_states = torch.stack(sim_state_list)
    observations_y = torch.stack(obs_y_list)
    simulations_y = torch.stack(sim_y_list)

    return rainfall_data, evap_coef_data, true_states[1:], sim_states[1:], observations_y, simulations_y


# Computes the gradient of the cost function w.r.t. to the state
def nonlinear_delta(x_forward: torch.Tensor,
                    innovations: torch.Tensor,
                    model: callable,
                    obs_operator: callable,
                    Q_inv: torch.Tensor,
                    R_inv: torch.Tensor,
                    window_model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
    '''
    :param x_forward: (assimilation_window, state_dim), state trajectory over the assimilation window
    :param innovations: (assimilation_window, y_dim), innovations over the assimilation window
    :param model: state transition
    :param obs_operator: observation operator
    :param Q_inv: (state_dim, state_dim), inverse of the state transition noise covariance matrix
    :param R_inv: (y_dim, y_dim), inverse of the observation noise covariance matrix
    :param window_model_input: dict of (assimilation_window, input_dim), external input for the state transition model
    :return: (state_dim, ), gradient of the cost function w.r.t. the initial state
    '''
    num_steps = len(innovations)

    # Initialize adjoint tensor with zeros
    adjoint = torch.zeros_like(x_forward)

    # Backward pass to compute gradients with respect to the initial state (x0)
    for t in range(num_steps - 1, -1, -1):  # Going backward in time

        # Gradients of the observation term (J_y)
        operator_tlm = torch.autograd.functional.jacobian(lambda x: obs_operator(x), x_forward[t])
        grad_obs = -operator_tlm.t() @ R_inv @ innovations[t]

        # Compute the adjoint at time t (this is the gradient w.r.t. the state at time t)
        adjoint[t] = grad_obs  # Adjoint at time t comes from the innovation term

        if t < num_steps - 1:
            # Backpropagate through the model: compute the adjoint of the model at time t
            step_model_input = {k: v[t + 1] for k, v in window_model_input.items()}
            model_tlm = torch.autograd.functional.jacobian(lambda x: model(x, **step_model_input), x_forward[t])

            # Chain rule to propagate the adjoint backward
            adjoint[t] += model_tlm.t() @ adjoint[t + 1]

    J_b_grad = Q_inv @ (x_forward[0] - x_background)  # Gradient of the background term
    x0_grad = adjoint[0] + J_b_grad
    return x0_grad


# Standard 4DVAR update using adjoint method
def standard_4dvar_update(x_background: torch.Tensor,
                          window_observations: torch.Tensor,
                          model: callable,
                          obs_operator: callable,
                          Q_inv: torch.Tensor,
                          R_inv: torch.Tensor,
                          window_model_input: Dict[str, torch.Tensor],
                          n_iter: int = 10,
                          lr: float = 0.01) -> torch.Tensor:
    '''
    :param x_background: (state_dim,), initial guess of state
    :param window_observations: (assimilation_window, y_dim), observations over the assimilation window
    :param model: state transition
    :param obs_operator: observation operator
    :param Q_inv: (state_dim, state_dim), inverse of the state transition noise covariance matrix
    :param R_inv: (y_dim, y_dim), inverse of the observation noise covariance matrix
    :param window_model_input: dict of (assimilation_window, input_dim), external input for the state transition model
    :param n_iter: number of iterations for the gradient descent
    :param lr: learning rate
    :return: (state_dim, ), updated state
    '''
    num_steps = len(window_observations)

    # Initialize the state
    x0 = x_background.clone()

    for _ in range(n_iter):
        # Forward integration (state propagation over the assimilation window)
        x_forward = torch.zeros((num_steps, x0.shape[0]), dtype=torch.float32)
        x_forward[0] = x0
        innovations = torch.zeros((num_steps, y_dim), dtype=torch.float32)
        for t in range(0, num_steps):
            if t > 0:
                step_model_input = {k: v[t] for k, v in window_model_input.items()}
                x_forward[t] = model(x_forward[t - 1], **step_model_input)
            innovations[t] = window_observations[t] - obs_operator(x_forward[t])

        # Backward integration to compute gradients of the cost w.r.t. initial state using adjoint
        x0_grad = nonlinear_delta(x_forward, innovations, model, obs_operator, Q_inv, R_inv, window_model_input)

        # Apply gradient descent to update x0
        x0 = x0 - lr * x0_grad

    return x0

# Compute the optimal increment of the initial state.
def linear_delta(x_forward: torch.Tensor,
                 innovations: torch.Tensor,
                 model: callable,
                 obs_operator: callable,
                 Q_inv: torch.Tensor,
                 R_inv: torch.Tensor,
                 window_model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
    '''
    :param x_forward: (assimilation_window, state_dim), state trajectory over the assimilation window
    :param innovations: (assimilation_window, y_dim), innovations over the assimilation window
    :param model: state transition
    :param obs_operator: observation operator
    :param Q_inv: (state_dim, state_dim), inverse of the state transition noise covariance matrix
    :param R_inv: (y_dim, y_dim), inverse of the observation noise covariance matrix
    :param window_model_input: dict of (assimilation_window, input_dim), external input for the state transition model
    :return: (state_dim, ), optimized increment of the initial state
    '''
    num_steps = len(innovations)
    state_dim = x_forward.shape[1]

    # Calculate the Tangent Linear Model (TLM) matrices
    model_tlm_stacks = torch.zeros((num_steps, state_dim, state_dim), dtype=torch.float32)
    operator_tlm_stacks = torch.zeros((num_steps, y_dim, state_dim), dtype=torch.float32)
    for t in range(num_steps):
        step_model_input = {k: v[t] for k, v in window_model_input.items()}
        model_tlm = torch.autograd.functional.jacobian(lambda x: model(x, **step_model_input), x_forward[t])
        operator_tlm = torch.autograd.functional.jacobian(lambda x: obs_operator(x), x_forward[t])
        model_tlm_stacks[t] = model_tlm
        operator_tlm_stacks[t] = operator_tlm

    def cost_function(dx0, num_steps, model_tlm_stacks_np, operator_tlm_stacks_np, innovations_np, Q_inv_np, R_inv_np):

        # Propagete perturbation using the TLM
        J_b = 0.5 * dx0.T @ Q_inv_np @ dx0
        J_y = 0.
        dxi = dx0.copy()
        for t in range(num_steps):
            if t > 0:
                dxi = model_tlm_stacks_np[t - 1] @ dxi
            dyi = operator_tlm_stacks_np[t] @ dxi
            J_y += 0.5 * (innovations_np[t] - dyi).T @ R_inv_np @ (innovations_np[t] - dyi)
        return J_b + J_y

    dx0_init = np.zeros_like(x_forward[0]) + .5
    model_tlm_stacks_np = model_tlm_stacks.numpy()
    operator_tlm_stacks_np = operator_tlm_stacks.numpy()
    innovations_np = innovations.numpy()
    Q_inv_np = Q_inv.numpy()
    R_inv_np = R_inv.numpy()

    res = minimize(
        cost_function, dx0_init,
        args=(num_steps, model_tlm_stacks_np, operator_tlm_stacks_np, innovations_np, Q_inv_np, R_inv_np),
        method='CG',
        options={'gtol': 1e-6}
    )
    dx0 = torch.tensor(res.x, dtype=torch.float32)

    return dx0


def incremental_4dvar_update(x_background: torch.Tensor,
                             window_observations: torch.Tensor,
                             model: callable,
                             obs_operator: callable,
                             Q_inv: torch.Tensor,
                             R_inv: torch.Tensor,
                             window_model_input: Dict[str, torch.Tensor],
                             n_iter: int = 10) -> torch.Tensor:
    '''
    :param x_background: (state_dim,), initial guess of state
    :param window_observations: (assimilation_window, y_dim), observations over the assimilation window
    :param model: state transition
    :param obs_operator: observation operator
    :param Q_inv: (state_dim, state_dim), inverse of the state transition noise covariance matrix
    :param R_inv: (y_dim, y_dim), inverse of the observation noise covariance matrix
    :param window_model_input: dict of (assimilation_window, input_dim), external input for the state transition model
    :param n_iter: number of iterations for the increment calculation
    :return: (state_dim, ), updated state
    '''
    num_steps = len(window_observations)

    # Initialize the state
    x0 = x_background.clone()

    for _ in range(n_iter):
        # Outer loop: forward integration (state propagation over the assimilation window)
        x_forward = torch.zeros((num_steps, x0.shape[0]), dtype=torch.float32)
        x_forward[0] = x0
        innovations = torch.zeros((num_steps, y_dim), dtype=torch.float32)
        for t in range(0, num_steps):
            if t > 0:
                step_model_input = {k: v[t] for k, v in window_model_input.items()}
                x_forward[t] = model(x_forward[t - 1], **step_model_input)
            innovations[t] = window_observations[t] - obs_operator(x_forward[t])

        # Compute the optimized increment
        dx0 = linear_delta(x_forward, innovations, model, obs_operator, Q_inv, R_inv, window_model_input)

        # Update the initial state (x0)
        x0 = x0 + dx0

    return x0


# 4DVAR loop using PyTorch
def vad_4dvar_loop(x_background: torch.Tensor,
                   observations_y: torch.Tensor,
                   assimilation_window: int,
                   model: callable,
                   obs_operator: callable,
                   Q: torch.Tensor,
                   R: torch.Tensor,
                   model_input: Dict[str, torch.Tensor],
                   method: str = "standard",
                   n_iter: int = 10,
                   lr: float = 0.01,
                   ) -> torch.Tensor:
    '''
    :param x_background: (state_dim,), initial guess of state
    :param observations_y: (time_steps, y_dim), observations
    :param assimilation_window: number of time steps to assimilate in a batch
    :param model: state transition
    :param obs_operator: observation operator
    :param Q: (state_dim, state_dim), state transition noise covariance matrix
    :param R: (y_dim, y_dim), observation noise covariance matrix
    :param model_input: dict of (time_steps, input_dim), external input for the state transition model
    :param method: "standard" or "incremental"
    :param n_iter: number of iterations for the gradient descent (standard) or increment calculation (incremental)
    :param lr: learning rate for the gradient descent
    :return: (time_steps, state_dim), updated state trajectory
    '''
    x_assimilated = []
    Q_inv, R_inv = torch.inverse(Q), torch.inverse(R)

    for start in range(0, len(observations_y), assimilation_window):
        window_observations = observations_y[start:(start + assimilation_window)]
        window_model_input = {k: v[start:(start + assimilation_window)] for k, v in model_input.items()}

        # Initialize the background state for the current window
        if start > 0:
            step_model_input = {k: v[0] for k, v in window_model_input.items()}
            x_background = model(x_assimilated[-1], **step_model_input)

        if method == "standard":
            x0 = standard_4dvar_update(x_background, window_observations, model, obs_operator, Q_inv, R_inv, window_model_input, n_iter, lr)
        elif method == "incremental":
            x0 = incremental_4dvar_update(x_background, window_observations, model, obs_operator, Q_inv, R_inv, window_model_input, n_iter)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Update simulation states
        x_assimilated.append(x0)
        for t in range(1, assimilation_window):  # end with the start of the next window
            step_model_input = {k: v[t] for k, v in window_model_input.items()}
            x_assimilated.append(model(x_assimilated[-1], **step_model_input))

    return torch.stack(x_assimilated)


# Generate data
rainfall_data, evap_coef_data, true_states, sim_states, observations_y, simulations_y = generate_data(time_steps, state_dim, measure_noise_std)

# Run 4DVAR
model_input = {"rainfall": rainfall_data, "evap_coeff": evap_coef_data}
model = state_transition
obs_operator = observation_operator
x_background = sim_states[0]
R = torch.diag((torch.ones((y_dim)) * measure_noise_std) ** 2)
Q = torch.diag((torch.ones((state_dim)) * model_noise_std) ** 2)
assimilation_window = 10
# Run standard 4DVAR
# x_assimilated_standard = vad_4dvar_loop(
#     x_background, observations_y, assimilation_window,
#     model, obs_operator, Q, R, model_input, method="standard", n_iter=10, lr=0.1
# )
# x_assimilated = x_assimilated_standard.clone()
# Run incremental 4DVAR
x_assimilated_incremental = vad_4dvar_loop(
    x_background, observations_y, assimilation_window,
    model, obs_operator, Q, R, model_input, method="incremental", n_iter=1
)
x_assimilated = x_assimilated_incremental.clone()
y_assimilated = np.array([observation_operator(x) for x in x_assimilated])


# Plot state results
obs_df = pd.DataFrame(true_states, columns=[f"state_{i + 1}" for i in range(true_states.shape[1])])
obs_df = obs_df.assign(type='Observed', time=range(true_states.shape[0])).melt(id_vars=['time', 'type'])
sim_df = pd.DataFrame(sim_states, columns=[f"state_{i + 1}" for i in range(sim_states.shape[1])])
sim_df = sim_df.assign(type='Simulated', time=range(sim_states.shape[0])).melt(id_vars=['time', 'type'])
asm_df = pd.DataFrame(x_assimilated, columns=[f"state_{i + 1}" for i in range(x_assimilated.shape[1])])
asm_df = asm_df.assign(type='Assimilated', time=range(x_assimilated.shape[0])).melt(id_vars=['time', 'type'])
df = pd.concat([obs_df, sim_df, asm_df], ignore_index=True)
g = sns.FacetGrid(df, col="variable", hue="type", col_wrap=1, aspect=4)
g.map(sns.lineplot, "time", "value")
g.add_legend()
for i in range(state_dim):
    print(f"RMSE State {i + 1} Sim: {root_mean_squared_error(true_states[:, i], sim_states[:, i])}")
    print(f"RMSE State {i + 1} Asm: {root_mean_squared_error(true_states[:, i], x_assimilated[:, i])}")

# Plot y results
obs_df = pd.DataFrame(observations_y, columns=[f"obs_{i + 1}" for i in range(observations_y.shape[1])])
obs_df = obs_df.assign(type='Observed', time=range(observations_y.shape[0])).melt(id_vars=['time', 'type'])
sim_df = pd.DataFrame(simulations_y, columns=[f"obs_{i + 1}" for i in range(simulations_y.shape[1])])
sim_df = sim_df.assign(type='Simulated', time=range(simulations_y.shape[0])).melt(id_vars=['time', 'type'])
asm_df = pd.DataFrame(y_assimilated, columns=[f"obs_{i + 1}" for i in range(y_assimilated.shape[1])])
asm_df = asm_df.assign(type='Assimilated', time=range(y_assimilated.shape[0])).melt(id_vars=['time', 'type'])
df = pd.concat([obs_df, sim_df, asm_df], ignore_index=True)
g = sns.FacetGrid(df, col="variable", hue="type", col_wrap=1, aspect=4)
g.map(sns.lineplot, "time", "value")
g.add_legend()
for i in range(y_dim):
    print(f"RMSE Obs {i + 1} Sim: {root_mean_squared_error(observations_y[:, i], simulations_y[:, i])}")
    print(f"RMSE Obs {i + 1} Asm: {root_mean_squared_error(observations_y[:, i], y_assimilated[:, i])}")

