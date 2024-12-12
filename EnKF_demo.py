import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import matplotlib.pyplot as plt

ensemble_size = 30
state_dim = 3
y_dim = 1
time_steps = 50
measure_noise_std = .5  # synthetic measurement error
inflation_noise_std = 1.0  # synthetic inflation noise


# Define the state transition model for observed data generation
def true_state_transition(states: NDArray[float], rainfall: NDArray[float], evap_coeff: NDArray[float]) -> NDArray[
    float]:
    '''
    :param states: (state_dim, )
    :param rainfall: (state_dim, )
    :param evap_coeff: (state_dim, )
    :return: (state_dim, )
    '''
    updated_states = states + rainfall - evap_coeff * states
    updated_states = np.maximum(updated_states, 0)  # Ensure non-negative states
    updated_states = np.minimum(updated_states, 50)  # Maximum state
    return updated_states


# Define the state transition model
def state_transition(states: NDArray[float], rainfall: NDArray[float], evap_coeff: NDArray[float]) -> NDArray[float]:
    updated_states = states ** 0.99 * 1.01 + rainfall * 1.02 - evap_coeff * states
    updated_states = np.maximum(updated_states, 0)  # Ensure non-negative states
    updated_states = np.minimum(updated_states, 50)  # Maximum state
    return updated_states


# Define the observation operator for observed data generation
def true_observation_operator(states: NDArray[float], measure_noise_std: float) -> NDArray[float]:
    '''
    :param states: (state_dim, )
    :return: (y_dim, )
    '''
    # Non-linear relationship
    y = np.array([(states.clip(0) ** 0.5).sum()])
    y += np.random.normal(0, measure_noise_std, size=len(y))
    return y


# Define the observation operator
def observation_operator(states: NDArray[float]) -> NDArray[float]:
    y = np.array([(states.clip(0) ** 0.5).sum()])
    return y


# Generate synthetic data
def generate_data(time_steps: int, state_dim: int, measure_noise_std: float):
    rainfall_data = np.random.uniform(0, 20, size=(time_steps, state_dim))
    rainfall_data = (rainfall_data - 10).clip(min=0)
    evap_coef_data = np.random.uniform(0.05, 0.1, size=state_dim)
    evap_coef_data = np.repeat(evap_coef_data[:, None], time_steps, axis=1).T

    # Initialize variables
    true_state_list = [np.random.uniform(20, 40, size=state_dim)]  # Initial true state
    sim_state_list = [np.random.uniform(20, 40, size=state_dim)]  # Initial simulated state

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

    true_states = np.array(true_state_list)
    sim_states = np.array(sim_state_list)
    observations_y = np.array(obs_y_list)
    simulations_y = np.array(sim_y_list)

    return rainfall_data, evap_coef_data, true_states[1:], sim_states[1:], observations_y, simulations_y


# EnKF step
def enkf_update(ensemble: NDArray[Tuple[float, float]],
                observation: NDArray[float],
                model: callable,
                obs_operator: callable,
                R: NDArray[Tuple[float, float]],
                **kwargs
                ) -> NDArray[Tuple[float, float]]:
    '''
    :param ensemble: (ensemble_size, state_dim)
    :param observation: (y_dim, )
    :param model: state transition
    :param obs_operator: observation operator
    :param R: (y_dim, y_dim)
    :param kwargs: external input for the state transition model
    :return: updated ensemble
    '''
    # Forecast step
    forecast_ensemble = np.array([model(state, **kwargs) for state in ensemble])
    H_ensemble = np.array([obs_operator(state) for state in forecast_ensemble])

    # Compute the covariance
    P_f_xy = np.cov(forecast_ensemble.T, H_ensemble.T)[:state_dim, state_dim:]
    # P_f_xy = (forecast_ensemble - forecast_ensemble.mean(axis=0)).T @ (H_ensemble - H_ensemble.mean(axis=0)) / (forecast_ensemble.shape[0] - 1)
    P_f_yy = np.cov(H_ensemble.T)

    # Kalman gain
    K = P_f_xy @ np.linalg.inv(P_f_yy + R)

    # Analysis step
    update_ensemble = np.empty_like(ensemble)
    for i in range(ensemble.shape[0]):
        innovation = observation - H_ensemble[i]
        update_ensemble[i] = forecast_ensemble[i] + K @ innovation

    return update_ensemble


# EnKF loop
def enkf_loop(initial_ensemble: NDArray[Tuple[float, float]],
              observations_y: NDArray[Tuple[float, float]],
              R: NDArray[Tuple[float, float]],
              model: callable,
              obs_operator: callable,
              model_input: dict[str, NDArray[Tuple[float, float]]],
              inflation_noise_std: float
              ) -> NDArray[Tuple[float, float, float]]:
    '''
    :param initial_ensemble: (ensemble_size, state_dim)
    :param observations_y: (time_steps, y_dim)
    :param R: (y_dim, y_dim)
    :param model: state transition
    :param obs_operator: observation operator
    :param model_input: dict of (time_steps, input_dim), external input for the state transition model
    :param inflation_noise_std: standard deviation of inflation noise
    :return: updated ensemble results (time_steps, ensemble_size, state_dim)
    '''
    ensemble = initial_ensemble.copy()
    ensemble_history = []
    for t in range(time_steps):
        observation = observations_y[t]
        step_model_input = {k: v[t] for k, v in model_input.items()}
        ensemble = enkf_update(
            ensemble=ensemble,
            observation=observation,
            model=model,
            obs_operator=obs_operator,
            R=R,
            **step_model_input
        )
        # Add inflation noise to avoid ensemble collapse
        ensemble += np.random.normal(0, inflation_noise_std, size=ensemble.shape)
        ensemble_history.append(ensemble.copy())
    ensemble_history = np.array(ensemble_history)
    return ensemble_history


# Generate data
rainfall_data, evap_coef_data, true_states, sim_states, observations_y, simulations_y = generate_data(time_steps, state_dim, measure_noise_std)

# Run EnKF
R = np.diag(np.repeat(measure_noise_std ** 2, y_dim))
initial_ensemble = np.random.uniform(10, 50, size=(ensemble_size, state_dim))
model = state_transition
obs_operator = observation_operator
model_input = {
    "rainfall": rainfall_data,
    "evap_coeff": evap_coef_data
}
ensemble_history = enkf_loop(initial_ensemble, observations_y, R, model, obs_operator, model_input, inflation_noise_std)

# Plot state results
posterior_mean_ensembles = ensemble_history.mean(axis=1)
ci_lower = np.percentile(ensemble_history, 2.5, axis=1)
ci_upper = np.percentile(ensemble_history, 97.5, axis=1)
colors = plt.cm.viridis(np.linspace(0, 1, state_dim))
plt.figure(figsize=(12, 6))
for i in range(state_dim):
    plt.plot(true_states[:, i], label=f"True State {i + 1}", color=colors[i])
    plt.plot(sim_states[:, i], label=f"Simulated State {i + 1}", linestyle="dotted", color=colors[i])
    plt.plot(posterior_mean_ensembles[:, i], label=f"Posterior Mean State {i + 1}", linestyle="dashed", color=colors[i])
    plt.fill_between(range(time_steps), ci_lower[:, i], ci_upper[:, i], alpha=0.2, label=f"95% CI {i + 1}", color=colors[i])
plt.xlabel("Time Steps")
plt.ylabel("State Values")
plt.title("Model States during EnKF")
plt.legend()
plt.show()

# Plot output y results
posterior_y = np.zeros((time_steps, ensemble_size, y_dim))
for t in range(time_steps):
    for i in range(ensemble_size):
        posterior_y[t, i] = observation_operator(ensemble_history[t][i])
posterior_mean_y = posterior_y.mean(axis=1)
ci_lower_y = np.percentile(posterior_y, 2.5, axis=1)
ci_upper_y = np.percentile(posterior_y, 97.5, axis=1)
colors = plt.cm.viridis(np.linspace(0, 1, y_dim))
plt.figure(figsize=(12, 6))
for i in range(y_dim):
    plt.plot(observations_y[:, i], label=f"Observation y {i + 1}", color=colors[i])
    plt.plot(simulations_y[:, i], label=f"Simulation y {i + 1}", linestyle="dotted", color=colors[i])
    plt.plot(posterior_mean_y[:, i], label=f"Posterior Mean y {i + 1}", linestyle="dashed", color=colors[i])
    plt.fill_between(range(time_steps), ci_lower_y[:, i], ci_upper_y[:, i], alpha=0.2, label=f"95% CI {i + 1}", color=colors[i])
plt.xlabel("Time Steps")
plt.ylabel("Observation Values")
plt.title("Output Variable during EnKF")
plt.legend()
plt.show()
