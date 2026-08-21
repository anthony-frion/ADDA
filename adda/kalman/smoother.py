import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import LinearGaussianObsOp
from adda.system.dynamics import LinearDynamics
from adda.system.state import State
from tensordict import TensorDict


def kalman_smoother(
    m_dyn: LinearDynamics,
    observations: ObservationSet,
    obs_op: LinearGaussianObsOp,
    initial_mean: State,
    initial_covariance: State,
    model_error_covariance: torch.Tensor = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
):
    """RTS Kalman smoother.

    Args:
        m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): The set of observations on which to compute the observation error
        obs_op (ObservationOperator): The observation operator used to compute the observation error
        initial_mean (State): Mean of the initial state
        initial_covariance (State): Covariance matrix of the initial state
        model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics.
            If None, it is assumed that there is no model error.
        dynamic_inputs (State): ignored
        static_inputs (State): ignored
    Returns:
        state_mean, state_covariance (State, State): The assimilated mean and covariance of the state over time.
    """
    if isinstance(m_dyn, torch.Tensor):
        m_dyn = LinearDynamics(m_dyn)
    else:
        assert isinstance(m_dyn, LinearDynamics), "m_dyn must be of type LinearDynamics"
    assert (
        dynamic_inputs is None and static_inputs is None
    ), "dynamic inputs and static inputs are not supported for Kalman filters and smoothers."

    ks = RTSKalmanSmoother(initial_mean, initial_covariance, m_dyn, model_error_covariance)
    return ks.assimilate(observations, obs_op)


def fixed_lag_kalman_smoother(
    m_dyn: LinearDynamics,
    observations: ObservationSet,
    obs_op: LinearGaussianObsOp,
    initial_mean: State,
    initial_covariance: State,
    model_error_covariance: torch.Tensor = None,
    lag: int = 1,
    dynamic_inputs: State = None,
    static_inputs: State = None,
):
    """Fixed-lag Kalman smoother.

    Args:
        m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): The set of observations on which to compute the observation error
        obs_op (ObservationOperator): The observation operator used to compute the observation error
        initial_mean (State): Mean of the initial state
        initial_covariance (State): Covariance matrix of the initial state
        model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics.
            If None, it is assumed that there is no model error.
        lag (int): number of time steps over which the smoothing is performed.
            With our convention, the Kalman filter corresponds to lag=1.
        dynamic_inputs (State): ignored
        static_inputs (State): ignored
    Returns:
        state_mean, state_covariance (State, State): The assimilated mean and covariance of the state over time.
    """
    if isinstance(m_dyn, torch.Tensor):
        m_dyn = LinearDynamics(m_dyn)
    else:
        assert isinstance(m_dyn, LinearDynamics), "m_dyn must be of type LinearDynamics"
    assert (
        dynamic_inputs is None and static_inputs is None
    ), "dynamic inputs and static inputs are not supported for Kalman filters and smoothers."

    ks = FixedLagKalmanSmoother(initial_mean, initial_covariance, m_dyn, model_error_covariance, lag)
    return ks.assimilate(observations, obs_op)


class KalmanSmoother:
    """Parent class for all variations of Kalman smoothers."""

    def __init__(
        self,
        initial_mean: State,
        initial_covariance: State,
        m_dyn: LinearDynamics,
        model_error_covariance: torch.Tensor,
    ):
        """Initialize the Kalman smoother.

        Args:
            initial_mean (State): Mean of the initial state
            initial_covariance (State): Covariance matrix of the initial state
            m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
            model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics.
                If None, it is assumed that there is no model error.
        """
        assert (
            initial_mean.n_fields() == 1 and initial_covariance.n_fields() == 1
        ), "Only states with one field are currently supported for Kalman filter/smoother."
        assert (
            list(initial_mean.fields.keys())[0] == list(initial_covariance.fields.keys())[0]
        ), "The fields of the initial mean and covariance are inconsistent."
        self.key = list(initial_mean.fields.keys())[0]
        self.n_variables = initial_mean.fields[self.key].shape[-1]
        assert initial_mean.fields[self.key].shape[1:] == (
            1,
            self.n_variables,
        ), "The shape of the initial mean's field must be (Batch, 1, N_dim)"
        assert initial_covariance.fields[self.key].shape[1:] == (
            1,
            self.n_variables,
            self.n_variables,
        ), "The shape of the initial covariance's field must be (Batch, 1, N_dim, N_dim)"
        if model_error_covariance is not None:
            assert isinstance(model_error_covariance, torch.Tensor) and model_error_covariance.shape[1:] == (
                self.n_variables,
                self.n_variables,
            ), "model error covariance must be a Tensor with shape (1, N_dim, N_dim) or (Batch, N_dim, N_dim)"
            if torch.sum(torch.abs(model_error_covariance)) < 1e-12:
                model_error_covariance = None
        self.model_error_covariance = model_error_covariance
        assert isinstance(m_dyn.operator, torch.Tensor) and m_dyn.operator.shape == (
            self.n_variables,
            self.n_variables,
        ), "the shape of m_dyn.operator must be (N_dim, N_dim)"
        self.mean = initial_mean.clone()
        self.covariance = initial_covariance.clone()
        self.m_dyn = m_dyn
        self.lag = 1  # by default, corresponds to a filter

    def compute_kalman_gain(self, covariance_lag, H, obs_covariance, no_inverse=True):
        """Computes the Kalman gain for the current covariance."""
        Sigma_Ht = covariance_lag @ H.fields[self.key].transpose(2, 3)  # Shape: (1, 1, n_variables, n_obs)
        if obs_covariance is None:
            obs_covariance = 0
        if no_inverse:
            LHS = H.fields[self.key] @ covariance_lag @ H.fields[self.key].transpose(2, 3) + obs_covariance
            K = torch.linalg.solve(LHS.transpose(-2, -1), Sigma_Ht.transpose(-2, -1)).transpose(-2, -1)
        else:
            K = Sigma_Ht @ torch.linalg.inv(
                (H.fields[self.key] @ covariance_lag @ H.fields[self.key].transpose(2, 3) + obs_covariance)
            )
        return K

    def forecast_step(self, dt):
        """Perform one forecast step for the mean and variance."""

        # Update the mean
        current_mean_state = self.mean.restrict_time_domain(self.mean.time_axis[-1], 2 * self.mean.time_axis[-1])
        next_mean_state = self.m_dyn(current_mean_state, dt, noiseless=True)
        self.mean = self.mean.cat(next_mean_state, 1)

        # Update the covariance
        current_covariance_tensor = self.covariance.fields[self.key][:, -1:]  # Shape: (batch, n_variables, n_variables)
        M = self.m_dyn.operator  # Tensor, Shape: (n_variables, n_variables)
        next_covariance_tensor = M @ current_covariance_tensor @ M.T  # Shape: (batch, n_variables, n_variables)
        if self.model_error_covariance is not None:
            # print("Adding model error covariance")
            next_covariance_tensor += self.model_error_covariance
        next_covariance_state = State(
            TensorDict({self.key: next_covariance_tensor}), self.covariance.time_axis[-1:] + dt
        )
        self.covariance = self.covariance.cat(next_covariance_state, 1)

    def analysis_step(self, observations_t, obs_op, obs_covariance):
        """Perform one analysis step for the mean and variance."""
        H = obs_op.H
        next_lag = min(self.lag, self.covariance.fields[self.key].shape[1])  # shorter lag for first few time steps
        covariance_lag = self.covariance.fields[self.key][
            :, -next_lag:
        ]  # Shape: (batch, lag, n_variables, n_variables)
        K = self.compute_kalman_gain(covariance_lag, H, obs_covariance)  # Shape: (n_obs, n_obs)
        H = State(H.fields, self.mean[:, -1:].time_axis)
        anomalies = (observations_t - obs_op.conditional_mean(self.mean)[self.key][:, -1:]).unsqueeze(-1)
        update_to_mean = (K @ anomalies).squeeze(-1)
        self.mean.fields[self.key][:, -next_lag:] = self.mean.fields[self.key][:, -next_lag:] + update_to_mean
        self.covariance.fields[self.key][:, -next_lag:] = (
            self.covariance.fields[self.key][:, -next_lag:] - K @ H.fields[self.key] @ covariance_lag
        )


class FixedLagKalmanSmoother(KalmanSmoother):
    """Fixed-lag Kalman smoother implementation."""

    def __init__(
        self,
        initial_mean: State,
        initial_covariance: State,
        m_dyn: LinearDynamics,
        model_error_covariance: torch.Tensor,
        lag: int,
    ):
        """Initialize the fixed-lag Kalman smoother.

        Args:
            initial_mean (State): Mean of the initial state
            initial_covariance (State): Covariance matrix of the initial state
            m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
            model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics.
                If None, it is assumed that there is no model error.
            lag (int): number of time steps over which the smoothing is performed.
                With our conventions, the Kalman filter corresponds to lag=1.
        """
        super().__init__(initial_mean, initial_covariance, m_dyn, model_error_covariance)
        self.lag = lag

    def assimilate(self, observations: ObservationSet, obs_op: LinearGaussianObsOp):
        """Assimilate the mean and covariance of the state over time."""

        # Get time step from observation operator - throw error if insufficient time steps
        if len(observations.state.time_axis) < 2:
            raise ValueError(
                f"The observation time axis must have at least 2 time steps to compute dt, "
                f"but got {len(observations.state.time_axis)} time step(s). "
                f"Cannot determine time step for assimilation."
            )

        # bruteforce computation of obs covariance from the Cholesky factor of its inverse
        assert isinstance(obs_op, LinearGaussianObsOp), f"obs_op must be a LinearGaussianObsOp, not a {type(obs_op)}"
        obs_covariance = torch.linalg.inv(obs_op.P_chol.fields[self.key][0, 0] @ obs_op.P_chol.fields[self.key][0, 0].T)
        obs_covariance = obs_covariance.reshape(1, 1, *obs_covariance.shape)

        self.analysis_step(observations.state.fields[self.key][:, 0], obs_op, obs_covariance)
        for t_idx in range(1, len(observations.state.time_axis)):
            dt = observations.state.time_axis[t_idx] - observations.state.time_axis[t_idx - 1]
            self.forecast_step(dt)
            observations_t = observations.state.fields[self.key][:, t_idx]  # Shape: (n_obs,)
            self.analysis_step(observations_t, obs_op, obs_covariance)

        return self.mean, self.covariance


class RTSKalmanSmoother(KalmanSmoother):
    """Rauch–Tung–Striebel (RTS) Kalman smoother implementation."""

    def __init__(
        self,
        initial_mean: State,
        initial_covariance: State,
        m_dyn: LinearDynamics,
        model_error_covariance: torch.Tensor,
    ):
        """Initialize the fixed-lag Kalman smoother.

        Args:
            initial_mean (State): Mean of the initial state
            initial_covariance (State): Covariance matrix of the initial state
            m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
            model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics.
                If None, it is assumed that there is no model error.
        """
        super().__init__(initial_mean, initial_covariance, m_dyn, model_error_covariance)

    def forward_pass(self, observations: ObservationSet, obs_op: LinearGaussianObsOp, obs_covariance: torch.Tensor):
        B, N_dim = self.mean.fields[self.key].shape[0], self.mean.fields[self.key].shape[-1]
        means_a_priori = torch.zeros(B, len(observations.state.time_axis) - 1, N_dim)
        covariances_a_priori = torch.zeros(B, len(observations.state.time_axis) - 1, N_dim, N_dim)
        for t_idx in range(1, len(observations.state.time_axis)):
            dt = observations.state.time_axis[t_idx] - observations.state.time_axis[t_idx - 1]
            self.forecast_step(dt)
            means_a_priori[:, t_idx - 1] = self.mean.fields[self.key][:, -1].clone()
            covariances_a_priori[:, t_idx - 1] = self.covariance.fields[self.key][:, -1].clone()

            observations_t = observations.state.fields[self.key][:, t_idx]  # Shape: (n_obs,)
            self.analysis_step(observations_t, obs_op, obs_covariance)

        return means_a_priori, covariances_a_priori

    def backward_pass(self, means_a_priori, covariances_a_priori, no_inverse=True):
        M = self.m_dyn.operator
        for t_idx in range(self.mean.fields[self.key].shape[1] - 2, -1, -1):
            if no_inverse:
                LHS = covariances_a_priori[:, t_idx]
                C = torch.linalg.solve(
                    LHS.transpose(-2, -1), (self.covariance.fields[self.key][:, t_idx] @ M.T).transpose(-2, -1)
                ).transpose(-2, -1)
            else:
                C = self.covariance.fields[self.key][:, t_idx] @ M.T @ torch.linalg.inv(covariances_a_priori[:, t_idx])
            self.mean.fields[self.key][:, t_idx] += (
                C @ (self.mean.fields[self.key][:, t_idx + 1] - means_a_priori[:, t_idx]).unsqueeze(-1)
            ).squeeze(-1)
            self.covariance.fields[self.key][:, t_idx] += (
                C
                @ (self.covariance.fields[self.key][:, t_idx + 1] - covariances_a_priori[:, t_idx])
                @ C.transpose(-2, -1)
            )

    def assimilate(self, observations: ObservationSet, obs_op: LinearGaussianObsOp):
        # Get time step from observation operator - throw error if insufficient time steps
        if len(observations.state.time_axis) < 2:
            raise ValueError(
                f"The observation time axis must have at least 2 time steps to compute dt, "
                f"but got {len(observations.state.time_axis)} time step(s). "
                f"Cannot determine time step for assimilation."
            )

        # bruteforce computation of obs covariance from the Cholesky factor of its inverse
        assert isinstance(obs_op, LinearGaussianObsOp), f"obs_op must be a LinearGaussianObsOp, not a {type(obs_op)}"
        obs_covariance = torch.linalg.inv(obs_op.P_chol.fields[self.key][0, 0] @ obs_op.P_chol.fields[self.key][0, 0].T)
        obs_covariance = obs_covariance.reshape(1, 1, *obs_covariance.shape)

        self.analysis_step(observations.state.fields[self.key][:, 0], obs_op, obs_covariance)
        means_a_priori, covariances_a_priori = self.forward_pass(observations, obs_op, obs_covariance)
        self.backward_pass(means_a_priori, covariances_a_priori)
        return self.mean, self.covariance
