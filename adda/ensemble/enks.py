from typing import Callable, Union

import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.system.state import State
from tensordict import TensorDict


class EnKS:
    """Ensemble Kalman Smoother implementation for data assimilation."""

    def __init__(self, n_members, n_variables, lag, initial_ensemble_std=1.0, inflation=1.0, linearize=True):
        """Initialize the EnKS.

        Args:
            n_members (int): Number of ensemble members
            n_variables (int): Number of state variables
            lag (int): Number of time steps on which to perform the analysis
            initial_ensemble_std (Union[float, Tensor]): Standard deviation for ensemble perturbations.
                If it is a float, assume a spherical Gaussian. If it is a Tensor, assume a diagonal Gaussian.
            inflation (float): Inflation factor to apply  on the ensemble after each analysis step
            linearize (bool): Indicates whether to linearize the observation operator or use a nonlinear approximation
        """
        self.n_members = n_members
        self.n_variables = n_variables
        self.lag = lag
        self.initial_ensemble_std = initial_ensemble_std
        self.inflation = inflation
        self.linearize = linearize
        self.ensemble = None

    def initialize_ensemble(self, initial_state: State) -> State:
        """Initialize ensemble members with perturbations around initial state.

        Args:
            initial_state (State): Initial state for the ensemble mean

        Returns:
            State: State object with batch dimension = n_members
        """

        assert isinstance(initial_state, (State)), "initial_state must be a State"
        assert initial_state.n_fields() == 1, "Only states with one field are currently supported for EnKF/EnKS."
        self.key = list(initial_state.fields.keys())[0]
        self.field_shape = initial_state.fields[self.key].shape[2:]
        initial_tensor = initial_state.fields[self.key].squeeze(0)  # Remove batch dimension if present
        initial_tensor = initial_tensor.flatten(1)  # flatten field dimensions
        time_axis = initial_state.time_axis

        # Generate all noise perturbations at once using vectorized operations
        # Shape: (n_members, *initial_tensor.shape)
        noise_shape = (self.n_members,) + initial_tensor.shape
        if isinstance(self.initial_ensemble_std, float):
            noise_perturbations = torch.normal(mean=0.0, std=self.initial_ensemble_std, size=noise_shape)
        elif isinstance(self.initial_ensemble_std, torch.Tensor):
            self.initial_ensemble_std = self.initial_ensemble_std.reshape(*initial_tensor.shape).unsqueeze(0)
            self.initial_ensemble_std = self.initial_ensemble_std.expand(self.n_members, *initial_tensor.shape).clone()
            std = self.initial_ensemble_std  # .unsqueeze(0).expand(*noise_shape)
            noise_perturbations = torch.normal(mean=torch.zeros(*noise_shape), std=std)  # , size=noise_shape)
        else:
            raise RuntimeError(f"Invalid type for initial_ensemble_std: {type(self.initial_ensemble_std)}")

        # Create ensemble by broadcasting and adding noise to initial state
        # initial_tensor shape: (time_steps, n_variables)
        # noise_perturbations shape: (n_members, time_steps, n_variables)
        # Result shape: (n_members, time_steps, n_variables)
        ensemble_tensor = initial_tensor.unsqueeze(0) + noise_perturbations.to(initial_tensor.device)

        ensemble_fields = TensorDict(
            x=ensemble_tensor,
            batch_size=(self.n_members, 1),
        )
        self.ensemble = State(ensemble_fields, time_axis=time_axis)
        return self.ensemble

    @staticmethod
    def compute_kalman_gain(x, H, R):
        """Computes the Kalman gain for the state x.

        Args:
            x (torch.Tensor): State over some domain on which the gain is defined, shape (n_members, lag, n_variables)
            H (torch.Tensor): Observation matrix, shape (n_obs, n_variables)
            R (torch.Tensor): Observation error covariance matrix, shape (n_obs, n_obs)
        """
        assert len(x.shape) == 3, f"Invalid shape for x: {x.shape}"
        n_members, lag, n_variables = x.shape
        xp_lag = x - x.mean(axis=0).reshape(lag, n_variables)  # (n_members, lag, n_variables)
        xp = xp_lag[:, -1].clone()  # (n_members, n_variables)
        xp_lag = xp_lag.flatten(1, 2)  # (n_members, lag*n_variables)
        """
        We want to compute (ignoring factor of n_members - 1):
        1) HBHt = H @ Xt @ X @ Ht = (H @ Xt) @ (X @ Ht) = (X @ Ht).T @ (X @ Ht)
        2) BHt = Xt @ (X @ Ht)
        """
        xHt = xp @ H.T  # (n_members, n_obs)
        HBHt = xHt.T @ xHt / (n_members - 1)  # (n_obs, n_obs)

        M = HBHt + R  # (n_obs, n_obs)
        BHt = xp_lag.T @ xHt / (n_members - 1)  # (lag*n_variables, n_obs)
        K = torch.linalg.solve(M.T, BHt.T).T
        return K

    def compute_kalman_gain_no_linearization(self, x, HXt, R, yhat_ensemble_members):
        """Estimates the Kalman gain for the state x without a linearized observation operator H.

        Args:
            x (torch.Tensor): State over some domain on which the gain is defined, shape (n_members, lag, n_variables)
            HXt (torch.Tensor): Nonlinear approximation of the linear observation,
                using the method of R.N. Bannister 2017 RMetS
            R (torch.Tensor): Observation error covariance matrix, shape (n_obs, n_obs),
                or its diagonal coefficients, shape (n_obs)
        """
        assert len(x.shape) == 3, f"Invalid shape for x: {x.shape}"
        if len(R.shape) == 1:
            R = torch.diag(R)
        n_members, lag, n_variables = x.shape
        xp_lag = x - x.mean(axis=0).reshape(lag, n_variables)  # (n_members, lag, n_variables)
        xp_lag = xp_lag.flatten(1, 2)  # (n_members, lag*n_variables)

        HBHt = HXt.T @ HXt / (n_members - 1)  # (n_obs, n_obs)
        M = HBHt + R  # (n_obs, n_obs)

        BHt = xp_lag.T @ yhat_ensemble_members / (n_members - 1)  # (lag*n_variables, n_obs)
        K = torch.linalg.solve(M.T, BHt.T).T
        return K

    def forecast_step(self, m_dyn, dt, t_idx, dynamic_inputs=None, static_inputs=None):
        """Forecast all ensemble members one time step forward using m_dyn.

        Args:
            m_dyn (Callable): Model dynamics function that handles batched State objects
            dt (float): Time step
            dynamic_inputs (State): Dynamic inputs for the model
            static_inputs (State): Static inputs for the model
        """
        forecasted_state = m_dyn(self.ensemble[:, t_idx - 1 : t_idx], dt, dynamic_inputs, static_inputs)
        if len(self.ensemble.time_axis) > t_idx:
            self.ensemble.fields[self.key][:, t_idx : t_idx + 1] = forecasted_state.fields[
                self.key
            ]  # fill appropriate value
        elif len(self.ensemble.time_axis) == t_idx:
            self.ensemble = self.ensemble.cat(
                forecasted_state, 1
            )  # concatenate the forecasted state to the current ensemble
        else:
            raise RuntimeError(
                f"Attempting to perform a forecast on time step {t_idx}"
                "when self.ensemble has only {len(self.ensemble.time_axis)} time steps"
            )

    def analysis_step(self, H, obs_data, obs_op, R, time_step_idx):
        """Perform the analysis (update) step using observations.

        Args:
            H (torch.Tensor): Observation matrix, shape (n_obs, n_variables)
            obs_data (Tensor): Available observations at the current time step
            obs_op (ObservationOperator): Observation operator used to map between system states and observations
            R (torch.Tensor): Observation error covariance matrix, shape (n_obs, n_obs)
            time_step_idx (int): Current time step index
        """
        n_obs = H.shape[0]
        assert n_obs > 0, "attempting to perform an analysis with no observations, which is not possible"
        assert isinstance(obs_data, torch.Tensor), f"obs_data argument must be a Tensor but got {obs_data}"
        assert isinstance(R, torch.Tensor) and R.shape == (
            n_obs,
            n_obs,
        ), f"R must be a Tensor of size (n_obs, n_obs) but got {R} with n_obs={n_obs}"

        # Compute Kalman gain using the provided function
        next_lag = min(self.lag, time_step_idx + 1)  # shorter lag for first few time steps
        X_lag = self.ensemble.fields[self.key][
            :, time_step_idx - next_lag + 1 : time_step_idx + 1, :
        ]  # Shape: (n_members, current_lag, n_variables)
        K = self.compute_kalman_gain(X_lag, H, R)  # Shape: (current_lag*n_variables, n_obs)

        # Construct the observation noise matrix (computes a Cholesky decomp, so it might be inefficient)
        obs_noise_matrix = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(n_obs, device=R.device), R
        ).sample((self.n_members,))

        # Broadcast observations and add noise for all ensemble members
        y_pert_matrix = obs_data.unsqueeze(0) + obs_noise_matrix  # Shape: (n_members, n_obs)
        # Compute innovations for all ensemble members at once
        # H @ X.T gives observation predictions for all members: (n_obs, n_members)
        # We need (n_members, n_obs), so we use X @ H.T
        yhat_ensemble_members = obs_op.conditional_mean(
            self.ensemble[:, time_step_idx : time_step_idx + 1], time_step_idx, time_step_idx + 1
        )
        yhat_ensemble_members = yhat_ensemble_members[self.key].reshape(
            (self.n_members, n_obs)
        )  # Shape: (n_members, n_obs)
        innovations = y_pert_matrix - yhat_ensemble_members
        # Update all ensemble members at once
        updates = innovations @ K.T  # Shape: (n_members, current_lag*n_variables)
        updated_ensemble = X_lag + updates.reshape_as(X_lag)  # Shape: (n_members, current_lag, n_variables)
        if self.inflation != 1.0:  # inflate the covariance of the last state
            updated_mean = torch.mean(updated_ensemble[:, -1], axis=0, keepdim=True)
            updated_anomalies = updated_ensemble[:, -1] - updated_mean
            updated_ensemble[:, -1] = updated_mean + self.inflation * updated_anomalies

        # Update the last time step of the ensemble with updated_ensemble
        self.ensemble.fields[self.key][:, time_step_idx - next_lag + 1 : time_step_idx + 1] = updated_ensemble

        # Return updated ensemble as tensor for compatibility
        return updated_ensemble  # Shape: (n_members, current_lag, n_variables)

    def analysis_step_no_linearization(self, obs_data, obs_op, R, time_step_idx):
        """Perform the analysis (update) step using observations, without a linearized observation operator H.

        Args:
            obs_data (Tensor): Available observations at the current time step
            obs_op (ObservationOperator): Observation operator used to map between system states and observations
            R (torch.Tensor): Observation error covariance matrix, shape (n_obs) or (n_obs, n_obs)
            time_step_idx (int): Current time step index
            n_obs (int): The number of observations.
                It can be computed from the observations argument, but we only compute it if it is not provided.
        """

        n_obs = obs_data.shape[0]
        assert n_obs > 0, "attempting to perform an analysis with no observations, which is not possible"
        assert isinstance(obs_data, torch.Tensor), f"obs_data argument must be a Tensor but got {obs_data}"
        assert isinstance(R, torch.Tensor) and R.shape == (
            n_obs,
            n_obs,
        ), f"R must be a Tensor of size (n_obs, n_obs) but got {R} with n_obs={n_obs}"

        # Convert ensemble State object to matrix form
        X = self.ensemble[:, time_step_idx : time_step_idx + 1]  # Last state in the ensemble

        # Compute Kalman gain using the provided function
        next_lag = min(self.lag, time_step_idx + 1)  # shorter lag for first few time steps
        X_lag = self.ensemble.fields[self.key][
            :, time_step_idx - next_lag + 1 : time_step_idx + 1, :
        ]  # Shape: (n_members, current_lag, n_variables)
        ensemble_mean = X.mean_on_batch_axis()
        yhat_ensemble_mean = obs_op.conditional_mean(ensemble_mean, time_step_idx, time_step_idx + 1)
        yhat_ensemble_mean = yhat_ensemble_mean[self.key].reshape((1, n_obs))  # Shape: (1, n_obs)
        yhat_ensemble_members = obs_op.conditional_mean(X, time_step_idx, time_step_idx + 1)
        yhat_ensemble_members = yhat_ensemble_members[self.key].reshape(
            (self.n_members, n_obs)
        )  # Shape: (n_members, n_obs)
        H_X = yhat_ensemble_members - yhat_ensemble_mean  # Shape: (n_members, n_obs)

        K = self.compute_kalman_gain_no_linearization(
            X_lag, H_X, R, yhat_ensemble_members
        )  # Shape: (current_lag*n_variables, n_obs)

        # Construct the observation noise matrix (computes a Cholesky decomp, so it might be inefficient)
        obs_noise_matrix = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n_obs), R).sample(
            (self.n_members,)
        )

        # Broadcast observations and add noise for all ensemble members
        y_pert_matrix = obs_data.unsqueeze(0) + obs_noise_matrix  # Shape: (n_members, n_obs)

        # Compute innovations for all ensemble members at once
        innovations = y_pert_matrix - yhat_ensemble_members  # Shape: (n_members, n_obs)
        # Update all ensemble members at once
        updates = innovations @ K.T  # Shape: (n_members, current_lag*n_variables)
        updated_ensemble = X_lag + updates.reshape_as(X_lag)  # Shape: (n_members, current_lag, n_variables)
        if self.inflation != 1.0:  # inflate the covariance of the last state
            updated_mean = torch.mean(updated_ensemble[:, -1], axis=0, keepdim=True)
            updated_anomalies = updated_ensemble[:, -1] - updated_mean
            updated_ensemble[:, -1] = updated_mean + self.inflation * updated_anomalies

        # Update the last time step of the ensemble with updated_ensemble
        self.ensemble.fields[self.key][:, time_step_idx - next_lag + 1 : time_step_idx + 1] = updated_ensemble

        # Return updated ensemble as tensor for compatibility
        return updated_ensemble  # Shape: (n_members, current_lag, n_variables)

    def assimilate(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationSet,
        x_init: State = None,
        dynamic_inputs: State = None,
        static_inputs: State = None,
        verbose: bool = False,
    ) -> tuple:
        """Main EnKS assimilation method.

        Args:
            m_dyn (Callable): Model dynamics function
            observations (ObservationSet): Set of observations to be assimilated
            obs_op (ObservationOperator): Observation operator used to map between system states and observations
            x_init (State): Initial state for the ensemble mean
            dynamic_inputs (State): Extra inputs defined for each input state (e.g. TOA)
            static_inputs (State): Extra inputs that don't vary over time (e.g. bathymetry)
            verbose (bool): If True, print detailed progress and diagnostic information

        Returns:
            - ensemble_state (State): Full ensemble State object with history (Default option)
        """
        # Initialize ensemble. This flattens all field dimensions.
        if x_init is None:
            raise ValueError("x_init must be provided to initialize the ensemble")

        initial_ensemble = self.initialize_ensemble(x_init)
        self.ensemble = State(
            TensorDict(
                {
                    self.key: torch.zeros_like(initial_ensemble.fields[self.key]).expand(
                        -1, len(observations.state.time_axis), -1
                    )
                }
            ).clone(),
            observations.state.time_axis,
        )
        self.ensemble.fields[self.key][:, 0:1] = initial_ensemble.fields[self.key]

        # Get time step from observation operator - throw error if insufficient time steps
        if len(observations.state.time_axis) < 2:
            raise ValueError(
                f"Observation operator time_axis must have at least 2 time steps to compute dt, "
                f"but got {len(obs_op.time_axis)} time step(s). "
                f"Cannot determine time step for assimilation."
            )
        self.R = (
            torch.linalg.inv(obs_op.P_chol.fields[self.key][0, 0] @ obs_op.P_chol.fields[self.key][0, 0].T)
            if hasattr(obs_op, "P_chol")
            else None
        )

        assert self.R is None or (
            len(self.R.shape) == 2 and self.R.shape[0] == self.R.shape[1]
        ), f"invalid shape for R: {self.R.shape}"

        for t_idx in range(1, len(observations.state.time_axis)):
            dt = observations.state.time_axis[t_idx] - observations.state.time_axis[t_idx - 1]
            di = dynamic_inputs[:, t_idx - 1 : t_idx] if dynamic_inputs is not None else None
            self.forecast_step(m_dyn, dt, t_idx, di, static_inputs)

            if hasattr(obs_op, "mask"):
                # Get the mask for this time step to identify observed variables
                mask_t = observations.mask.fields[self.key][0, t_idx, :]  # Shape: (n_variables,)
                # Extract observed values where mask is True
                observations_t = observations.state.fields[self.key][0, t_idx, mask_t]  # Shape: (n_obs,)
                # Extract sigma values only at observed locations
                if self.R is None:
                    sigma_obs = obs_op.sigma.fields[self.key][0, t_idx, mask_t]  # Shape: (n_obs,)
            else:
                observations_t = observations.state.fields[self.key][0, t_idx]
                if self.R is None:
                    sigma_obs = obs_op.sigma.fields[self.key][0, t_idx]

            if observations_t.shape[0] > 0:
                # Create diagonal R matrix with variances (sigma^2) at observed locations only
                R_t = torch.diag(sigma_obs**2) if self.R is None else self.R  # Shape: (n_obs) or (n_obs, n_obs)
                if self.linearize:
                    assert hasattr(obs_op, "linearize"), "The observation operator must have a 'linearize' method."
                    # Linearize observation operator at current time step to get H matrix
                    H_t = obs_op.linearize(idx_time=t_idx)
                    self.analysis_step(H_t, observations_t, obs_op, R_t, t_idx)
                else:
                    self.analysis_step_no_linearization(observations_t, obs_op, R_t, t_idx)
            else:
                print("No observations available, skipping analysis step")

        # Final summary
        final_ensemble = self.ensemble.fields[self.key][:, -1, :].squeeze(1)  # Shape: (n_members, n_variables)
        final_spread = final_ensemble.std(dim=0).mean()

        if verbose:
            final_mean = final_ensemble.mean(dim=0)
            print("Final ensemble statistics:")
            print(f"  Number of members: {self.n_members}")
            print(f"  State dimension: {self.n_variables}")
            print(f"  Final spread: {final_spread:.4f}")
            print(f"  Final mean range: [{final_mean.min():.4f}, {final_mean.max():.4f}]")

        # un-flatten the final ensemble with the orignial field shape
        ensemble_shape = self.ensemble.fields[self.key].shape
        self.ensemble.fields[self.key] = self.ensemble.fields[self.key].reshape(
            ensemble_shape[0], ensemble_shape[1], *self.field_shape
        )
        return self.ensemble

    def get_ensemble_statistics(self):
        """Compute ensemble statistics.

        Returns:
            dict: Dictionary containing ensemble mean, std, and individual members
        """
        if self.ensemble is None:
            return None

        # Extract ensemble matrix from State object
        # Get the last time step: shape (n_members, n_variables)
        ensemble_matrix = self.ensemble.fields[self.key][:, -1, :]

        return {
            "mean": ensemble_matrix.mean(dim=0),  # Mean across ensemble members
            "std": ensemble_matrix.std(dim=0),  # Std across ensemble members
            "members": ensemble_matrix,  # Individual ensemble members
            "spread": ensemble_matrix.std(dim=0).mean(),  # Average spread across variables
            "full_history": self.ensemble.fields[self.key],  # Full ensemble history
            "time_axis": self.ensemble.time_axis,  # Time axis
            "n_time_steps": self.ensemble.fields[self.key].shape[1],  # Number of time steps
        }


def enks(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    lag=10,
    n_members: int = 20,
    initial_ensemble_std: Union[float, torch.Tensor] = 1.0,
    inflation: float = 1.0,
    linearize: bool = True,
    verbose: bool = False,
) -> tuple:
    """Perform ensemble Kalman smoother data assimilation using EnKS.assimilate() method.

    This function is a wrapper around the EnKS.assimilate() method for backwards compatibility
    and consistent API design.

    Args:
        m_dyn (Callable): Model dynamics function
        observations (ObservationSet): Set of observations to be assimilated
        obs_op (ObservationOperator): Observation operator used to map between system states and observations
        x_init (State): Initial state for the ensemble mean
        dynamic_inputs (State): Extra inputs defined for each input state (e.g. TOA)
        static_inputs (State): Extra inputs that don't vary over time (e.g. bathymetry)
        lag (int): Number of time steps on which the smoothing is performed
        n_members (int): Number of ensemble members
        initial_ensemble_std (Union[float, torch.Tensor]): Standard deviation for ensemble perturbations.
            If it is a float, assume a spherical Gaussian. If it is a Tensor, assume a diagonal Gaussian.
        inflation (float): Inflation factor to apply  on the ensemble after each analysis step
        linearize (bool): Indicates whether to linearize the observation operator or use a nonlinear approximation
        verbose (bool): If True, print detailed progress and diagnostic information

    Returns:
            - ensemble_state (State): Full ensemble State object with history (Default option)
    """

    assert hasattr(obs_op, "time_axis"), "The obs_op argument must have a time_axis attribute for EnKF/EnKS"

    # Simply call the assimilate method of the EnKS object
    key = list(x_init.fields.keys())[0]

    with torch.no_grad():
        enks_obj = EnKS(
            n_members=n_members,
            n_variables=x_init.fields[key].shape[-1],
            lag=lag,
            initial_ensemble_std=initial_ensemble_std,
            inflation=inflation,
            linearize=linearize,
        )
        return enks_obj.assimilate(
            m_dyn=m_dyn,
            observations=observations,
            obs_op=obs_op,
            x_init=x_init,
            dynamic_inputs=dynamic_inputs,
            static_inputs=static_inputs,
            verbose=verbose,
        )
