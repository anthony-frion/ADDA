import torch
from adda.kalman.smoother import FixedLagKalmanSmoother
from adda.observation.data import ObservationSet
from adda.observation.operators import LinearGaussianObsOp
from adda.system.dynamics import LinearDynamics
from adda.system.state import State


def kalman_filter(
    m_dyn: LinearDynamics,
    observations: ObservationSet,
    obs_op: LinearGaussianObsOp,
    initial_mean: State,
    initial_covariance: State,
    model_error_covariance: torch.Tensor = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
):
    """Kalman filter.

    Args:
        m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): The set of observations on which to compute the observation error
        obs_op (ObservationOperator): The observation operator used to compute the observation error
        initial_mean (State): Mean of the initial state
        initial_covariance (State): Covariance matrix of the initial state
        model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics
        dynamic_inputs (State): ignored
        static_inputs (State): ignored
    Returns:
        xhat: optimized state trajectory. first dimension is batch, second is time.
    """
    if isinstance(m_dyn, torch.Tensor):
        m_dyn = LinearDynamics(m_dyn)
    else:
        assert isinstance(m_dyn, LinearDynamics), "m_dyn must be of type LinearDynamics"
    assert (
        dynamic_inputs is None and static_inputs is None
    ), "dynamic inputs and static inputs are not supported for Kalman filters and smoothers."

    kf = KalmanFilter(initial_mean, initial_covariance, m_dyn, model_error_covariance)
    return kf.assimilate(observations, obs_op)


class KalmanFilter(FixedLagKalmanSmoother):
    """Kalman filter implementation."""

    def __init__(
        self,
        initial_mean: State,
        initial_covariance: State,
        m_dyn: LinearDynamics,
        model_error_covariance: torch.Tensor,
    ):
        """Initialize the Kalman filter.

        Args:
            initial_mean (State): Mean of the initial state
            initial_covariance (State): Covariance matrix of the initial state
            m_dyn (LinearDynamics): linear time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
            model_error_covariance (Tensor, optional): Covariance matrix for one time step of the state dynamics
        """
        lag = 1
        super().__init__(initial_mean, initial_covariance, m_dyn, model_error_covariance, lag)
