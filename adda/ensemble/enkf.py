from typing import Callable, Union

import torch
from adda.ensemble.enks import EnKS
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.system.state import State


class EnKF(EnKS):
    """Ensemble Kalman Filter implementation for data assimilation.

    We consider an EnKF to be nothing more than an EnKS with time lag 1.
    """

    def __init__(self, n_members, n_variables, initial_ensemble_std=1.0, inflation=1.0, linearize=True):
        """Initialize the EnKF.

        Args:
            n_members (int): Number of ensemble members
            n_variables (int): Number of state variables
            initial_ensemble_std (Union[float, Tensor]): Standard deviation for ensemble perturbations.
                If it is a float, assume a spherical Gaussian. If it is a Tensor, assume a diagonal Gaussian.
            inflation (float): Inflation factor to apply to the ensemble after each analysis step
            linearize (bool): Indicates whether to linearize the observation operator or use a nonlinear approximation
        """
        lag = 1  # we see an EnKF as an EnKS with lag 1
        super().__init__(
            n_members=n_members,
            n_variables=n_variables,
            lag=lag,
            initial_ensemble_std=initial_ensemble_std,
            inflation=inflation,
            linearize=linearize,
        )


def enkf(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    n_members: int = 20,
    initial_ensemble_std: Union[float, torch.Tensor] = 1.0,
    inflation: float = 1.0,
    linearize: bool = True,
    verbose: bool = False,
) -> tuple:
    """Perform ensemble Kalman filter data assimilation using EnKF.assimilate() method.

    This function is a wrapper around the EnKF.assimilate() method for backwards compatibility
    and consistent API design.

    Args:
        m_dyn (Callable): Model dynamics function
        observations (ObservationSet): Set of observations to be assimilated
        obs_op (ObservationOperator): Observation operator used to map between system states and observations
        x_init (State): Initial state for the ensemble mean
        dynamic_inputs (State): Extra inputs defined for each input state (e.g. TOA)
        static_inputs (State): Extra inputs that don't vary over time (e.g. bathymetry)
        n_members (int): Number of ensemble members
        initial_ensemble_std (Union[float, Tensor]): Standard deviation for ensemble perturbations.
            If it is a float, assume a spherical Gaussian. If it is a Tensor, assume a diagonal Gaussian.
        inflation (float): Inflation factor to apply  on the ensemble after each analysis step
        linearize (bool): Indicates whether to linearize the observation operator or use a nonlinear approximation
        verbose (bool): If True, print detailed progress and diagnostic information

    Returns:
            - ensemble_state (State): Full ensemble State object with history (Default option)
    """
    # Simply call the assimilate method of the EnKF object
    assert isinstance(x_init, (State)), "x_init must be a State"
    assert x_init.n_fields() == 1, "Only states with one field are supported for EnKF now."
    key = list(x_init.fields.keys())[0]

    with torch.no_grad():
        enkf_obj = EnKF(
            n_members=n_members,
            n_variables=x_init.fields[key].shape[-1],
            initial_ensemble_std=initial_ensemble_std,
            inflation=inflation,
            linearize=linearize,
        )
        return enkf_obj.assimilate(
            m_dyn=m_dyn,
            observations=observations,
            obs_op=obs_op,
            x_init=x_init,
            dynamic_inputs=dynamic_inputs,
            static_inputs=static_inputs,
            verbose=verbose,
        )
