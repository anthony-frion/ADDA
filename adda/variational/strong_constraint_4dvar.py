from typing import Callable, Type

import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.probability.distributions import Distribution
from adda.system.state import State
from adda.util.optimization import optimize
from adda.util.state_space import rollout
from adda.util.typing import ListOfStates
from adda.variational.base import process_4dvar_inputs, sliding_window_4dvar
from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class SC4DVarLoss(_Loss):
    """Loss function class for strong-constraint 4D-Var."""

    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State = None,
        static_inputs: State = None,
        background_prior: Distribution = None,
        normalize: bool = False,
        state_time_axis: Tensor = None,
        c_proj: Callable = None,
    ):
        """Initialize loss.

        Args:
            m_dyn (Callable): function that takes 4 inputs: x, dt, dynamic_inputs, static_inputs. All inputs are State
            objects except dt, which can be a float or Tensor. In general, this will be a user-supplied function that
            describes the known dynamics of a state space model.
            observations (ObservationSet): set of observations to be assimilated.
            obs_op (ObservationOperator): observation operator used to map between system states and observations
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
            background_prior (Distribution): prior distribution on initial state. Defaults to None.
            normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
            state_time_axis (Tensor): time axis of simulated system states
            c_proj (Callable): optional projection from control-space fields to model-space fields.
                It must take and return a TensorDict with batch size (1, 1).
                The background prior is evaluated in control space before this projection.
        """
        super().__init__()
        (
            self.m_dyn,
            self.observations,
            self.obs_op,
            self.dynamic_inputs,
            self.static_inputs,
            self.background_prior,
            self.c_proj,
        ) = (
            m_dyn,
            observations,
            obs_op,
            dynamic_inputs,
            static_inputs,
            background_prior,
            c_proj,
        )
        self.x0 = State(TensorDict(batch_size=(1, 1)), time_axis=obs_op.time_axis[0:1])
        self.normalize = normalize
        self.state_time_axis = state_time_axis if state_time_axis is not None else obs_op.time_axis

    def forward(self, fields: TensorDict) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for initial system state

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        self.x0.fields = fields
        logp = 0 if self.background_prior is None else self.background_prior.log_prob(self.x0.fields).squeeze(1)
        # logp now has only a single singleton batch dimension
        if self.c_proj is not None:
            projected_fields = self.c_proj(self.x0.fields)
            self.x0.fields = projected_fields

        x_all = rollout(self.m_dyn, self.state_time_axis, self.x0, self.dynamic_inputs, self.static_inputs)

        logp = logp + self.obs_op.log_prob(x_all, self.observations).reshape(
            1,
        )
        if self.normalize:
            logp /= self.obs_op.time_axis.nelement()
        return -logp.mean(dim=0)  # mean over batch axis


def sc4dvar_single_window(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationSet,
    x_init: State = None,
    state_time_axis=None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    background_prior: Distribution = None,
    normalize: bool = False,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    loss_list: list = None,
    check_inputs: bool = True,
    c_proj: Callable = None,
) -> State:
    """Strong-constraint 4D-Var for a single window of observations.

    Args:
        m_dyn: time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        oobservations (ObservationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        x_init (State): first guess of initial state for the first window. uses background prior mean if None.
          Should have a leading time dimension of length 1.
        state_time_axis (float, optional): the time axis for the state predictions during optimization.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (Distribution): prior distribution on initial state
        normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False
        loss_list (list, optional): Has to be empty. If provided, the loss values are inserted in-place.
        check_inputs: if False, input checking is skipped
        c_proj (Callable): optional projection from control-space fields to model-space fields.
            It must take and return a TensorDict with batch size (1, 1).
            The background_prior is evaluated in control space before this projection.

    Returns:
        x0 (State): optimized initial state
    """
    if check_inputs:
        dynamic_inputs, static_inputs, x_init, state_time_axis, _, _, _ = process_4dvar_inputs(
            obs_op,
            m_dyn,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
            state_time_axis=state_time_axis,
            c_proj=c_proj,
        )

    loss = SC4DVarLoss(
        m_dyn,
        observations,
        obs_op,
        dynamic_inputs,
        static_inputs,
        background_prior=background_prior,
        normalize=normalize,
        state_time_axis=state_time_axis,
        c_proj=c_proj,
    )  # instantiate loss

    fields = optimize(
        loss,
        x_init.fields,
        optimizer_class,
        optimizer_pars,
        scheduler_class,
        scheduler_pars,
        n_steps=n_steps,
        verbose=verbose,
        loss_list=loss_list,
    )
    return State(fields, time_axis=x_init.time_axis)


def sc4dvar_sliding_window(*args, **kwargs) -> ListOfStates:
    """Strong-constraint 4D-Var for overlapping windows of observations.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        window_duration (Union[float, int]): window length in time units
        shift_fraction (float, optional): shift as fraction of window. Defaults to 0.25.
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State, optional): first guess of initial state. Uses prior mean if None.
          Should have leading batch, then time dimension of length 1
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (Distribution, optional): background prior for first window. Defaults to None.
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False

    Returns:
        ListOfStates: initial state for each assimilation window
    """
    return sliding_window_4dvar(sc4dvar_single_window, *args, **kwargs)
