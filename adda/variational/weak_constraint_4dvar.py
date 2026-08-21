from typing import Callable, Type

import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.probability.distributions import Distribution
from adda.system.state import State
from adda.util.optimization import optimize
from adda.util.typing import ListOfStates
from adda.variational.base import process_4dvar_inputs, sliding_window_4dvar
from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class WC4DVarLoss(_Loss):
    """Base object for representing the weak-constraint 4D-Var loss function."""

    def __init__(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationOperator,
        dynamic_inputs: State,
        static_inputs: State,
        model_error_distribs: Distribution,
        alpha: float = 1.0,
        background_prior: Distribution = None,
        normalize: bool = False,
        state_time_axis: Tensor = None,
    ):
        """Loss for weak constraint 4dvar.

        Args:
            one_step_func (Callable): callable that computes advancement of the state by one time step
            observations (ObservationSet): observations for each time step
            obs_op (ObservationOperator): observation operators for each time step
            dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
            static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
            model_error_distribs (Distribution): probability distributions of the model errors at each time step
            alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
            background_prior (Distribution, optional): prior on initial state. Defaults to None.
            normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
            state_time_axis (Tensor, optional): The time axis for the optimized x. Has to be regular.
                If unspecified, tries using obs_op.time_axis instead.
        """
        super().__init__()
        (
            self.m_dyn,
            self.observations,
            self.obs_op,
            self.dynamic_inputs,
            self.static_inputs,
            self.model_error_distribs,
            self.alpha,
            self.background_prior,
            self.normalize,
        ) = (
            m_dyn,
            observations,
            obs_op,
            dynamic_inputs,
            static_inputs,
            model_error_distribs,
            alpha,
            background_prior,
            normalize,
        )
        self.state_time_axis = obs_op.time_axis if state_time_axis is None else state_time_axis
        assert len(self.state_time_axis) >= 2, f"wc4dvar cannot be used with the state time axis {self.state_time_axis}"
        assert torch.allclose(
            self.state_time_axis.diff(), self.state_time_axis.diff()[0], atol=1e-6, rtol=1e-3
        ), f"Only regular state_time_axis are supported for wc4dvar, but the provided one is {self.state_time_axis}"
        self.state_time_axis = state_time_axis
        self.dt = self.state_time_axis.diff()[0]
        self.x = State(TensorDict(batch_size=(1, self.state_time_axis.nelement())), self.state_time_axis)

    def forward(self, fields: TensorDict) -> Tensor:
        """Compute loss function.

        Args:
            fields (TensorDict): variable fields for system state trajectory

        Returns:
            Tensor: loss value, summed over fields and all dimensions, and averaged over the batch
        """
        self.x.fields = fields
        x0, x1 = self.x[:, :-1], self.x[:, 1:]
        x1_pred = self.m_dyn(x0, self.dt, dynamic_inputs=self.dynamic_inputs, static_inputs=self.static_inputs)

        logp = 0.0 if self.background_prior is None else self.background_prior.log_prob(self.x.fields[:, :1]).squeeze(1)

        # logp now has only a single singleton batch dimension

        logp = logp + self.obs_op.log_prob(self.x, self.observations).reshape(
            1,
        )
        logp += self.alpha * self.model_error_distribs.log_prob(x1_pred.fields - x1.fields).sum(
            axis=1
        )  # sum over time axis
        if self.normalize:
            logp /= self.obs_op.time_axis.nelement()
        return -logp.mean(dim=0)  # mean over batch axis


def wc4dvar_single_window(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationOperator,
    model_error_distribs: Distribution,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    alpha: float = 1.0,
    background_prior: Distribution = None,
    normalize: bool = False,
    state_time_axis: Tensor = None,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    scheduler_class: Type[LRScheduler] = None,
    scheduler_pars: dict = None,
    n_steps: int = 10,
    verbose: bool = False,
    loss_list: list = None,
    check_inputs: bool = True,
    rollout_init: bool = True,
):
    """Weak constraint 4DVAR for a single window of observations.

    Args:
        m_dyn: time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): The set of observations on which to compute the observation error.
        obs_op (ObservationOperator): The observation operator used to compute the observation error.
        model_error_distribs: distributions of model errors at each time step
        x_init (State): first guess of state trajectory. Uses background prior mean if None. Should have a leading batch
          dimension, followed by a time dimension.
        state_time_axis (Tensor, optional): the time axis for the optimized x. Has to be regular.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        alpha (float, optional): weighting factor for model error loss term. Defaults to 1.0.
        background_prior: prior distribution of the initial state, defaults to None.
        normalize (bool): if True, normalizes the assimilation cost by the number of time steps. Defaults to False.
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
            If unspecified, uses the time axis of x_init if possible, otherwise the time axis of obs_op.
        optimizer: class of a pytorch optimizer
        optimizer_pars: dictionary containing parameters for optimizer (learning rate etc.)
        n_steps: int indicating the number of optimizer steps
        verbose (bool): if True, print loss at each iteration. default is False
        loss_list (list, optional): Has to be empty. If provided, the loss values are inserted in-place.
        check_inputs: if False, input checking is skipped
        rollout_init (bool): indicates whether to copy or rollout the input state if it is only one time step
    Returns:
        x0: optimized state trajectory. first dimension is batch, second is time.
    """
    if check_inputs:
        dynamic_inputs, static_inputs, x_init, state_time_axis, _, _, model_error_distribs = process_4dvar_inputs(
            obs_op,
            m_dyn,
            dynamic_inputs,
            static_inputs,
            x_init=x_init,
            background_prior=background_prior,
            observations=observations,
            model_error_distribs=model_error_distribs,
            state_time_axis=state_time_axis,
            rollout_init=rollout_init,
        )
    loss = WC4DVarLoss(
        m_dyn,
        observations,
        obs_op,
        dynamic_inputs,
        static_inputs,
        model_error_distribs,
        alpha=alpha,
        background_prior=background_prior,
        normalize=normalize,
        state_time_axis=state_time_axis,
    )

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


def wc4dvar_sliding_window(*args, **kwargs) -> ListOfStates:
    """Weak constraint 4DVAR for overlapping windows of observations.

    Args:
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        observations (ObservationSet): tensor or list of observations at each time point
        obs_op (ObservationOperator): list of observation operators at each time point
        window_duration (Union[float, int]): window length in time units
        shift_fraction (float, optional): shift as fraction of window. Defaults to 0.25.
        model_error_distribs (Distribution): distributions of model errors at each time step (None for hard constraint)
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State): first guess of state trajectory. Uses prior mean if None.
          Should have leading batch, then time dimension
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
            Has to be regular.
            If unspecified, uses the time axis of x_init if possible, otherwise the time axis of obs_op.
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (Distribution, optional): background prior for first window. Defaults to None.
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False

    Returns:
        ListOfStates: state trajectory for each assimilation window
    """
    return sliding_window_4dvar(wc4dvar_single_window, *args, **kwargs)
