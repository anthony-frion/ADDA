import torch
from adda.system.state import State
from tensordict import TensorDict
from torch import Tensor


class LinearDynamics:
    """This class implements a linear time stepping operator compatible with the call signature of dynamics functions
    used by the da-tools package.

    Instances of this class act as functions that take inputs x, dt, dynamic_inputs and static_inputs.
    """

    def __init__(self, operator: Tensor, model_error_covariance: Tensor = None):
        """Initialize linear dynamics.

        Args:
            operator (Tensor): Matrix describing linear dynamics. Rows are outputs, columns are inputs.
            model_error_covariance (Tensor): Covariance matrix for one step of the dynamics
        """
        assert operator.ndim == 2 and operator.shape[-2] == operator.shape[-1], "operator must be a square matrix"
        self.operator = operator
        if model_error_covariance is not None:
            assert (
                isinstance(model_error_covariance, Tensor)
                and model_error_covariance.ndim == 2
                and model_error_covariance.shape[0] == model_error_covariance.shape[1]
            ), "model_error_covariance should be None or a Tensor"
            if torch.sum(torch.abs(model_error_covariance)) < 1e-12:
                model_error_covariance = None
        self.model_error_covariance = model_error_covariance

    def __call__(self, x: State, dt: float, dynamic_inputs: State = None, static_inputs: State = None, noiseless=False):
        """Call linear dynamics time stepping function.

        Args:
            x (State): state object with a single field "x"
            dt (float): time step, ignored as dynamics are the same for each time step.
            dynamic_inputs (State): ignored.
            static_inputs (State): ignored.

        Returns:
            State: updated system state
        """
        state_dim = self.operator.shape[0]
        key = list(x.fields.keys())[0]
        v = x.fields[key]
        s = v.shape
        v = v.reshape(s[0] * s[1], -1, 1)
        new_tensor = (self.operator @ v).reshape(s)
        if self.model_error_covariance is not None and not noiseless:
            new_tensor += (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(state_dim), self.model_error_covariance
                )
                .sample((s[0] * s[1],))
                .reshape(s)
            )
        new_fields = TensorDict({key: new_tensor}, batch_size=x.fields.batch_size)
        output = State(new_fields, time_axis=x.time_axis + dt)
        return output
