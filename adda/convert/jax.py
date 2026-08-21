import jax
import torch


def j2t(x):
    """Convert a JAX array to a PyTorch tensor using DLPack."""
    # Ensure the input is a JAX array
    assert hasattr(x, "__dlpack__"), "Input must be a JAX array."
    return torch.from_dlpack(x)


def t2j(x: torch.Tensor):
    """Convert a PyTorch tensor to a JAX array using DLPack."""
    x = x.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    # Ensure the input is a PyTorch tensor
    assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor."
    return jax.dlpack.from_dlpack(x)


def _forward_impl(jax_fn, *args):
    """Forward pass implementation for the JAX function.

    So we can jit compile it and use it in the backward pass.
    """
    return jax_fn(*args)


def _backward_impl(jax_fn, *args_and_cot):
    """Backward pass implementation for the JAX function.

    Again, we jit compile it so we can use it in the backward pass.
    """
    *args, cot = args_and_cot
    _, vjp_fn = jax.vjp(jax_fn, *args)
    grads = vjp_fn(cot)
    return grads


_forward = jax.jit(_forward_impl, static_argnums=0)
_backward = jax.jit(_backward_impl, static_argnums=0)


class _JaxFunction(torch.autograd.Function):
    """A custom PyTorch autograd function that wraps a JAX function.

    For more details, see https://docs.pytorch.org/docs/2.13/notes/extending.html
        and https://docs.jax.dev/en/latest/jax-primitives.html
    This allows the JAX function to be used in a PyTorch computation graph,
    enabling automatic differentiation and gradient computation.
    Args:
        jax_fn: A JAX function to be wrapped. This function should take JAX
          arrays as input and return a JAX array as output.
        *torch_args: A variable number of PyTorch tensors that will be converted
          to JAX arrays and passed to the JAX function.
    Returns:
        A PyTorch tensor that is the result of applying the JAX function to the
        converted input tensors. The output tensor will be part of the PyTorch
        computation graph, allowing for gradient computation through the JAX function.
    """

    @staticmethod
    def forward(ctx, jax_fn, *torch_args):
        jax_args = tuple(t2j(x) for x in torch_args)
        out = _forward(jax_fn, *jax_args)
        ctx.jax_fn = jax_fn
        ctx.jax_args = jax_args
        ctx.save_for_backward(*torch_args)
        return j2t(out)

    @staticmethod
    def backward(ctx, *grad_outputs):
        jax_args = ctx.jax_args
        cot = t2j(grad_outputs[0])
        jax_grads = _backward(ctx.jax_fn, *jax_args, cot)
        torch_grads = tuple(j2t(g) for g in jax_grads)
        return (None,) + torch_grads


def jax_to_torch_fn(jax_fn):
    """Wrap a JAX function for use in PyTorch."""

    def wrapped(*torch_args):
        return _JaxFunction.apply(jax_fn, *torch_args)

    return wrapped


class _JaxFunctionEagerVjp(torch.autograd.Function):
    """A custom PyTorch autograd function that wraps a JAX function and computes gradients using eager VJP (vector-
    Jacobian product).

    We jit the Jax function separately for this one. For more details, see
    https://docs.pytorch.org/docs/2.13/notes/extending.html
    and https://docs.jax.dev/en/latest/jax-primitives.html
    """

    @staticmethod
    def forward(ctx, jax_fn, *torch_args):
        jax_args = tuple(t2j(x) for x in torch_args)
        out, vjp_fn = jax.vjp(jax_fn, *jax_args)
        ctx.vjp_fn = vjp_fn
        return j2t(out)

    @staticmethod
    def backward(ctx, *grad_outputs):
        cot = t2j(grad_outputs[0])
        jax_grads = ctx.vjp_fn(cot)
        torch_grads = tuple(j2t(g) for g in jax_grads)
        return (None,) + torch_grads


def jax_to_torch_fn_eager_vjp(jax_fn):
    """Wrap a JAX function for use in PyTorch with eager VJP (vector-Jacobian product) computation."""

    def wrapped(*torch_args):
        return _JaxFunctionEagerVjp.apply(jax_fn, *torch_args)

    return wrapped
