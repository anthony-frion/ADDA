import jax.numpy as jnp
import torch
from adda.convert.jax import jax_to_torch_fn


def test_forward_matches_plain_jax_single_output():
    """Test that the forward pass of a JAX function wrapped in a PyTorch autograd function matches the output of the
    original JAX function."""

    jax_square_sum = lambda x, y: jnp.sum(x**2 + y)

    torch_fn = jax_to_torch_fn(jax_square_sum)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.5, 2.5, 2.5])
    out = torch_fn(x, y)
    expected = jax_square_sum(jnp.array(x.numpy()), jnp.array(y.numpy()))
    assert torch.allclose(out, torch.tensor(float(expected)), atol=1e-5)


def test_backward_matches_analytic_gradient():
    """Test that the backward pass of a JAX function wrapped in a PyTorch autograd function produces gradients that
    match the analytic gradient."""

    jax_fn = lambda x: jnp.sum(x**3)

    torch_fn = jax_to_torch_fn(jax_fn)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = torch_fn(x)
    out.backward()
    expected_grad = 3 * x.detach() ** 2
    assert torch.allclose(x.grad, expected_grad, atol=1e-5)


def test_non_contiguous_input_is_handled():
    """Test that a non-contiguous input tensor is correctly handled by the JAX to PyTorch conversion."""

    jax_fn = lambda x: jnp.sum(x)

    torch_fn = jax_to_torch_fn(jax_fn)
    base = torch.arange(12.0, requires_grad=False).reshape(3, 4)
    non_contig = base.t()
    assert not non_contig.is_contiguous()
    non_contig = non_contig.clone().requires_grad_(True)
    out = torch_fn(non_contig)
    assert torch.allclose(out, base.sum())


def test_gradient_flows_correctly_in_larger_torch_graph():
    """Test that gradients flow correctly through a larger PyTorch computation graph that includes a JAX function."""

    jax_fn = lambda x: jnp.sin(x)

    torch_fn = jax_to_torch_fn(jax_fn)
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = x * 2
    z = torch_fn(y)
    loss = z.sum()
    loss.backward()
    expected_grad = 2 * torch.cos(2 * x.detach())
    assert torch.allclose(x.grad, expected_grad, atol=1e-4)
