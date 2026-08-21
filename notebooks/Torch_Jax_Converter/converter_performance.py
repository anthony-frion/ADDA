import statistics
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
from tjconverter import j2t, jax_to_torch_fn, t2j


def sync():
    """Synchronize the device if using CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_block(fn, n_iters=100, warmup=20):
    """Time the execution of a block of code, excluding warmup iterations."""
    for _ in range(warmup):
        fn()
    sync()
    durations = []
    for _ in range(n_iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        durations.append(time.perf_counter() - start)
    return durations


def report(name, durations):
    """Report the mean and standard deviation of a list of durations."""
    mean = statistics.mean(durations)
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0
    print(f"{name:<45s} mean={mean*1e3:.2f} ms std={std*1e3:.2f} ms")
    return mean, std


@jax.jit
def jax_fn(x, W, b):
    """A simple JAX function that performs a linear transformation followed by a non-linear activation and sums the
    result."""
    y = jnp.dot(x, W) + b
    return jnp.sum(jnp.tanh(y))


def main():
    sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    bridged_fwd_times = []
    bridged_fwd_bwd_times = []
    conversion_only_times = []
    pure_jax_fwd_times = []
    pure_jax_fwd_bwd_times = []
    total_overhead = []
    overhead_bridged_ratio = []
    overhead_jax_ratio = []
    bridged_jax_ratio = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device is: ", device)

    torch_fn = jax_to_torch_fn(jax_fn)
    for n in sizes:
        x = torch.randn(n, n, requires_grad=True, device=device)

        W_jax = jax.random.normal(jax.random.PRNGKey(0), (n, n))
        b_jax = jax.random.normal(jax.random.PRNGKey(1), (n,))

        W = j2t(W_jax).requires_grad_(False)
        b = j2t(b_jax).requires_grad_(False)

        def fwd_only():
            with torch.no_grad():
                torch_fn(x, W, b)

        fwd_only_time = time_block(fwd_only)
        fwd_only_mean, _ = report("Bridged forward", fwd_only_time)
        bridged_fwd_times.append((n, fwd_only_mean))

        def fwd_bwd():
            x.grad = None
            out = torch_fn(x, W, b)
            out.backward()

        fwd_bwd_time = time_block(fwd_bwd)
        fwd_bwd_mean, _ = report("Bridged forward+backward", fwd_bwd_time)
        bridged_fwd_bwd_times.append((n, fwd_bwd_mean))

        def conversion_only():
            jx = t2j(x)
            j2t(jx)

        conversion_only_time = time_block(conversion_only)
        conversion_only_mean, _ = report("t2j/j2t conversion overhead only", conversion_only_time)
        conversion_only_times.append((n, conversion_only_mean))

        jx = jax.random.normal(jax.random.PRNGKey(0), (n, n))
        jit_jax_fn = jax.jit(jax_fn)
        jit_jax_fn(jx, W_jax, b_jax).block_until_ready()

        def pure_jax_fwd():
            jit_jax_fn(jx, W_jax, b_jax).block_until_ready()

        pure_jax_fwd_time = time_block(pure_jax_fwd)
        pure_jax_fwd_mean, _ = report("Pure jax.jit forward", pure_jax_fwd_time)
        pure_jax_fwd_times.append((n, pure_jax_fwd_mean))
        grad_fn = jax.jit(jax.grad(jax_fn, argnums=0))

        def pure_jax_fwd_bwd():
            g = grad_fn(jx, W_jax, b_jax)
            g.block_until_ready()

        pure_jax_fwd_bwd_time = time_block(pure_jax_fwd_bwd)
        pure_jax_fwd_bwd_mean, _ = report("Pure jax.jit forward+grad", pure_jax_fwd_bwd_time)
        pure_jax_fwd_bwd_times.append((n, pure_jax_fwd_bwd_mean))
        total_overhead.append((n, fwd_bwd_mean - pure_jax_fwd_bwd_mean))
        overhead_bridged_ratio.append((n, (fwd_bwd_mean - pure_jax_fwd_bwd_mean) / fwd_bwd_mean))
        overhead_jax_ratio.append((n, (fwd_bwd_mean - pure_jax_fwd_bwd_mean) / pure_jax_fwd_bwd_mean))
        bridged_jax_ratio.append((n, fwd_bwd_mean / pure_jax_fwd_bwd_mean))

    figure, ax = plt.subplots()
    for label, times in [
        ("Bridged forward", bridged_fwd_times),
        ("Bridged forward+backward", bridged_fwd_bwd_times),
        ("t2j/j2t conversion only", conversion_only_times),
        ("Pure jax.jit forward", pure_jax_fwd_times),
        ("Pure jax.jit forward+grad", pure_jax_fwd_bwd_times),
    ]:
        sizes, durations = zip(*times)
        mean_durations = [t for t in durations]
        ax.plot(sizes, mean_durations, label=label)
    ax.set_xlabel("Weight matrix size (n x n)")
    ax.set_ylabel("Mean execution time (s)")
    ax.set_title("Performance comparison of bridged vs pure jax.jit")
    ax.legend()
    figure.savefig("performance_comparison.png", dpi=300)
    figure, ax = plt.subplots()
    for label, times in [
        ("Overhead Bridged Ratio", overhead_bridged_ratio),
        ("Overhead Jax Ratio", overhead_jax_ratio),
        ("Bridged Jax Ratio", bridged_jax_ratio),
    ]:
        sizes, ratios = zip(*times)
        ax.plot(sizes, ratios, label=label)
    ax.set_xlabel("Weight matrix size (n x n)")
    ax.set_ylabel("Overhead ratio")
    ax.legend()
    figure.savefig("overhead_ratios.png", dpi=300)


if __name__ == "__main__":
    main()
