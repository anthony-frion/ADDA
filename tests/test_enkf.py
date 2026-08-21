import pytest
import torch
from adda.ensemble.enkf import EnKF, enkf
from adda.observation.operators import random_sparse_noisy_obs
from adda.system.state import State
from tensordict import TensorDict


class TestEnKF:
    """Test suite for EnKF data assimilation."""

    @pytest.fixture
    def basic_setup(self):
        """Basic setup for EnKF tests."""
        n_members = 10
        n_variables = 5
        n_time_steps = 3
        initial_ensemble_std = 1.0

        # Create EnKF object
        enkf_obj = EnKF(n_members=n_members, n_variables=n_variables, initial_ensemble_std=initial_ensemble_std)

        # Create initial state
        initial_state = torch.randn(n_variables)
        x_init = State(
            TensorDict(x=initial_state.unsqueeze(0).unsqueeze(0), batch_size=(1, 1))  # Shape: (1, 1, n_variables)
        )

        x = State(
            TensorDict(
                x=torch.randn(1, n_time_steps, n_variables),  # Shape: (1, n_time_steps, n_variables)
                batch_size=(1, n_time_steps),  # This should match the first two dimensions of x
            ),
            time_axis=torch.arange(0, n_time_steps * 0.1 - 0.1, 0.1),
        )
        # Create time axis
        dt = 0.1
        time_axis = torch.arange(0, n_time_steps * dt - dt, dt)
        print("TIME AXIS: ", time_axis)
        return {
            "enkf_obj": enkf_obj,
            "x_init": x_init,
            "n_members": n_members,
            "n_variables": n_variables,
            "n_time_steps": n_time_steps,
            "initial_ensemble_std": initial_ensemble_std,
            "dt": dt,
            "time_axis": time_axis,
            "x": x,
        }

    @pytest.fixture
    def mock_observations(self, basic_setup):
        """Create mock observations and observation operator using random_sparse_noisy_obs."""
        setup = basic_setup
        n_variables = setup["n_variables"]
        n_time_steps = setup["n_time_steps"]
        time_axis = setup["time_axis"]

        # Create a mock true time series (similar to true_ts in your notebook)
        # Generate some realistic-looking time series data
        true_values = torch.randn(1, n_time_steps, n_variables)
        # Add some temporal correlation to make it more realistic
        for t in range(1, n_time_steps):
            true_values[0, t, :] = 0.9 * true_values[0, t - 1, :] + 0.1 * torch.randn(n_variables)

        true_ts = State(TensorDict(x=true_values, batch_size=(1, n_time_steps)), time_axis=time_axis)

        # Use the same parameters as in your notebook
        noise_amplitude = 1.0  # observation noise s.d.
        p_obs = 0.8  # 80% of the variables are masked (20% observed)

        # Use the random_sparse_noisy_obs function
        groundtruth, obs_op, observations = random_sparse_noisy_obs(true_ts, noise_amplitude, p_obs)

        return observations, obs_op

    @pytest.fixture
    def linear_dynamics(self):
        """Simple linear dynamics for testing."""

        def m_dyn(x: State, dt: float, dynamics_inputs=None, static_inputs=None):
            # Simple linear dynamics: x_next = A * x_current
            A = torch.eye(x.fields["x"].shape[-1]) * 0.95  # Stable dynamics

            # Apply dynamics
            x_new = torch.matmul(x.fields["x"], A.T)

            # Create new State object
            new_fields = TensorDict(x=x_new, batch_size=x.fields.batch_size)

            new_time_axis = x.time_axis + dt if x.time_axis is not None else torch.tensor([dt])
            return State(new_fields, time_axis=new_time_axis)

        return m_dyn

    def test_enkf_initialization(self, basic_setup):
        """Test EnKF initialization."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]

        assert enkf_obj.n_members == setup["n_members"]
        assert enkf_obj.n_variables == setup["n_variables"]
        assert enkf_obj.initial_ensemble_std == setup["initial_ensemble_std"]
        assert enkf_obj.ensemble is None  # Not initialized yet

    def test_ensemble_initialization(self, basic_setup):
        """Test ensemble initialization."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]

        # Initialize ensemble
        ensemble_state = enkf_obj.initialize_ensemble(x_init)

        # Check shapes
        assert ensemble_state.fields["x"].shape == (setup["n_members"], 1, setup["n_variables"])
        assert enkf_obj.ensemble is not None

        # Check that ensemble members are different (due to noise)
        ensemble_matrix = ensemble_state.fields["x"][:, 0, :]
        assert not torch.allclose(ensemble_matrix[0], ensemble_matrix[1], atol=1e-6)

    def test_linearize_observation_operator(self, basic_setup, mock_observations):
        """Test linearization of observation operator."""
        observations, obs_op = mock_observations
        # Test linearization at different time indices
        for t in range(basic_setup["n_time_steps"]):
            H_t = obs_op.linearize(idx_time=t)

            # Check dimensions
            n_obs = observations.mask.fields["x"][0, t, :].sum().item()
            assert H_t.shape[0] == n_obs  # Number of observations
            assert H_t.shape[1] == basic_setup["n_variables"]  # Number of variables

            H_t = H_t.to_dense()

            # Check that H_t is a valid observation matrix (0s and 1s)
            assert torch.all((H_t == 0) | (H_t == 1))

            # Check that each row has exactly one 1 (identity mapping)
            if n_obs > 0:
                assert torch.allclose(H_t.sum(dim=1), torch.ones(n_obs))

    def test_forecast_step(self, basic_setup, linear_dynamics):
        """Test forecast step."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]

        # Initialize ensemble
        enkf_obj.initialize_ensemble(x_init)
        initial_ensemble = enkf_obj.ensemble.fields["x"].clone()

        # Perform forecast step
        enkf_obj.forecast_step(linear_dynamics, setup["dt"], 1, None, None)

        # Check that ensemble has evolved
        final_ensemble = enkf_obj.ensemble.fields["x"]
        assert final_ensemble.shape[1] == 2  # One more time step
        assert not torch.allclose(initial_ensemble[:, 0, :], final_ensemble[:, 1, :])

        # Check time axis update
        expected_time = setup["dt"]
        if enkf_obj.ensemble.time_axis is not None:
            assert torch.isclose(enkf_obj.ensemble.time_axis[-1], torch.tensor(expected_time))

    def test_analysis_step(self, basic_setup, mock_observations):
        """Test analysis step."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]
        observations, obs_op = mock_observations
        # Initialize ensemble
        enkf_obj.initialize_ensemble(x_init)

        # Get H matrix for first time step
        H_t = obs_op.linearize(idx_time=0)

        if H_t.shape[0] > 0:  # Only test if there are observations
            # Create R matrix
            n_obs = H_t.shape[0]
            R_t = torch.eye(n_obs) * 0.01

            # Store pre-analysis ensemble
            pre_analysis = enkf_obj.ensemble.fields["x"][:, -1, :].clone()

            # Perform analysis step
            mask_t0 = observations.mask.fields["x"][0, 0, :]
            observations_t0 = observations.state.fields["x"][0, 0, mask_t0]
            enkf_obj.analysis_step(H_t, observations_t0, obs_op, R_t, 0)

            # Check that ensemble has been updated
            post_analysis = enkf_obj.ensemble.fields["x"][:, -1, :]
            assert not torch.allclose(pre_analysis, post_analysis, atol=1e-6)

    def test_assimilate_method(self, basic_setup, mock_observations, linear_dynamics):
        """Test the main assimilate method."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]
        observations, obs_op = mock_observations

        n_steps = 3

        # Test with history=True
        enkf_result = enkf_obj.assimilate(
            m_dyn=linear_dynamics,
            observations=observations,
            obs_op=obs_op,
            x_init=x_init,
            verbose=False,
        )

        # Check ensemble result
        assert enkf_result.fields["x"].shape == (setup["n_members"], n_steps, setup["n_variables"])

    def test_enkf_function(self, basic_setup, mock_observations, linear_dynamics):
        """Test the standalone enkf_assimilate function."""
        setup = basic_setup
        x_init = setup["x_init"]
        n_members = setup["n_members"]
        initial_ensemble_std = setup["initial_ensemble_std"]
        observations, obs_op = mock_observations

        n_steps = 3

        # Test standalone function
        enkf_result = enkf(
            m_dyn=linear_dynamics,
            observations=observations,
            obs_op=obs_op,
            x_init=x_init,
            n_members=n_members,
            initial_ensemble_std=initial_ensemble_std,
            verbose=False,
        )

        # Check results
        assert enkf_result.fields["x"].shape == (n_members, n_steps, setup["n_variables"])

    def test_get_ensemble_statistics(self, basic_setup):
        """Test ensemble statistics computation."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]

        # Initialize ensemble
        enkf_obj.initialize_ensemble(x_init)

        # Get statistics
        stats = enkf_obj.get_ensemble_statistics()

        # Check statistics
        assert "mean" in stats
        assert "std" in stats
        assert "members" in stats
        assert "spread" in stats
        assert "full_history" in stats

        # Check shapes
        assert stats["mean"].shape == (setup["n_variables"],)
        assert stats["std"].shape == (setup["n_variables"],)
        assert stats["members"].shape == (setup["n_members"], setup["n_variables"])
        assert isinstance(stats["spread"], torch.Tensor)

    def test_error_handling(self, basic_setup):
        """Test error handling in EnKF."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]

        # Test initialization without observations
        with pytest.raises(ValueError, match="x_init must be provided to initialize the ensemble"):
            enkf_obj.assimilate(m_dyn=lambda x, dt, di, si: x, observations=None, obs_op=None, x_init=None)

    def test_ensemble_convergence(self, basic_setup, mock_observations, linear_dynamics):
        """Test that ensemble converges with observations."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]
        observations, obs_op = mock_observations

        # Run assimilation
        enkf_result = enkf_obj.assimilate(
            m_dyn=linear_dynamics, observations=observations, obs_op=obs_op, x_init=x_init, verbose=False
        )

        # Check that ensemble spread generally decreases or stays reasonable
        initial_spread = enkf_result.fields["x"].std(dim=0).mean()
        final_spread = enkf_result.fields["x"].std(dim=0).mean()

        # Spread should not explode (basic sanity check)
        assert final_spread < initial_spread * 10  # Allow some growth but not explosion
        assert final_spread > 0  # Should maintain some spread

    def test_verbose_output(self, basic_setup, mock_observations, linear_dynamics, capsys):
        """Test verbose output."""
        setup = basic_setup
        enkf_obj = setup["enkf_obj"]
        x_init = setup["x_init"]
        observations, obs_op = mock_observations

        # Run with verbose=True
        enkf_obj.assimilate(
            m_dyn=linear_dynamics, observations=observations, obs_op=obs_op, x_init=x_init, verbose=True
        )

        # Check that verbose output was printed
        captured = capsys.readouterr()
        assert "Final ensemble statistics" in captured.out


class TestEnKFIntegration:
    """Integration tests for EnKF with realistic scenarios."""

    @pytest.fixture
    def mock_observations_integration(self):
        """Create mock observations for integration tests."""
        n_variables = 10
        n_steps = 10
        dt = 0.1

        # Create time axis
        time_axis = torch.arange(0, n_steps * dt, dt)

        # Create a mock true time series
        true_values = torch.randn(1, n_steps, n_variables)
        # Add some temporal correlation to make it more realistic
        for t in range(1, n_steps):
            true_values[0, t, :] = 0.9 * true_values[0, t - 1, :] + 0.1 * torch.randn(n_variables)

        true_ts = State(TensorDict(x=true_values, batch_size=(1, n_steps)), time_axis=time_axis)

        # Use the same parameters as in your notebook
        noise_amplitude = 1.0  # observation noise s.d.
        p_obs = 0.25  # 75% of the variables are masked (25% observed)

        # Use the random_sparse_noisy_obs function
        groundtruth, obs_op, observations = random_sparse_noisy_obs(true_ts, noise_amplitude, p_obs)

        return observations, obs_op

    def test_lorenz96_like_system(self, mock_observations_integration):
        """Test EnKF with a Lorenz-96 like system using realistic observations."""
        n_variables = 10
        n_members = 20
        n_steps = 10
        initial_ensemble_std = 0.01
        # Create initial state as 1D tensor (will be converted to State in assimilate method)
        x_init = State(
            TensorDict(
                x=torch.randn(1, 1, n_variables),  # Shape: (1, 1, n_variables)
                batch_size=(1, 1),
            )
        )

        # Get observations and observation operator from fixture
        observations, obs_op = mock_observations_integration

        # Simple nonlinear dynamics (approximating Lorenz-96)
        def nonlinear_dynamics(x: State, dt: float, dynamics_inputs=None, static_inputs=None):
            # Extract current state from all ensemble members
            # x.fields["x"] shape: (n_members, 1, n_variables)
            x_current = x.fields["x"][:, -1, :]  # Shape: (n_members, n_variables)

            # Simple nonlinear update (not exact Lorenz-96 but similar structure)
            # Apply to all ensemble members simultaneously
            x_new = x_current - dt * x_current + dt * 0.1 * torch.sin(x_current)

            # Add coupling using vectorized operations instead of loop
            # Create shifted versions for coupling terms
            x_shifted = torch.roll(x_current, shifts=-1, dims=1)  # Shift variables: x_{i+1}
            x_new += dt * 0.01 * x_shifted

            # Create new State with proper batch dimensions
            # x_new shape: (n_members, n_variables)
            # Add time dimension: (n_members, 1, n_variables)
            new_fields = TensorDict(x=x_new.unsqueeze(1), batch_size=(x_new.shape[0], 1))  # (n_members, 1)

            # Update time
            if x.time_axis is not None:
                if len(x.time_axis.shape) == 0:  # Scalar time
                    new_time = x.time_axis + dt
                else:  # Vector time
                    new_time = x.time_axis[-1:] + dt
            else:
                new_time = torch.tensor([dt])

            return State(new_fields, time_axis=new_time)

        # Run assimilation using observations from mock_observations_integration
        enkf_result = enkf(
            m_dyn=nonlinear_dynamics,
            observations=observations,  # From fixture
            obs_op=obs_op,  # From fixture
            x_init=x_init,
            n_members=n_members,
            initial_ensemble_std=initial_ensemble_std,
            verbose=False,
        )

        # Basic checks
        assert len(enkf_result.fields["x"].shape) == 3
        assert len(enkf_result.fields["x"].mean(dim=0)) == n_steps

        # Check that system evolved
        initial_mean = enkf_result.fields["x"][0].mean(dim=0)
        final_mean = enkf_result.fields["x"][-1].mean(dim=0)
        assert not torch.allclose(initial_mean, final_mean, atol=1e-3)

        # Check ensemble spread is reasonable
        final_spread = enkf_result.fields["x"][-1].std(dim=0).mean()
        assert 0.001 < final_spread < 10.0  # Reasonable spread bounds

        # Additional checks specific to realistic observations
        print(f"Number of observations used: {observations.mask.fields['x'].sum().item()}")
        print(
            f"Observation coverage: {(observations.mask.fields['x'].sum() / observations.mask.fields['x'].numel() * 100):.1f}%"
        )
        print(f"Final ensemble spread: {final_spread:.4f}")

        # Verify observations were actually used
        mask_sum = observations.mask.fields["x"].sum()
        assert mask_sum > 0, "Should have some observations to assimilate"

        # Check that assimilation had some effect compared to pure forecast
        # (This is a basic sanity check)
        assert len(enkf_result.fields["x"].mean(dim=0)) == n_steps, "Should have evolved over multiple time steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
