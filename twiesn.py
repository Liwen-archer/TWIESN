import numpy as np
from numpy.random import RandomState


def linear_interpolate(sequence, index):
    """Performs linear interpolation on a 1D or 2D sequence."""
    if index < 0: return sequence[0]
    if index >= len(sequence) - 1: return sequence[-1]
    
    idx_floor, idx_ceil = int(np.floor(index)), int(np.ceil(index))
    if idx_floor == idx_ceil: return sequence[idx_floor]
        
    weight = index - idx_floor
    return (1.0 - weight) * sequence[idx_floor] + weight * sequence[idx_ceil]


class TWIESN:
    """
    Core TWIESN class acting as a feature extractor.
    
    This class manages the reservoir dynamics but does not include a readout layer.
    Its purpose is to transform an input time series into a sequence of
    high-dimensional reservoir states.
    """
    def __init__(self, n_inputs, n_reservoir=200, spectral_radius=0.95, sparsity=0.9, noise=0.001, random_state=None):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        
        if random_state is not None:
            self.random_state = RandomState(random_state)
        else:
            self.random_state = RandomState()
        
        self._initialize_weights()
            
    
    def _initialize_weights(self):
        """Initializes the input and reservoir weight matrices."""
        self.W_in = self.random_state.uniform(-1, 1, (self.n_reservoir, self.n_inputs))
        
        W_res_raw = self.random_state.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        
        W_res_raw[self.random_state.rand(*W_res_raw.shape) < self.sparsity] = 0
        
        eigvals, _ = np.linalg.eig(W_res_raw)
        self.W_res = W_res_raw * (self.spectral_radius / np.max(np.abs(eigvals)))
        
    
    def generate_state(self, input_sequence, washout_period=50):
        """
        Processes an input sequence and returns the sequence of reservoir states.
        
        Args:
            input_sequence (np.ndarray): A sequence of shape (n_timesteps, n_inputs).
            washout_period (int): Number of initial timesteps to discard.
            
        Returns:
            np.ndarray: A sequence of reservoir states of shape (n_timesteps - washout, n_reservoir).
        """
        T = len(input_sequence)
        reservoir_state = np.zeros(self.n_reservoir)
        phase = 0.0
        collected_states = []
        
        for t in range(T):
            u_interpolated = linear_interpolate(input_sequence, phase)

            state_pre_activation = self.W_in @ u_interpolated + self.W_res @ reservoir_state
            reservoir_state = np.tanh(state_pre_activation) + self.noise * (self.random_state.rand(self.n_reservoir))
            
            phase += 1.0
            phase = np.clip(phase, 0, T - 1)

            if t >= washout_period:
                collected_states.append(reservoir_state)
        
        return np.array(collected_states)

    
    