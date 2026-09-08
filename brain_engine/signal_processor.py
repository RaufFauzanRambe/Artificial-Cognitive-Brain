"""
signal_processor.py
-------------------
Signal Processing Module for the Artificial Cognitive Brain Architecture.

This module acts as the sensory feature extraction layer, transforming high-dimensional
or noisy continuous environmental signals into structured feature embeddings suitable 
for ingestion by cognitive control units (e.g., NeuralController).

Key Features:
    - Multi-channel temporal signal normalization and filtering.
    - Fast Fourier Transform (FFT) spectral feature extraction.
    - Sliding window framing for continuous stream processing.
    - Vector compression and latent projection for memory integration.

Power Networks and Next-Generation Communication Module for the Artificial Cognitive Brain Architecture.

This module acts as the wireless telemetry, signal routing, and dynamic power management 
interface between internal cognitive controllers and external 5G/6G communication backbones.
It manages dynamic power states, models adaptive beamforming, handles high-bandwidth signal 
transmissions, and integrates directly with pre-processed sensory feature streams.

Key Features:
    - 5G/6G network dynamic beamforming, power efficiency management, and carrier state routing.
    - Integration with internal signal processors and cognitive neural controllers.
    - Automated voice input trigger signal processing and status feedback loops.
    - Real-time continuous state monitoring across 24-hour operational cycles.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, Union, List


class SignalProcessor:
    """
    Processes raw sensory signals (audio, spatial coordinates, biomechanical feeds, 
    or synthetic continuous streams) and converts them into standardized neural representations.
    """

    def __init__(
        self, 
        sample_rate: int = 1000, 
        window_size: int = 128, 
        hop_length: int = 64,
        output_dim: int = 128
    ):
        """
        Initialize the SignalProcessor.

        :param sample_rate: Sampling frequency of the incoming continuous signal (Hz).
        :param window_size: Frame length for sliding window windowing.
        :param hop_length: Step size between consecutive frame windows.
        :param output_dim: Target dimensionality of the extracted feature vector.
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.output_dim = output_dim

        # Pre-calculated Hann window for smoothing time-domain frames
        self.hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)))
        
        # Projection matrix to map spectral/temporal features to target output dimension
        # Derived feature size: Half of window size (FFT magnitude) + time-domain stats (4)
        derived_feature_dim = (window_size // 2 + 1) + 4
        self.projection_matrix = np.random.randn(derived_feature_dim, output_dim) * (1.0 / np.sqrt(derived_feature_dim))

    def normalize_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Applies z-score normalization to incoming raw signals to stabilize neural dynamics.

        :param raw_signal: 1D or 2D array of raw amplitude/sensor values.
        :return: Standardized signal with zero mean and unit variance.
        """
        mean = np.mean(raw_signal, axis=-1, keepdims=True)
        std = np.std(raw_signal, axis=-1, keepdims=True) + 1e-8
        return (raw_signal - mean) / std

    def apply_bandpass_filter(
        self, 
        signal: np.ndarray, 
        low_cutoff: float = 0.5, 
        high_cutoff: float = 100.0
    ) -> np.ndarray:
        """
        Applies a frequency-domain bandpass filter to eliminate low-frequency drift and high-frequency noise.

        :param signal: 1D array representing the continuous signal sequence.
        :param low_cutoff: Lower bound cutoff frequency (Hz).
        :param high_cutoff: Upper bound cutoff frequency (Hz).
        :return: Filtered 1D signal.
        """
        n = len(signal)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        fft_spectrum = np.fft.rfft(signal)

        # Create frequency mask
        mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
        filtered_spectrum = fft_spectrum * mask

        # Inverse FFT to convert back to time domain
        return np.fft.irfft(filtered_spectrum, n=n)

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts temporal and spectral representations from a single signal frame.

        :param frame: 1D numpy array of length `window_size`.
        :return: Concatenated feature vector containing statistical metrics and FFT magnitudes.
        """
        if len(frame) != self.window_size:
            # Pad or truncate frame if needed
            frame = np.pad(frame, (0, max(0, self.window_size - len(frame))), mode='constant')[:self.window_size]

        # Apply smoothing window
        windowed_frame = frame * self.hann_window

        # Time-domain statistical features
        mean_val = np.mean(windowed_frame)
        std_val = np.std(windowed_frame)
        energy_val = np.sum(windowed_frame ** 2) / self.window_size
        rms_val = np.sqrt(energy_val)
        time_stats = np.array([mean_val, std_val, energy_val, rms_val])

        # Spectral domain features (FFT Magnitude Spectrum)
        fft_magnitudes = np.abs(np.fft.rfft(windowed_frame))

        # Concatenate time-domain stats and spectral features
        combined_features = np.concatenate([time_stats, fft_magnitudes])
        return combined_features

    def process_stream(self, raw_signal_stream: np.ndarray) -> np.ndarray:
        """
        Processes an entire continuous signal sequence into a sequence of encoded feature vectors.

        :param raw_signal_stream: 1D array of temporal continuous sensor inputs.
        :return: Array of shape (num_frames, output_dim) containing latent feature embeddings.
        """
        # Step 1: Preprocessing & Normalization
        cleaned_signal = self.normalize_signal(raw_signal_stream)
        filtered_signal = self.apply_bandpass_filter(cleaned_signal)

        # Step 2: Framing using sliding window
        signal_length = len(filtered_signal)
        if signal_length < self.window_size:
            frames = [filtered_signal]
        else:
            num_frames = 1 + (signal_length - self.window_size) // self.hop_length
            frames = [
                filtered_signal[i * self.hop_length : i * self.hop_length + self.window_size]
                for i in range(num_frames)
            ]

        # Step 3: Feature Extraction per frame
        feature_list = []
        for frame in frames:
            frame_feats = self.extract_frame_features(frame)
            feature_list.append(frame_feats)

        feature_matrix = np.array(feature_list)

        # Step 4: Projection to target dimensionality (Neural Embedding Space)
        latent_embeddings = np.dot(feature_matrix, self.projection_matrix)
        latent_embeddings = np.tanh(latent_embeddings)  # Non-linear scaling

        return latent_embeddings

    def process_single_step(self, recent_buffer: np.ndarray) -> np.ndarray:
        """
        Real-time pipeline step: Accepts recent buffer data and outputs a single embedded feature vector 
        matching `output_dim` for direct feed into `NeuralController`.

        :param recent_buffer: 1D numpy array containing recent signal history.
        :return: 2D array of shape (1, output_dim).
        """
        normalized_buf = self.normalize_signal(recent_buffer)
        frame_feats = self.extract_frame_features(normalized_buf)
        
        # Project frame to latent space
        latent_vec = np.dot(frame_feats, self.projection_matrix)
        latent_vec = np.tanh(latent_vec)
        
        return np.expand_dims(latent_vec, axis=0)

class PowerNetworks:
    """
    Manages high-frequency 5G/6G wireless communication channels and dynamic power networks
    for cognitive neural systems.
    """

    def __init__(
        self, 
        base_power_watt: float = 45.0, 
        carrier_frequency_ghz: float = 28.0, 
        total_channels: int = 64
    ):
        """
        Initialize the PowerNetworks infrastructure.

        :param base_power_watt: Default power draw of the transmission node in Watts.
        :param carrier_frequency_ghz: Operational frequency (e.g., 28GHz mmWave for 5G, 100GHz+ for 6G).
        :param total_channels: Total spatial MIMO beamforming channels available.
        """
        self.base_power_watt = base_power_watt
        self.carrier_frequency_ghz = carrier_frequency_ghz
        self.total_channels = total_channels
        
        # System state tracking
        self.system_on: bool = False
        self.active_connections: int = 0
        self.channel_weights = np.random.randn(total_channels, total_channels) * 0.01
        self.transmission_log: List[Dict[str, Any]] = []

    def set_system_power_state(self, power_switch: bool) -> str:
        """
        Toggles the physical transceiver power supply ON or OFF.

        :param power_switch: True for turning the network module ON, False for OFF.
        :return: String status message.
        """
        self.system_on = bool(power_switch)
        state_str = "ACTIVE (ONLINE)" if self.system_on else "STANDBY (OFFLINE)"
        return f"Power Network System State changed to: {state_str}"

    def network_5g_6g(
        self, 
        on_off_system: bool, 
        actions_communications: np.ndarray, 
        vendoval_matrix: np.ndarray,
        x_start: float = 0.0,
        y_24_clock: float = 12.0,
        voice_input_signal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Main 5G/6G interface controller method. Synthesizes spatial state signals, 
        executes action-driven communication protocols, and handles voice trigger inputs.

        :param on_off_system: Boolean flag indicating if the transmission sub-system is active.
        :param actions_communications: Vector of action logits derived from executive controllers.
        :param vendoval_matrix: Array defining dynamic spatial path-loss and channel fading factors.
        :param x_start: Initial temporal offset or spatial position coordinate.
        :param y_24_clock: Time of day in hours (0.0 to 24.0) for temporal load scaling.
        :param voice_input_signal: Optional raw or embedded array of acoustic/voice inputs.
        :return: Dictionary containing complete telemetry, signal states, and completion status.
        """
        # Step 1: Manage System Power Transition
        self.set_system_power_state(on_off_system)

        if not self.system_on:
            return {
                "status": "TERMINATED",
                "message": "System is powered down. No communication actions executed.",
                "active_power_watt": 0.0
            }

        # Step 2: Formulate State Signal Coordinates
        state_signal: Tuple[float, float] = (float(x_start), float(y_24_clock))

        # Step 3: Compute Adaptive Beamforming Transmission Matrix
        # Combine action logits with environmental channel fading factors (vendoval matrix)
        if actions_communications.ndim == 1:
            actions_communications = np.expand_dims(actions_communications, axis=0)

        raw_transmission_logits = np.dot(actions_communications, vendoval_matrix)
        
        # Apply 6G High-Frequency non-linear modulation scaling
        modulated_signal = np.tanh(raw_transmission_logits)

        # Step 4: Temporal Load Coefficient (24-Hour Cycle Scaling)
        # Power demand fluctuates periodically throughout the day
        load_factor = 1.0 + 0.3 * np.sin(2.0 * np.pi * (y_24_clock / 24.0))
        computed_power_consumption = self.base_power_watt * load_factor * (1.0 + np.mean(np.abs(modulated_signal)))

        # Step 5: Process Voice Input Signal Stream (if present)
        voice_processed: bool = False
        voice_energy: float = 0.0

        if voice_input_signal is not None:
            voice_energy = float(np.sum(voice_input_signal ** 2) / max(1, voice_input_signal.size))
            voice_processed = True

        # Step 6: Construct Final Signal Structure & Telemetry
        finished_signal: Dict[str, Any] = {
            "voice_input_detected": voice_processed,
            "voice_energy_level": voice_energy,
            "transmission_matrix_shape": modulated_signal.shape,
            "signal_integrity_index": float(np.mean(modulated_signal)),
            "completion_timestamp": time.time()
        }

        execution_summary = {
            "system_power_status": "ONLINE",
            "state_signal_coordinate": state_signal,
            "operational_power_watt": computed_power_consumption,
            "carrier_frequency": f"{self.carrier_frequency_ghz} GHz",
            "modulated_output": modulated_signal,
            "finished_signal": finished_signal
        }

        self.transmission_log.append(execution_summary)
        return execution_summary


class SignalProcessor:
    """
    Signal Processing Engine for feature extraction and temporal windowing of continuous inputs.
    """

    def __init__(
        self, 
        sample_rate: int = 1000, 
        window_size: int = 128, 
        hop_length: int = 64, 
        output_dim: int = 128
    ):
        """
        Initialize the SignalProcessor.

        :param sample_rate: Sampling frequency of the incoming signal in Hz.
        :param window_size: Size of the frame window.
        :param hop_length: Number of samples between successive frames.
        :param output_dim: Target dimension for latent projections.
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.output_dim = output_dim
        self.hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)))
        
        derived_feature_dim = (window_size // 2 + 1) + 4
        self.projection_matrix = np.random.randn(derived_feature_dim, output_dim) * (1.0 / np.sqrt(derived_feature_dim))

    def normalize_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Applies z-score normalization to incoming raw signals."""
        mean = np.mean(raw_signal, axis=-1, keepdims=True)
        std = np.std(raw_signal, axis=-1, keepdims=True) + 1e-8
        return (raw_signal - mean) / std

    def apply_bandpass_filter(self, signal: np.ndarray, low_cutoff: float = 0.5, high_cutoff: float = 100.0) -> np.ndarray:
        """Applies frequency-domain filtering across cutoff ranges."""
        n = len(signal)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        fft_spectrum = np.fft.rfft(signal)
        mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
        return np.fft.irfft(fft_spectrum * mask, n=n)

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extracts statistical and FFT features from a single frame."""
        if len(frame) != self.window_size:
            frame = np.pad(frame, (0, max(0, self.window_size - len(frame))), mode='constant')[:self.window_size]
        
        windowed_frame = frame * self.hann_window
        mean_val = np.mean(windowed_frame)
        std_val = np.std(windowed_frame)
        energy_val = np.sum(windowed_frame ** 2) / self.window_size
        rms_val = np.sqrt(energy_val)
        time_stats = np.array([mean_val, std_val, energy_val, rms_val])
        
        fft_magnitudes = np.abs(np.fft.rfft(windowed_frame))
        return np.concatenate([time_stats, fft_magnitudes])

    def process_stream(self, raw_signal_stream: np.ndarray) -> np.ndarray:
        """Processes continuous signal streams into encoded feature embedding matrices."""
        cleaned_signal = self.normalize_signal(raw_signal_stream)
        filtered_signal = self.apply_bandpass_filter(cleaned_signal)
        
        signal_length = len(filtered_signal)
        if signal_length < self.window_size:
            frames = [filtered_signal]
        else:
            num_frames = 1 + (signal_length - self.window_size) // self.hop_length
            frames = [
                filtered_signal[i * self.hop_length : i * self.hop_length + self.window_size]
                for i in range(num_frames)
            ]
            
        feature_matrix = np.array([self.extract_frame_features(f) for f in frames])
        latent_embeddings = np.tanh(np.dot(feature_matrix, self.projection_matrix))
        return latent_embeddings

    def process_single_step(self, recent_buffer: np.ndarray) -> np.ndarray:
        """Processes real-time window frames into single embedding vectors."""
        normalized_buf = self.normalize_signal(recent_buffer)
        frame_feats = self.extract_frame_features(normalized_buf)
        latent_vec = np.tanh(np.dot(frame_feats, self.projection_matrix))
        return np.expand_dims(latent_vec, axis=0)


if __name__ == "__main__":
    # Demonstration & Verification
    print("--- Initializing Signal Processor ---")
    processor = SignalProcessor(sample_rate=1000, window_size=128, hop_length=64, output_dim=128)

    # Simulate a 2-second noisy sinusoidal signal
    time_steps = np.linspace(0, 2.0, 2000)
    signal_50hz = np.sin(2 * np.pi * 50 * time_steps)
    signal_120hz = 0.5 * np.sin(2 * np.pi * 120 * time_steps)
    random_noise = 0.2 * np.random.randn(len(time_steps))
    synthetic_signal = signal_50hz + signal_120hz + random_noise

    # Batch Process Sequence
    encoded_stream = processor.process_stream(synthetic_signal)
    print(f"Input Signal Length: {len(synthetic_signal)} samples")
    print(f"Processed Stream Output Shape: {encoded_stream.shape}")

    # Single-step Real-time Input Simulation
    buffer_window = synthetic_signal[-128:]
    realtime_embedding = processor.process_single_step(buffer_window)
    print(f"Real-time Single-step Embedding Shape: {realtime_embedding.shape}")

    print("\n--- Initializing Power Networks (5G/6G Subsystem) ---")
    power_network_controller = PowerNetworks(base_power_watt=50.0, carrier_frequency_ghz=28.0)

    # Generate synthetic network actions and dynamic vendoval spatial matrix
    action_vector = np.random.randn(1, 128)
    np_vendoval = np.random.randn(128, 64) * 0.05
    dummy_voice_input = synthetic_signal[:256]

    # Execute 5G/6G Network Control Step
    network_output = power_network_controller.network_5g_6g(
        on_off_system=True,
        actions_communications=action_vector,
        vendoval_matrix=np_vendoval,
        x_start=0.0,
        y_24_clock=14.5,
        voice_input_signal=dummy_voice_input
    )

    print(f"System Power Status: {network_output['system_power_status']}")
    print(f"Operational Power Draw: {network_output['operational_power_watt']:.2f} W")
    print(f"State Signal Coordinates: {network_output['state_signal_coordinate']}")
    print(f"Modulated Signal Shape: {network_output['modulated_output'].shape}")
    print(f"Voice Input Processed: {network_output['finished_signal']['voice_input_detected']}")
    print(f"Signal Integrity Metric: {network_output['finished_signal']['signal_integrity_index']:.4f}")
