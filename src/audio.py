import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

class SpectrogramToAudio:
    def __init__(self, sample_rate=44100, block_size=512, num_bins=64, history_duration=10.0):
        """
        Initialize the spectrogram-to-audio converter.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            block_size: Number of samples per audio block
            num_bins: Number of frequency bins in the spectrogram
            history_duration: Duration of spectrogram history to display (seconds)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.num_bins = num_bins
        self.history_duration = history_duration
        
        # Generate frequency bins (logarithmic spacing is more perceptually relevant)
        self.frequencies = np.logspace(np.log10(100), np.log10(8000), num_bins)
        
        # Pre-compute phase accumulators for each frequency
        self.phase_left = np.zeros(num_bins)
        self.phase_right = np.zeros(num_bins)
        
        # Time array for one block
        self.t = np.arange(block_size) / sample_rate
        
        # Calculate update rate and max history length
        self.update_interval = block_size / sample_rate
        self.max_history_frames = int(history_duration / self.update_interval)
        
        # History buffers for visualization
        self.history_left = deque(maxlen=self.max_history_frames)
        self.history_right = deque(maxlen=self.max_history_frames)
        self.history_times = deque(maxlen=self.max_history_frames)
        self.start_time = time.time()
        
        print(f"Initialized with {num_bins} frequency bins")
        print(f"Frequency range: {self.frequencies[0]:.1f} Hz to {self.frequencies[-1]:.1f} Hz")
        print(f"History duration: {history_duration} seconds ({self.max_history_frames} frames)")
    
    def generate_audio_block(self, magnitudes_left, magnitudes_right):
        """
        Generate one block of audio from spectrogram coefficients.
        
        Args:
            magnitudes_left: Array of magnitudes for left channel (length = num_bins)
            magnitudes_right: Array of magnitudes for right channel (length = num_bins)
            
        Returns:
            2D array of shape (block_size, 2) containing stereo audio
        """
        # Store in history for visualization
        current_time = time.time() - self.start_time
        self.history_left.append(magnitudes_left.copy())
        self.history_right.append(magnitudes_right.copy())
        self.history_times.append(current_time)
        
        # Initialize output
        audio_left = np.zeros(self.block_size)
        audio_right = np.zeros(self.block_size)
        
        # Generate each frequency component and sum them
        for i, freq in enumerate(self.frequencies):
            # Calculate phase increment for this block
            phase_increment = 2 * np.pi * freq * self.t
            
            # Generate sinusoids with continuous phase
            audio_left += magnitudes_left[i] * np.sin(phase_increment + self.phase_left[i])
            audio_right += magnitudes_right[i] * np.sin(phase_increment + self.phase_right[i])
            
            # Update phase accumulators for continuity between blocks
            self.phase_left[i] += 2 * np.pi * freq * self.block_size / self.sample_rate
            self.phase_right[i] += 2 * np.pi * freq * self.block_size / self.sample_rate
            
            # Keep phase in [0, 2π] range
            self.phase_left[i] = self.phase_left[i] % (2 * np.pi)
            self.phase_right[i] = self.phase_right[i] % (2 * np.pi)
        
        # Normalize to prevent clipping
        max_val = max(np.abs(audio_left).max(), np.abs(audio_right).max())
        if max_val > 0:
            audio_left /= max_val
            audio_right /= max_val
        
        # Combine into stereo
        stereo_audio = np.column_stack((audio_left, audio_right))
        
        return stereo_audio.astype(np.float32)
    
    def start_stream(self, callback_function):
        """
        Start the audio output stream.
        
        Args:
            callback_function: Function that returns (magnitudes_left, magnitudes_right)
                              when called. Should return arrays of length num_bins.
        """
        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            
            # Get current spectrogram coefficients
            mags_left, mags_right = callback_function()
            
            # Generate audio block
            audio_block = self.generate_audio_block(mags_left, mags_right)
            
            # Write to output
            outdata[:] = audio_block[:frames]
        
        # Create and start stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=2,
            callback=audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        print("Audio stream started!")
    
    def stop_stream(self):
        """Stop the audio output stream."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            print("Audio stream stopped!")
    
    def visualize_live(self):
        """
        Create a live visualization of the spectrogram for both channels.
        This should be called in the main thread after starting the audio stream.
        """
        # Set up the figure and subplots
        fig, (ax_left, ax_right) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Live Spectrogram Visualization', fontsize=16)
        
        # Initialize empty images
        im_left = ax_left.imshow(
            np.zeros((self.num_bins, 100)),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        im_right = ax_right.imshow(
            np.zeros((self.num_bins, 100)),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Set up axes
        ax_left.set_ylabel('Frequency (Hz)')
        ax_left.set_title('Left Channel')
        ax_right.set_ylabel('Frequency (Hz)')
        ax_right.set_xlabel('Time (seconds)')
        ax_right.set_title('Right Channel')
        
        # Add colorbars
        plt.colorbar(im_left, ax=ax_left, label='Magnitude')
        plt.colorbar(im_right, ax=ax_right, label='Magnitude')
        
        def update_plot(frame):
            if len(self.history_left) == 0:
                return im_left, im_right
            
            # Convert deques to arrays
            spec_left = np.array(self.history_left).T  # Shape: (num_bins, time_frames)
            spec_right = np.array(self.history_right).T
            times = np.array(self.history_times)
            
            # Update image data
            im_left.set_data(spec_left)
            im_right.set_data(spec_right)
            
            # Update extents for proper time axis
            if len(times) > 0:
                time_extent = [times[0], times[-1], self.frequencies[0], self.frequencies[-1]]
                im_left.set_extent([times[0], times[-1], 0, self.num_bins])
                im_right.set_extent([times[0], times[-1], 0, self.num_bins])
            
            # Update y-axis labels to show actual frequencies
            freq_ticks = np.linspace(0, self.num_bins-1, 8, dtype=int)
            freq_labels = [f"{self.frequencies[i]:.0f}" for i in freq_ticks]
            ax_left.set_yticks(freq_ticks)
            ax_left.set_yticklabels(freq_labels)
            ax_right.set_yticks(freq_ticks)
            ax_right.set_yticklabels(freq_labels)
            
            # Update color limits based on current data
            vmax = max(spec_left.max(), spec_right.max()) if spec_left.size > 0 else 1.0
            im_left.set_clim(0, vmax)
            im_right.set_clim(0, vmax)
            
            # Update x-axis
            if len(times) > 1:
                ax_left.set_xlim(times[0], times[-1])
                ax_right.set_xlim(times[0], times[-1])
            
            return im_left, im_right
        
        # Create animation
        ani = FuncAnimation(
            fig,
            update_plot,
            interval=50,  # Update every 50ms
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return ani


# Example usage: Demo with animated spectrogram
if __name__ == "__main__":
    # Create converter with 10-second history
    converter = SpectrogramToAudio(num_bins=32, history_duration=10.0)
    
    # Example: Generate a time-varying spectrogram
    time_counter = [0]  # Use list to maintain state across function calls
    
    def get_spectrogram_coefficients():
        """
        Example function that generates spectrogram coefficients.
        Replace this with your actual spectrogram source.
        """
        t = time_counter[0]
        time_counter[0] += 0.01
        
        # Example 1: Moving peak across frequencies
        peak_pos = (np.sin(t * 2) + 1) / 2  # Oscillates between 0 and 1
        
        mags_left = np.exp(-((np.arange(converter.num_bins) - peak_pos * converter.num_bins) ** 2) / 20)
        mags_right = np.exp(-((np.arange(converter.num_bins) - (1-peak_pos) * converter.num_bins) ** 2) / 20)
        
        # Scale to reasonable amplitude
        mags_left *= 0.1
        mags_right *= 0.1
        
        return mags_left, mags_right
    
    # Start the audio stream
    converter.start_stream(get_spectrogram_coefficients)
    
    print("\nAudio stream started!")
    print("Opening visualization window...")
    print("Close the plot window to stop.")
    
    try:
        # Start visualization (this blocks until window is closed)
        converter.visualize_live()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        converter.stop_stream()