#!/usr/bin/env python3
# Real-Time Wake Word Detection System

# Suppress warnings related to near-zero floating point numbers (irrelevant for our purpsoes)
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os
import sys
import contextlib
import time
import termios
import tty
import select
import argparse
import wave
import datetime
from collections import deque
import numpy as np
import scipy.io.wavfile as wav
import pyaudio
import tflite_runtime.interpreter as tflite
from python_speech_features import mfcc

# ============================================
# CONFIGURATION
# ============================================

# Audio Configuration
I2S_DEVICE_NAME    = "voicehat"
I2S_RATE           = 48000         # Hardware sample rate (voicehat only supports 48kHz)
I2S_AMPLIFY_DB     = 30            # Match training data amplification

TARGET_SAMPLE_RATE = 16000         # Target sample rate for model

MODEL_PATH = 'wake_word_float32_spring26_38k_culled_aug.tflite'

# Window Configuration
WINDOW_SIZE_MS   = 1250             # 1.25 seconds 
WINDOW_STRIDE_MS = 40               # Process every 40ms
CHUNK_SIZE       = 1920             # 40ms at 48kHz (will be downsampled to 1/3 or 640 samples)
                                    # or: TARGET_SAMPLE_RATE / 1000 / WINDOW_STRIDE_MS
NUM_MFCC         = 13

# Detection Parameters
CONFIDENCE_THRESHOLD    = 0.60      # relatively high threshold (reduce false positive)
CONSECUTIVE_DETECTIONS  = 3         # moderate frames required (high false positive)
COOLDOWN_MS             = 2000      # duration of cooldown period
COOLDOWN_FRAMES         = COOLDOWN_MS // WINDOW_STRIDE_MS # expressed in frames


# ============================================
# 1. KEY HANDLING: non-blocking ssh keyboard input
# ============================================
class KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # raw mode
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self):
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None


# ============================================
# 2. TERMINAL DISPLAY: functions to show meters 
# ============================================
def meter_bar(db, width=30):
    db = max(-60, min(0, db))
    filled = int(((db + 60) / 60) * width)
    return "█" * filled + " " * (width - filled)

def confidence_bar(conf, width=30):
    conf = max(0.0, min(1.0, float(conf)))
    filled = int(conf * width)
    return "█" * filled + " " * (width - filled)

# Add your own optional data here as the final parameter. Should be a string. 
def print_status(vol_db, conf, data=None):
    vol_text = f"{vol_db:6.1f} dB"
    conf_text = f"{conf*100:5.1f}%"
    line = (
        f"VOL:  [{meter_bar(vol_db)}] {vol_text}   "
        f"CONF: [{confidence_bar(conf)}] {conf_text}   "
    )
    if data is not None:
        line = line + "   " + str(data)
    print(line, end="\r", flush=True)


# ============================================
# 3. AUDIO BUFFER: Simple ring buffer for audio
# ============================================
class AudioBuffer:
    def __init__(self, sample_rate, duration_seconds):
        buffer_size = int(sample_rate * duration_seconds)
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.write_pos = 0
        
    # add data to ring buffer
    def add(self, audio_chunk):
        chunk_len = len(audio_chunk)
        # Handle writing round the wrap
        if self.write_pos + chunk_len <= len(self.buffer): 
            self.buffer[self.write_pos:self.write_pos + chunk_len] = audio_chunk
        else:
            first_part = len(self.buffer) - self.write_pos
            self.buffer[self.write_pos:] = audio_chunk[:first_part]
            self.buffer[:chunk_len - first_part] = audio_chunk[first_part:]
        
        self.write_pos = (self.write_pos + chunk_len) % len(self.buffer)
    
    # get n samples from ring buffer
    def get_last_n_samples(self, n):
        start = (self.write_pos - n) % len(self.buffer)
        if start + n <= len(self.buffer):
            return self.buffer[start:start + n].copy()
        else:
            # Handle reading round the wrap
            part1 = self.buffer[start:]
            part2 = self.buffer[:n - len(part1)]
            return np.concatenate([part1, part2])


# ============================================
# 4. STATE MACHINE: simplest version, very limited jitter handling
# ============================================
class WakeWordDetector:
    def __init__(self, threshold, consecutive_needed, cooldown):
        self.threshold          = threshold             # threshold for wake word detection
        self.consecutive_needed = consecutive_needed    # consecutive frames avoid threshold for wake word detection
        self.cooldown_frames    = cooldown              # number of frames for cooldown
        
        self.consecutive_count  = 0                     # state var - above threshold counter
        self.cooldown_count     = 0                     # state var - cooldown counter
        self.in_cooldown        = False                 # Flag to indicate cooldown state
        self.total_detections   = 0                     # Summary stat (persistant)
        
    # Update detector with new confidence score - returns true if wake word detected
    def update(self, wake_word_confidence):
        # In cooldown state - ignore everything and count cooldown time down
        if self.in_cooldown:
            self.cooldown_count += 1
            if self.cooldown_count >= self.cooldown_frames:
                self.in_cooldown = False
                self.cooldown_count = 0
                self.consecutive_count = 0
            return False
        
        # Else check confidence against threshold and update cons. count if above
        if wake_word_confidence > self.threshold:
            self.consecutive_count += 1                 # inc count
            if self.consecutive_count >= self.consecutive_needed: # check count
                self.total_detections += 1              # add one to persistant detection count
                self.in_cooldown       = True           # move to cooldown state
                self.cooldown_count    = 0              # reset cooldown count
                return True                             # Trigger wake word found! 
        else:
            # Confidence dropped
            # Option 1: Decrease count to handle jitter. Tolerates some drops
            # self.consecutive_count = max(0, self.consecutive_count - 1)
            # Option 2: A harder reset here - go straight to zero on one drop. 
            self.consecutive_count = 0

        return False
    
    # Get current state info
    def get_state(self):
        if self.in_cooldown:
            return f"COOLDOWN ({self.cooldown_count}/{self.cooldown_frames})"
        elif self.consecutive_count > 0:
            return f"DETECTING ({self.consecutive_count}/{self.consecutive_needed})"
        else:
            return "IDLE"


# ============================================
# 5. STREAMING SYSTEM: Process audio and perform inference
# ============================================
class StreamingSystem:    

    # amplify a signal by a given dB 
    def amplify(self, sig, db):
        return np.clip(sig * (10 ** (db / 20)), -32768, 32767) #.astype(np.int16)     

    # resample a signal (beware: np.interp is simple but inefficient)
    def resample(self, sig, orig_rate, target_rate):
        if orig_rate == target_rate:
            return sig
        target_len = int(len(sig) * target_rate / orig_rate)
        return np.interp(np.linspace(0, len(sig)-1, target_len), np.arange(len(sig)), sig) #.astype(np.int16)

    # Helper function to check the buffer is coherent - saves the current version to a file. 
    def save_circular_buffer_snapshot(self):
        window = self.buffer.get_last_n_samples(self.window_samples)  # Get samples
        #audio_int16 = np.clip(window, -32768, 32767).astype(np.int16) # Convert float32 in int16 range to int16
        audio_int16 = np.clip(window * 32768.0, -32768, 32767).astype(np.int16)  # reverse internal [-1,1] scaling
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # get date/time for filename  
        filename = f"circular_buffer_{timestamp}.wav"                 # set filename
        with wave.open(filename, 'w') as wf:                          # write audio file
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        print(f"\nSaved circular buffer to {filename}")               

    # Helper function to test the model with a wav file (for example, one of the saved circular buffer files)
    # Or one of the training files - can help with a sanity check
    def test_with_wav(self, wav_path):
        print(f"\n=== Testing with WAV file: {wav_path} ===")
        rate, signal = wav.read(wav_path)                                       # Load WAV 
        if rate!=TARGET_SAMPLE_RATE:
            print(f"\nSample and target sample rate differ; resampling.")
            signal = self.resample(signal, rate, TARGET_SAMPLE_RATE)
        #signal = signal / np.max(np.abs(signal) + 1e-10)                       # Normalize 
        signal = signal.astype(np.float32) / 32768.0 
        features = mfcc(signal, samplerate=TARGET_SAMPLE_RATE, numcep=NUM_MFCC) # Extract features
        confidence_scores = self._infer(features)                               # inference
        
        print(f"\nResults:")
        print(f"  Wake word (w): {confidence_scores[0]:.3f}")
        print(f"  Other word (o): {confidence_scores[1]:.3f}")
        print(f"  Noise (n): {confidence_scores[2]:.3f}")
        print("="*50 + "\n")

    # Warm up model - run it with dummy data
    def _warmup(self):
        dummy = np.zeros((1,) + tuple(self.input_details[0]['shape'][1:]), 
                        dtype=self.input_details[0]['dtype'])
        for _ in range(3):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy)
            self.interpreter.invoke()

    # modified
    def close_logging(self):
        """
        Close the log file if it is currently open.
        """

        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()
    #######


    # init the streaming system
    def __init__(self, model_path, threshold, consecutive, cooldown, profile=False, opt_features=False):
        print(f"Loading model: {model_path}")
        self.interpreter    = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_scale         = self.input_details[0]['quantization'][0]
        self.is_quantized   = input_scale > 0
        print(f"Model: {'INT8' if self.is_quantized else 'Float32'}")
        
        # Audio setup - samples per window, per stride (e.g., 40ms) and when downsampled to 16k (from 48k)
        self.window_samples = int(TARGET_SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
        self.stride_samples = int(TARGET_SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
        self.downsampled_chunk_size = int(CHUNK_SIZE * TARGET_SAMPLE_RATE / I2S_RATE)
        
        # volume bar paramters (for smoothing; to reduce jitter)
        self.smoothed_volume = -100.0  # initialize
        self.volume_alpha    = 0.3     # smoothing factor, 0 < alpha <= 1

        # buffer for incrementing MFCC features. 
        self.mfcc_cache = None  # initially empty
        self.opt_features = opt_features # by default off; calc features on full 1250 frames

        # Components
        self.buffer     = AudioBuffer(TARGET_SAMPLE_RATE, 3.0)
        self.detector   = WakeWordDetector(threshold, consecutive, cooldown)
        
        # Performance tracking
        self.profile            = profile
        self.inference_times    = deque(maxlen=100)
        self.frame_count        = 0
        self.peak_confidence    = 0.0
        self.peak_conf_hold     = 0  # Hold peak for a bit after an utterance ends

        ############ modified#####
        # Logging setup
        self.log_file = None
        # Create a CSV log file when profile mode is enabled
        if self.profile:
            fn = "log_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            # Open log file in write mode
            self.log_file = open(fn, "w")
            # Write CSV header
            self.log_file.write("timestamp,confidence,peak_confidence\n")
        
        #####

        # Warm up
        self._warmup()
        
        print(f"\nReady to detect 'Hey KAISTian'")
        print(f"Capture rate: {I2S_RATE}Hz, Model rate: {TARGET_SAMPLE_RATE}Hz")
        print(f"Threshold: {threshold}, Consecutive: {consecutive}, Cooldown: {cooldown} frames")
        print(f"Audio preprocessing: DC offset removal + {I2S_AMPLIFY_DB}dB amplification + downsample\n")
    
    # optional - testing MFCC caching for speed/accuracy. 
    def _get_mfcc_cached(self, window):
        new_audio = self.buffer.get_last_n_samples(self.downsampled_chunk_size)
        new_features = mfcc(new_audio, samplerate=TARGET_SAMPLE_RATE, numcep=NUM_MFCC)
        if self.mfcc_cache is None:
            self.mfcc_cache = mfcc(window, samplerate=TARGET_SAMPLE_RATE, numcep=NUM_MFCC)
        else:
            self.mfcc_cache = np.concatenate([self.mfcc_cache[len(new_features):], new_features])
        return self.mfcc_cache


    def process_chunk(self, audio_chunk_in):
        audio_chunk = audio_chunk_in.astype(np.float32)                          # Convert to float32
        audio_chunk -=np.mean(audio_chunk)                                       # Remove DC offset: note that using "-=" allows efficient in-place calculation. 
        audio_chunk = self.amplify(audio_chunk, I2S_AMPLIFY_DB)                  # Apply amplification
        audio_chunk = audio_chunk / 32768.0                                      # convert to -1, 1
        audio_chunk = self.resample(audio_chunk, I2S_RATE, TARGET_SAMPLE_RATE)   # Downsample

        #  Calculate RMS volume for meter (shows audio as sent to model; after pre-processing)
        rms = np.sqrt(np.mean(audio_chunk ** 2))                                 # RMS 
        vol_db_raw = 20 * np.log10(rms) if rms > 0 else -100
        self.smoothed_volume = (self.volume_alpha * vol_db_raw + (1 - self.volume_alpha) * self.smoothed_volume)

        # accumulate the new data in the buffer
        self.buffer.add(audio_chunk)

        self.frame_count += 1
        
        start = time.time()
        window = self.buffer.get_last_n_samples(self.window_samples)            # Get window and extract features
        if not self.opt_features:
            # Calc MFCCs. Note this mirrors the colab script, but is highly inefficient - it re-computes features
            # over the whole 1.25s sample every 40ms. We do this to match the colab training script (which also
            # works in this way). In a real system, we would calculate features on 40ms windows and accumulate/cull
            # however, this leads to various production level issues (e.g., with feature selection and windowing)
            # that go beyond the scope of the class. So we are keeping it simple - this inefficient option is the default
            features = mfcc(window, samplerate=TARGET_SAMPLE_RATE, numcep=NUM_MFCC) 
        else:
            # This is a simple version of that more complex implementation
            # It assumes (generally incorrectly) that incrementallt calculated MFCC features == whole sample features
            # Use with caution, as features are systemically different than in the training script. 
            # Note this is OFF by default in the script. 
            features = self._get_mfcc_cached(window)
        featureTime = (time.time() - start) * 1000 

        # Run inference - track time
        start = time.time()
        confidence_scores = self._infer(features)                          
        inference_time = (time.time() - start) * 1000
        self.inference_times.append(inference_time)
        
        # output - draw bars to terminal for volume and confidence in wake word clcass
        print_status(self.smoothed_volume, confidence_scores[0], f"{featureTime:.2f}")

        # update peak confidence score for wake word
        if confidence_scores[0] > self.peak_confidence:
            self.peak_confidence = confidence_scores[0]
            self.peak_conf_hold = self.frame_count + 20  # Hold for ~0.8 seconds; adjust as needed
        # reset peak confidence if the confidence for wake word is ~zero
        if confidence_scores[0] < 0.01 and self.frame_count > self.peak_conf_hold:
            self.peak_confidence = 0.0

        # Update detector; print output if detected
        if self.detector.update(confidence_scores[0]):
            self._on_detection(confidence_scores[0])
        
        # Optionally print some performance data (if profile == true)
        if self.profile and len(self.inference_times) == 100:
            self._print_stats(confidence_scores)

        # Add temperature checks every 10 seconds
        if self.frame_count % 250 == 0:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp') as f:
                    temp = int(f.read()) / 1000.0
                    if temp > 78: # after 80 the PI will "automatically throttle its performance to prevent damage"
                        print(f"\n⚠️  Pi is hot ({temp:.1f}°C)! Pausing to cool down.")
                        print(f"Pausing for 30 seconds to cool down...")
                        print(f"Or quit and take a break.")
                        print(f"{'='*50}\n")
                        time.sleep(30)
                
                        # Check again after cooling - we need open again to get an updated reading
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f2:
                            new_temp = int(f2.read()) / 1000.0
                            print(f"Temperature after cooling: {new_temp:.1f}°C")
                            print("Beware - performance may throttle now due to audio buffer overruns. Restart your Pi to start recording again!")
            except Exception as e:
                print(f"\n⚠️ Exception {e} on reading the Pi temperature.")
                
    

    def _infer(self, features):
        input_data = features[np.newaxis, ..., np.newaxis]
        
        # Quantize if needed
        if self.is_quantized:
            scale, zero_point = self.input_details[0]['quantization']
            input_data = input_data / scale + zero_point
            input_data = input_data.astype(self.input_details[0]['dtype'])
        else:
            input_data = input_data.astype(np.float32)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Dequantize if needed - cut this if not needed. CHECKED IT WITH QUANT MODEL
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        return output

    
    def _on_detection(self, confidence):
        print(f"\n{'='*50}")
        print(f"🎯 WAKE WORD DETECTED!")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Time: {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}\n")

        ######### modified ##############
        # Record detection information to the log file
        if self.log_file is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Store detection timestamp, current confidence, and peak confidence
            self.log_file.write(
                f"{timestamp},{confidence:.3f},{self.peak_confidence:.3f}\n"
            )
            # Immediately save data to disk
            self.log_file.flush()
    ################

    def _print_stats(self, confidence_scores):
        mean_time = np.mean(self.inference_times)
        
        print(f"\nState: {self.detector.get_state()}")
        print(f"Confidence: W={confidence_scores[0]:.3f} O={confidence_scores[1]:.3f} N={confidence_scores[2]:.3f}")
        print(f"Peak W Conf.: {self.peak_confidence:.3f}")
        print(f"Inference: {mean_time:.1f}ms")
        print(f"Detections: {self.detector.total_detections}")
        
        if mean_time > WINDOW_STRIDE_MS:
            print(f"⚠️ Warning: Inference too slow!")


# ============================================
# 6. AUDIO CAPTURE: Acquire audio signals; main loop
# ============================================
# add error supression for audio init (check last week's script)
def find_audio_device(p, name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0 and name.lower() in dev.get('name').lower():
            print(f"Using device: {dev.get('name')}")
            return i
    raise RuntimeError(f"Device '{name}' not found")

def run_detection(model_path, device_name=I2S_DEVICE_NAME,
                 threshold=CONFIDENCE_THRESHOLD,
                 consecutive=CONSECUTIVE_DETECTIONS,
                 cooldown=COOLDOWN_FRAMES,
                 profile=False,
                 opt_features=False,
                 test_wav=None):  # Add this parameter to test with a file
    # Initialize system
    system = StreamingSystem(model_path, threshold, consecutive, cooldown, profile, opt_features)
    
    # Test with WAV file if provided (as a command line arg)
    if test_wav:
        system.test_with_wav(test_wav)
        system.close_logging()
        return  # Exit after testing

    # Setup audio - using context lib here to suppress messages
    # If you are having issues recording, remove this line to check the audio init messages 
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        p = pyaudio.PyAudio()
    device_idx = find_audio_device(p, device_name)
    
    # open audio stream - use a internal buffer larger than our chunk size to
    # better accomodate any latency in main loop execution
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=I2S_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE*4,
        input_device_index=device_idx)
    
    print("Listening... (Press 's' to save latest buffer, 'c' to clear peak conf, Ctrl+C to stop)\n")

    # Non-blocking key poller - this is the main loop of the program
    with KeyPoller() as poller:
        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False) # read from audio stream. Note we disable overflow warnings here. 
                system.process_chunk(np.frombuffer(data, dtype=np.int16))   # process audio in streaming sys.

                key = poller.poll()                                         # check key press
                if key == 's':                                              # responds to 's' over ssh
                    system.save_circular_buffer_snapshot()                  # save a file from the ring buffer - debug!
                    time.sleep(0.3)                                         # debounce - prevent rep. activation
                elif key == 'c':
                    system.peak_confidence = 0.0                            # clear peak conf. score
                    print("\nPeak confidence cleared.")

        except KeyboardInterrupt:
            print("\nStopping...")                                          # ctrl c (or whatever this is on your PC)
        finally:
            ###### modified ########
            # Ensure the log file is properly closed before exiting
            system.close_logging()
            ###########
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Done.")


# ============================================
# 7 MAIN: Process command line args, start main loop
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Wake word detection')
    parser.add_argument('--model',                      default=MODEL_PATH,             help='TFLite model path')
    parser.add_argument('--threshold',      type=float, default=CONFIDENCE_THRESHOLD,   help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--consecutive',    type=int,   default=CONSECUTIVE_DETECTIONS, help='Consecutive frames needed')
    parser.add_argument('--cooldown',       type=int,   default=COOLDOWN_FRAMES,        help='Cooldown frames')
    parser.add_argument('--profile',        action='store_true',                        help='Show performance stats')
    parser.add_argument('--opt_features',   action='store_true',                        help='Switches to an efficient (not default) feature calc mode - see inline comments for details')
    parser.add_argument('--testwav',                                                    help='Test with a WAV file instead of live audio')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    run_detection(args.model, I2S_DEVICE_NAME, args.threshold, 
                 args.consecutive, args.cooldown, args.profile, args.opt_features, args.testwav)

if __name__ == '__main__':
    main()