import os
import time
import wave
import numpy as np
import pyaudio
from datetime import datetime

# ===== CONFIGURATION =====
I2S_DEVICE_NAME = "voicehat"
I2S_RATE = 48000
I2S_CHUNK = 1024
I2S_RECORD_SECONDS = 10
I2S_SKIP_SECONDS = 2  # Warm-up time
I2S_AMPLIFY_DB = 30

RMS_THRESHOLD_DB  = -40  # Threshold (dB) between silence and speech (adjust 30-40, adjust as needed, most likely problem)
MIN_WORD_DURATION = 0.4  # Shortest word duration we expect (possibly adjust for short “other” words)
MAX_WORD_DURATION = 2.0  # Longest word duration (likely reasonable; clipping irrelevant)
WINDOW_SIZE       = 0.02 # Size (in seconds) of the chunks of audio we analyze (20ms)
MAX_GAP_FRAMES    = 3    # Silence period between words (in #window_size) (adjust 1–5)

PRE_PAD_SEC         = 0.1  # the amount of audio from the sample we pad before onset detection
POST_PAD_SEC        = 0.3  # the amount of audio from the sample we pad after end detection


DATASET_DIR = "wake_word_dataset"
TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 1.25    # for four syllables
LABELS = ["w", "o", "n"] # (w)ake_word, (o)ther_word, (n)oise

# ===== SETUP =====
os.makedirs(DATASET_DIR, exist_ok=True)

# ===== HELPERS =====
# Open audio device
def find_device(p, name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0 and name.lower() in dev.get('name').lower():
            print(f"Using device: {dev.get('name')}")
            return i
    raise RuntimeError(f"Device '{name}' not found")

# apply an amplification to the samples
def amplify(sig, db):
    return np.clip(sig * (10 ** (db / 20)), -32768, 32767).astype(np.int16)

# report the RMS in dB
def rms_db(sig):
    rms = np.sqrt(np.mean(sig.astype(np.float32) ** 2))
    return 20 * np.log10(rms / 32768.0) if rms > 0 else -100

# resample from the mic's 48Khz to the 16Khz we want
def resample(sig, orig_rate, target_rate):
    if orig_rate == target_rate:
        return sig
    target_len = int(len(sig) * target_rate / orig_rate)
    return np.interp(np.linspace(0, len(sig)-1, target_len), np.arange(len(sig)), sig).astype(np.int16)

# pad or crop to the fixed size. 
def pad_or_crop(sig, target_len):
    diff = len(sig) - target_len
    if diff == 0:
        return sig
    if diff > 0:
        start = diff // 2
        return sig[start:start + target_len]
    pad = [-diff // 2, -diff + diff // 2]
    return np.pad(sig, pad, constant_values=0)

# ===== SEGMENTATION =====
def smooth_gaps(speech, max_gap):
    n = len(speech)
    i = 0
    while i < len(speech):
        if not speech[i]:
            start = i
            while i < n and not speech[i]: 
                i += 1
            if start > 0 and i < n and speech[start-1] and speech[i] and (i - start) <= max_gap: 
                speech[start:i] = True
        else:
            i += 1
    return speech

def segment_audio(audio):
    win = int(WINDOW_SIZE * I2S_RATE)  # 20 ms windows for smoother VAD
    if len(audio) < win:
        return []

    rms_vals = np.array([rms_db(audio[i:i+win]) for i in range(0, len(audio), win)])
    above = rms_vals > RMS_THRESHOLD_DB
    smooth_gaps(above, MAX_GAP_FRAMES)
  
    # find rising and falling edges
    edges = np.diff(above.astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]

    # handle leading/trailing speech
    if above[0]: starts = np.insert(starts, 0, 0)
    if above[-1]: ends = np.append(ends, len(rms_vals) - 1)

    segments = []
    for s, e in zip(starts, ends):
        dur = (e - s) * win / I2S_RATE
        if MIN_WORD_DURATION <= dur <= MAX_WORD_DURATION:
            s_idx = max(0, s * win - int(PRE_PAD_SEC * I2S_RATE))
            e_idx = min(len(audio), e * win + int(POST_PAD_SEC * I2S_RATE))
            segments.append(audio[s_idx:e_idx])
    print(f"Speech mode: Found {len(segments)} segments")
    return segments



def segment_noise(audio):
    target_len = int(TARGET_SAMPLE_RATE * TARGET_DURATION * (I2S_RATE / TARGET_SAMPLE_RATE))
    segments = []
    
    hop = target_len   # non-overlapping
    for start in range(0, len(audio), hop):
        end = start + hop
        chunk = audio[start:end]
        if len(chunk) < hop // 2:   # avoid tiny fragments at end
            continue
        segments.append(chunk)
    
    print(f"Noise mode: created {len(segments)} fixed length segments")
    return segments


# ===== DATASET I/O =====
def save_segments(segments, label):
    if not segments:
        print("No segments to save.")
        return

    # Automatically cull edge segments for speech recordings; truncation is highly likely. 
    if label in ["w", "o"] and len(segments) >= 3:
        print("Auto-culling first/last speech segments to avoid truncated clips.")
        segments = segments[1:-1]

    outdir = os.path.join(DATASET_DIR, label)
    os.makedirs(outdir, exist_ok=True)
    target_len = int(TARGET_SAMPLE_RATE * TARGET_DURATION)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, seg in enumerate(segments, 1):
        seg_length = (len(seg) / I2S_RATE) - PRE_PAD_SEC - POST_PAD_SEC
        seg_rms = rms_db(seg)
        seg = resample(seg, I2S_RATE, TARGET_SAMPLE_RATE)
        seg = pad_or_crop(seg, target_len)
        fname = os.path.join(outdir, f"{ts}_{i:03d}.wav")
        with wave.open(fname, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(seg.tobytes())
        print(f"  [{i}] {fname} ({seg_rms:.1f} dB and {seg_length:.2f} seconds of audio before padding/cropping).")
    print(f"✓ Saved {len(segments)} to '{label}'")

def dataset_stats():
    return {lbl: len([f for f in os.listdir(os.path.join(DATASET_DIR, lbl)) if f.endswith('.wav')]) 
            if os.path.exists(os.path.join(DATASET_DIR, lbl)) else 0 for lbl in LABELS}

# ===== RECORDING =====
def record_and_segment(p, device_index, label):
    print(f"\n=== Recording '{label}' ===")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=I2S_RATE,
                    input=True, frames_per_buffer=I2S_CHUNK,
                    input_device_index=device_index)
    frames = []
    start = time.time()
    for _ in range(int(I2S_RATE / I2S_CHUNK * I2S_RECORD_SECONDS)):
        data = np.frombuffer(stream.read(I2S_CHUNK, exception_on_overflow=False), dtype=np.int16).copy()
        data = data - np.mean(data)
        data = amplify(data, I2S_AMPLIFY_DB)
        if time.time() - start > I2S_SKIP_SECONDS:
            frames.append(data)
    stream.stop_stream(); stream.close()
    audio = np.concatenate(frames)
    if label == "n":  
        segments = segment_noise(audio)
    else:
        segments = segment_audio(audio)
    save_segments(segments, label)

# ===== MAIN =====
def main():
    print("\n=== WAKE WORD DATASET RECORDER ===")
    p = pyaudio.PyAudio()
    device_idx = find_device(p, I2S_DEVICE_NAME)
    print("\nCurrent dataset:", dataset_stats())
    print(f"Sample rate: {TARGET_SAMPLE_RATE} Hz, Duration: {TARGET_DURATION}s")
    try:
        while True:
            print("Labels are (w)ake_word, (o)ther_word, (n)oise")
            label = input(f"\nEnter label ({'/'.join(LABELS)}) or 'q' to quit: ").strip()
            if label == 'q': break
            if label not in LABELS:
                print("Invalid label.")
                continue
            record_and_segment(p, device_idx, label)
            print("Updated:", dataset_stats())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()