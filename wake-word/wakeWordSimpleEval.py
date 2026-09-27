import numpy as np
import time
import os
import sys
from pathlib import Path
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import tflite_runtime.interpreter as tflite

# ============================================
# CONFIGURATION
# ============================================
SAMPLE_RATE = 16000
WINDOW_MS = 1250
WINDOW_SAMPLES = int(SAMPLE_RATE * (WINDOW_MS / 1000.0))

NUM_MFCC = 13

CATEGORIES = ['w', 'o', 'n']
WAKE_IDX = 0  # 'w' is the wake word class

MODEL_PATH = str(Path(__file__).parent / 'models' / 'wake_word_baseline_float32.tflite')
TEST_DATA_PATH = '../test_data_partner/'

DEFAULT_WARMUP_RUNS = 10  # Excluded from timing statistics


# ============================================
# LOAD DATA
# ============================================
def load_test_data(test_path):
    """Load test samples from directory."""
    samples = []
    for label_idx, category in enumerate(CATEGORIES):
        cat_path = os.path.join(test_path, category)
        if not os.path.exists(cat_path):
            print(f"Warning: {cat_path} not found, skipping.")
            continue
        for wav_file in Path(cat_path).glob("*.wav"):
            rate, signal = wav.read(wav_file)
            if rate != SAMPLE_RATE:
                print(f"Warning: skipping {wav_file.name} (rate {rate} != {SAMPLE_RATE})")
                continue
            signal = signal.astype(np.float32)
            if signal.ndim == 2:
                signal = signal.mean(axis=1)
            if len(signal) < WINDOW_SAMPLES:
                signal = np.pad(signal, (0, WINDOW_SAMPLES - len(signal)))
            else:
                signal = signal[:WINDOW_SAMPLES]
            samples.append({'signal': signal, 'label': label_idx, 'file': wav_file.name})
    return samples


def preprocess(signal, input_details):
    """MFCC feature extraction + quantization handling."""
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    features = mfcc(signal, samplerate=SAMPLE_RATE, numcep=NUM_MFCC)
    features = features[..., np.newaxis]
    x = np.expand_dims(features, axis=0).astype(np.float32)

    input_scale, input_zero_point = input_details[0]['quantization']
    if input_scale > 0:  # quantized model
        limits = np.iinfo(input_details[0]['dtype'])
        x = np.clip(np.rint(x / input_scale + input_zero_point), limits.min, limits.max)
        x = x.astype(input_details[0]['dtype'])
    return x


# ============================================
# TEST MODEL
# ============================================
def test_model(model_path, test_samples, num_warmup=DEFAULT_WARMUP_RUNS):
    """Run inference on all samples and collect timing + confusion matrix.

    Warm-up runs are executed before measurement and are NOT counted in timing.
    """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantized = input_details[0]['quantization'][0] > 0

    # ---- Warm-up (excluded from statistics) ----
    print(f"\nRunning {num_warmup} warm-up inferences (excluded from timing)...")
    x_warm = preprocess(test_samples[0]['signal'], input_details)
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]['index'], x_warm)
        interpreter.invoke()

    # ---- Real measurement ----
    times = []
    # rows = true label, cols = predicted label
    confusion = np.zeros((3, 3), dtype=int)

    for sample in test_samples:
        x = preprocess(sample['signal'], input_details)

        start = time.perf_counter()  # higher resolution than time.time()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

        output = interpreter.get_tensor(output_details[0]['index'])
        pred = int(np.argmax(output))
        confusion[sample['label'], pred] += 1

    return times, confusion, is_quantized


# ============================================
# METRICS / REPORT
# ============================================
def report(model_path, test_path, times, confusion, is_quantized):
    total = confusion.sum()
    correct = int(np.trace(confusion))
    accuracy = correct / total * 100 if total > 0 else 0.0

    # Per-category totals (true label counts)
    n_wake  = int(confusion[WAKE_IDX, :].sum())
    n_other = int(confusion[1, :].sum())
    n_noise = int(confusion[2, :].sum())
    n_nonwake = n_other + n_noise

    # False Negative Rate: wake word missed (predicted as o or n)
    fn = int(confusion[WAKE_IDX, 1] + confusion[WAKE_IDX, 2])
    fnr = fn / n_wake * 100 if n_wake > 0 else 0.0

    # False Positive Rate: non-wake misclassified as wake
    fp = int(confusion[1, WAKE_IDX] + confusion[2, WAKE_IDX])
    fpr = fp / n_nonwake * 100 if n_nonwake > 0 else 0.0

    # Inference time statistics
    t = np.array(times)
    mean_t   = float(np.mean(t))
    std_t    = float(np.std(t, ddof=1)) if len(t) > 1 else 0.0
    median_t = float(np.median(t))
    min_t, max_t = float(np.min(t)), float(np.max(t))

    size_kb = os.path.getsize(model_path) / 1024.0

    print("\n" + "=" * 62)
    print(f"  Model      : {model_path}")
    print(f"  Test data  : {test_path}")
    print(f"  Quantized  : {is_quantized}")
    print(f"  File size  : {size_kb:.1f} KB")
    print("=" * 62)

    print("\n--- Inference Time (warm-up excluded) ---")
    print(f"  Mean   : {mean_t:8.3f} ms")
    print(f"  SD     : {std_t:8.3f} ms")
    print(f"  Median : {median_t:8.3f} ms")
    print(f"  Min/Max: {min_t:.3f} / {max_t:.3f} ms")

    print("\n--- Accuracy ---")
    print(f"  Total samples : {total}")
    print(f"  Correct       : {correct}")
    print(f"  Accuracy      : {accuracy:.2f} %")

    print("\n--- Per-class breakdown ---")
    print(f"  Wake  (w): {n_wake:4d}   correct: {confusion[0,0]:4d}")
    print(f"  Other (o): {n_other:4d}   correct: {confusion[1,1]:4d}")
    print(f"  Noise (n): {n_noise:4d}   correct: {confusion[2,2]:4d}")

    print("\n--- Error rates (wake-word focused) ---")
    print(f"  False Negative Rate (wake word missed)  : {fnr:6.2f} %  ({fn}/{n_wake})")
    print(f"  False Positive Rate (non-wake -> wake)  : {fpr:6.2f} %  ({fp}/{n_nonwake})")
    print(f"     - other -> wake : {int(confusion[1, WAKE_IDX])}/{n_other}")
    print(f"     - noise -> wake : {int(confusion[2, WAKE_IDX])}/{n_noise}")

    print("\n--- Confusion Matrix (rows=true, cols=pred) ---")
    print(f"            pred_w  pred_o  pred_n")
    print(f"  true_w   {confusion[0,0]:7d} {confusion[0,1]:7d} {confusion[0,2]:7d}")
    print(f"  true_o   {confusion[1,0]:7d} {confusion[1,1]:7d} {confusion[1,2]:7d}")
    print(f"  true_n   {confusion[2,0]:7d} {confusion[2,1]:7d} {confusion[2,2]:7d}")

    # Easy-to-paste CSV row for the report table
    print("\n--- CSV row (model,test_data,size_kb,quantized,acc,fpr,fnr,mean_ms,sd_ms,median_ms) ---")
    print(f"{os.path.basename(model_path)},{os.path.basename(os.path.normpath(test_path))},"
          f"{size_kb:.1f},{is_quantized},{accuracy:.2f},{fpr:.2f},{fnr:.2f},"
          f"{mean_t:.3f},{std_t:.3f},{median_t:.3f}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--test-data', default=TEST_DATA_PATH)
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP_RUNS,
                        help='Number of warm-up inferences excluded from timing')
    args = parser.parse_args()

    samples = load_test_data(args.test_data)
    if len(samples) == 0:
        print("No test data found!")
        sys.exit(1)
    print(f"Loaded {len(samples)} test samples from {args.test_data}")

    times, confusion, is_quantized = test_model(args.model, samples, num_warmup=args.warmup)
    report(args.model, args.test_data, times, confusion, is_quantized)