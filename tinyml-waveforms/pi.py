import math, time
import numpy as np
import plotext as plt
from tflite_runtime.interpreter import Interpreter
from pathlib import Path

WAVE_FUNCS = {
    'sine':     lambda x: np.sin(3 * x),
    'rect':     lambda x: np.abs(np.sin(1.5 * x)),
    'sawtooth': lambda x: 2*(3*x/(2*np.pi) - np.floor(3*x/(2*np.pi)+0.5))
}

MODEL_FILES = [
    ('sine',     'sine_float.tflite',     False, 'Float32'),
    ('sine',     'sine_quant.tflite',     True,  'INT8'),
    ('rect',     'rect_float.tflite',     False, 'Float32'),
    ('rect',     'rect_quant.tflite',     True,  'INT8'),
    ('sawtooth', 'sawtooth_float.tflite', False, 'Float32'),
    ('sawtooth', 'sawtooth_quant.tflite', True,  'INT8'),
]

def quantize_scalar(value, scale, zero_point, dtype):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale must be finite and positive")
    limits = np.iinfo(dtype)
    return np.array([[np.clip(np.rint(value / scale + zero_point), limits.min, limits.max)]], dtype=dtype)

x_vals = np.arange(0, 2*math.pi, 0.05)

def benchmark(path, func_name, is_quant):
    y_true = WAVE_FUNCS[func_name](x_vals)
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    if is_quant:
        in_scale,  in_zero  = in_det[0]['quantization']
        out_scale, out_zero = out_det[0]['quantization']

    # 워밍업 10회
    for _ in range(10):
        if is_quant:
            x_in = quantize_scalar(0, in_scale, in_zero, in_det[0]["dtype"])
        else:
            x_in = np.array([[0.0]], dtype=np.float32)
        interp.set_tensor(in_det[0]['index'], x_in)
        interp.invoke()

    y_pred, times = [], []
    for x in x_vals:
        if is_quant:
            x_in = quantize_scalar(x, in_scale, in_zero, in_det[0]["dtype"])
        else:
            x_in = np.array([[x]], dtype=np.float32)
        interp.set_tensor(in_det[0]['index'], x_in)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
        y = interp.get_tensor(out_det[0]['index'])[0][0]
        if is_quant:
            y = (float(y) - out_zero) * out_scale
        y_pred.append(y)

    y_pred = np.array(y_pred)
    times  = np.array(times)
    size   = Path(path).stat().st_size
    mae     = np.mean(np.abs(y_pred - y_true))
    rmse    = np.sqrt(np.mean((y_pred - y_true)**2))
    max_err = np.max(np.abs(y_pred - y_true))

    return {
        'size':    size,
        'mean':    np.mean(times),
        'sd':      np.std(times),
        'median':  np.median(times),
        'mae':     mae,
        'rmse':    rmse,
        'max_err': max_err,
        'y_pred':  y_pred,
        'y_true':  y_true,
    }

# 표 출력
print(f"{'Dataset':<10} {'Type':<8} {'Size(B)':>8} {'Mean(ms)':>10} {'SD(ms)':>8} {'Med(ms)':>8} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8}")
print("-" * 82)

results = []
for func_name, path, is_quant, model_type in MODEL_FILES:
    path = str(Path(__file__).parent / path)
    if not Path(path).exists():
        print(f"Not found: {path}")
        continue
    r = benchmark(path, func_name, is_quant)
    results.append((func_name, model_type, path, r))
    print(f"{func_name:<10} {model_type:<8} {r['size']:>8} {r['mean']:>10.4f} {r['sd']:>8.4f} {r['median']:>8.4f} {r['rmse']:>8.6f} {r['mae']:>8.6f} {r['max_err']:>8.6f}")

# 플롯: 6개 모델 각각 predicted vs true
for func_name, model_type, path, r in results:
    plt.clf()
    plt.plot(x_vals, r['y_true'], label='True')
    plt.plot(x_vals, r['y_pred'], label='Predicted')
    plt.title(f"{func_name} {model_type} - Predicted vs True")
    plt.xlabel("x (radians)")
    plt.ylabel("y")
    plt.show()