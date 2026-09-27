# Raspberry Pi 음성 처리와 TinyML

2026년 봄 EE305에서 I2S 오디오 입력을 분석하고, wake-word 모델과 파형 근사 모델을 학습해 TFLite로 변환했습니다. 신호 처리부터 양자화·실시간 추론까지 이어지는 임베디드 ML 실습입니다.

## 구현

| 폴더 | 내용 |
|---|---|
| `audio-processing/` | 실시간 spectrum·Mel spectrogram |
| `wake-word/` | 녹음, MFCC 전처리, baseline/augmentation 학습, float/INT8 평가, streaming |
| `tinyml-waveforms/` | sine·rect·sawtooth 근사 모델의 학습과 float/INT8 비교 |

교과 실습 코드를 기반으로 모델을 학습·변환하고 Raspberry Pi에서 실험했습니다. 추론 시간 측정에서는 warm-up을 제외했으며, 혼동행렬과 오탐·미탐 분석, streaming 조건 조정을 구현했습니다. 양자화 입력은 반올림 후 dtype 범위로 제한하고, 다채널 WAV는 mono로 변환합니다.

`streamWord_spring26.py`는 수업 제공 코드를 기반으로 합니다. `wake-word/models/`에는 학습한 baseline/augmented 모델이 있고, `tinyml-waveforms/*.tflite`에는 파형 근사 모델이 있습니다. Streaming 실행에 필요한 모델은 별도로 준비합니다.

## 환경과 실행

Raspberry Pi Linux, I2S `voicehat` 입력 48 kHz, 모델 입력 16 kHz, 1.25초 오디오 창을 사용합니다. 장치와 Python 버전에 호환되는 TFLite runtime을 설치합니다.

```bash
python -m pip install -r requirements-pi.txt
python wake-word/wakeWordSimpleEval.py --model wake-word/models/wake_word_baseline_float32.tflite --test-data /path/to/test_data
python tinyml-waveforms/pi.py
```

평가 데이터는 `w/`, `o/`, `n/` 폴더의 16 kHz WAV로 준비합니다. 학습에는 `requirements-training.txt`의 TensorFlow 환경을 사용하고, notebook의 데이터 경로와 오디오 장치 설정을 맞춥니다. Streaming 모델 경로는 `--model`로 지정합니다(`--help` 참고).

## 실험 자료와 테스트

- Wake-word baseline/augmented 및 파형 근사 모델의 float/INT8 TFLite 파일
- 2026-05-22 streaming 실험의 집계 로그: 수업 제공 streaming 모델을 사용한 실행 기록
- Python 문법 검사와 합성 입력을 사용한 양자화 경계 검사 2개 통과

자동 테스트는 입력 처리와 양자화 연산을 대상으로 합니다. Raspberry Pi 오디오 입력과 TFLite 모델 추론은 자동 테스트 범위에 포함되지 않습니다.
