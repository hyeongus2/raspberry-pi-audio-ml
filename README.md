# Raspberry Pi 음성 처리와 TinyML 실습

2026년 봄 EE305에서 I2S 입력의 spectrum/Mel 분석, wake-word 모델 학습·추론, 파형 함수의 TFLite 양자화 실험을 수행했습니다. 본인 학습 모델과 수정 구현·측정 로그를 모았습니다.

| 영역 | 내용 |
|---|---|
| `audio-processing/` | 실시간 spectrum·Mel spectrogram notebook |
| `wake-word/` | 녹음, MFCC 전처리, baseline/augmentation 학습, float/INT8 평가, streaming 조건 조정 |
| `tinyml-waveforms/` | sine·rect·sawtooth 학습과 float/INT8 모델 비교 |

## 구현과 역할

수업 제공 실습 코드를 기반으로 모델 학습·변환과 Pi 실험을 수행하고, 추론 시간에서 warm-up을 제외하는 평가·혼동행렬·오탐/미탐 분석 및 streaming 조건을 수정했습니다. `streamWord_spring26.py`의 기반은 수업 제공 코드이며 `modified` 주석이 있는 개인 변경과 함께 보존했습니다. 녹음 원음, 파트너 음성, 수업 제공 streaming 모델은 포함하지 않습니다. `wake-word/models/`는 개인 baseline/augmented 모델이고 `tinyml-waveforms/*.tflite`는 파형 실험 모델입니다. 파형 근사 결과를 음성 인식 성과로 합치지 않습니다.

## 환경과 실행

추론은 Raspberry Pi Linux의 I2S `voicehat` 입력(48 kHz), 모델 입력 16 kHz, 1.25초 창을 전제로 한 코드입니다. 정확한 Pi 보드·OS 버전은 보관 자료에서 확정하지 않았으므로 실제 장치와 TFLite runtime 지원 Python 버전을 확인해 설치합니다.

```bash
python -m pip install -r requirements-pi.txt
python wake-word/wakeWordSimpleEval.py --model wake-word/models/wake_word_baseline_float32.tflite --test-data /path/to/test_data
python tinyml-waveforms/pi.py
```

평가 입력은 `w/`, `o/`, `n/` 폴더 아래 16 kHz WAV입니다. 다채널 WAV는 mono 평균으로 변환합니다. 학습은 별도 Python/TensorFlow 환경에서 `requirements-training.txt`를 설치한 뒤 해당 notebook을 실행합니다. 녹음·dataset 경로와 오디오 장치 설정을 자신의 환경에 맞춥니다. streaming 코드는 제공받은 모델 경로를 CLI의 `--model` 인자로 지정해 사용합니다(`--help` 참고).

## 결과와 검증

개인 float/INT8 모델과 2026-05-22의 streaming 집계 로그가 남아 있습니다. 로그는 당시 사용한 제공 streaming 모델·환경의 기록이며 개인 학습 모델의 새 정확도 검증으로 해석하지 않습니다. notebook의 저장 출력은 제거했습니다.

공개판에서 양자화 입력을 반올림하고 dtype 범위에 제한해 overflow를 막았고, 모델 기본 경로를 파일 위치 기준으로 수정했습니다. Python 문법과 합성 양자화 경계 검사를 수행했습니다. Pi 실기기·TFLite 추론·재학습은 현재 환경에서 다시 수행하지 않았습니다.
