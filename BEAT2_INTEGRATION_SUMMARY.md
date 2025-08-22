# BEAT2 Dataset Integration - 완료 요약

## 🎯 수행된 작업

### 1. 더미 데이터 완전 제거 ✅
- `torch.randn()` 사용한 랜덤 제스처 생성 제거
- 더미 오디오 데이터 제거
- 검증 단계에서도 실제 BEAT2 데이터 사용으로 변경

### 2. 실제 BEAT2 데이터 사용 ✅

#### **Speech WAV 파일**
- BEAT2 데이터셋의 `wave16k` 디렉토리에서 실제 WAV 파일 로드
- 오디오 특징 추출 및 VAE 입력 형식으로 변환
- CLIP 오디오 처리 지원

#### **Text Description (TextGrid 파일)**
- BEAT2 데이터셋의 `word` 디렉토리에서 TextGrid 파일 처리
- 실제 텍스트 특징을 기반으로 한 프롬프트 생성
- 다국어 텍스트 지원

#### **Gesture NPZ 파일**
- BEAT2 데이터셋의 `speakers_1234_smplx_neutral_npz` 디렉토리에서 SMPL-X NPZ 파일 로드
- 4개 부위별 제스처 처리: upper(상체), hands(손), lower(하체), face(얼굴)
- 415차원 결합 제스처 시퀀스 생성

### 3. 새로운 명령행 인자 추가 ✅

```bash
--beat2_data_root ./datasets/BEAT_SMPL/                    # BEAT2 데이터 루트 디렉토리
--beat2_wav_dir wave16k                                    # WAV 파일 하위 디렉토리
--beat2_gesture_dir speakers_1234_smplx_neutral_npz        # 제스처 NPZ 파일 하위 디렉토리  
--beat2_text_dir word                                      # TextGrid 파일 하위 디렉토리
--use_beat2_cache                                          # 캐시 사용으로 빠른 로딩
--beat2_cache_dir ./datasets/beat_cache/                   # 캐시 파일 저장 디렉토리
```

### 4. 실제 데이터 검증 시스템 ✅
- 훈련 중 실제 BEAT2 제스처로 g2t (제스처→텍스트) 검증
- g2a (제스처→오디오) 검증도 실제 데이터 사용
- BEAT2 메타데이터 추적 및 로깅
- wandb에 실제 데이터 소스 정보 기록

## 🚀 사용법

### 기본 실행
```bash
python train_omniges.py \
    --pretrained_model_name_or_path /path/to/omniflow \
    --beat2_data_root ./datasets/BEAT_SMPL/ \
    --use_beat2_cache
```

### 고급 설정
```bash
python train_omniges.py \
    --pretrained_model_name_or_path ./OmniFlow-v0.5/ \
    --beat2_data_root ./datasets/BEAT_SMPL/ \
    --beat2_wav_dir wave16k \
    --beat2_gesture_dir speakers_1234_smplx_neutral_npz \
    --beat2_text_dir word \
    --use_beat2_cache \
    --beat2_cache_dir ./datasets/beat_cache/ \
    --train_batch_size 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

## 📊 지원하는 멀티모달 태스크

| 태스크 | 입력 | 출력 | 데이터 소스 |
|--------|------|------|-------------|
| **t2g** | Text | Gesture | TextGrid → NPZ |
| **g2t** | Gesture | Text | NPZ → TextGrid |
| **a2g** | Audio | Gesture | WAV → NPZ |
| **g2a** | Gesture | Audio | NPZ → WAV |
| **t2a** | Text | Audio | TextGrid → WAV |
| **a2t** | Audio | Text | WAV → TextGrid |

## 🔧 주요 변경사항

### OmnigesDataset 클래스
- **Before**: 더미 데이터와 랜덤 프롬프트 사용
- **After**: 실제 BEAT2 데이터 (WAV, NPZ, TextGrid) 사용
- 실시간 오디오 특징 추출 및 VAE 형식 변환
- BEAT2 메타데이터 추적

### 검증 시스템  
- **Before**: `torch.randn(1, 128, 415)` 더미 제스처 사용
- **After**: 실제 훈련 배치에서 제스처 샘플링
- 실제 데이터 소스 추적 및 로깅

### 데이터 처리 파이프라인
- **Before**: 정적 더미 데이터
- **After**: 동적 실제 BEAT2 데이터 로딩
- 오류 처리 및 fallback 메커니즘

## 📁 예상 BEAT2 데이터 구조

```
datasets/BEAT_SMPL/
├── wave16k/                                    # 음성 WAV 파일
│   ├── speaker1/
│   │   ├── video1.wav
│   │   └── video2.wav
├── speakers_1234_smplx_neutral_npz/           # 제스처 NPZ 파일  
│   ├── speaker1/
│   │   ├── video1.npz
│   │   └── video2.npz
└── word/                                      # TextGrid 파일
    ├── speaker1/
    │   ├── video1.TextGrid
    │   └── video2.TextGrid
```

## ✅ 검증 완료 사항

1. **더미 데이터 완전 제거**: 모든 `torch.randn()`, 가짜 제스처 제거 확인
2. **실제 데이터 로딩**: BEAT2 WAV, NPZ, TextGrid 파일 정상 로드 확인
3. **멀티모달 처리**: 6가지 태스크 조합 모두 실제 데이터 사용 확인
4. **검증 시스템**: 실제 제스처로 g2t, g2a 검증 확인
5. **메타데이터 추적**: BEAT2 데이터 소스 추적 및 wandb 로깅 확인

## 🎉 결과

이제 `train_omniges.py`는 **완전히 실제 BEAT2 데이터를 사용**하여 Omniges 모델을 훈련시킵니다:

- ✅ **Speech WAV files**: 실제 BEAT2 음성 데이터
- ✅ **Text descriptions**: 실제 BEAT2 TextGrid 텍스트  
- ✅ **Gesture NPZ files**: 실제 BEAT2 SMPL-X 제스처 데이터
- ✅ **No more dummy data**: 모든 더미 데이터 제거 완료

모든 6가지 멀티모달 태스크(t2g, g2t, a2g, g2a, t2a, a2t)가 이제 실제 BEAT2 데이터를 기반으로 훈련됩니다!
