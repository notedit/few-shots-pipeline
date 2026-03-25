# few-shots-pipeline

从长音频中自动提取指定说话人的 few-shot TTS 训练数据，用于 CosyVoice3 等 TTS 模型。

---

## 处理流程

```
source_audio (m4a/mp4/wav)
        │
        ▼
┌─────────────────┐
│ audio_extract   │  PyAV 解码 → full_audio.wav (16kHz mono)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ denoise         │  MossFormer2_SE_48K 全量音频降噪
└────────┬────────┘     → full_audio_denoised.wav (48kHz)
         │
         ▼
┌─────────────────┐
│ vad_segment     │  Silero VAD v6 (pip) 在干净音频上切分
└────────┬────────┘     → segments/seg_XXXXX.wav (16kHz, 250ms padding)
         │
         ▼
┌─────────────────┐
│ speaker_filter  │  wespeaker embedding 余弦相似度
└────────┬────────┘     与 target_audio (reference) 对比
         │  ✓ score ≥ threshold (默认 0.40)
         ▼
┌─────────────────┐
│ vad_trim        │  对每段再次 VAD，精确裁掉首尾静音
└────────┬────────┘     → segments_trimmed/ (保留 200ms padding)
         │
         ▼
┌─────────────────┐
│ speaker_trim    │  1.5s 滑窗逐窗打分，提取最长连续目标说话人区间
└────────┬────────┘     去除段内他声，丢弃 < 1.5s 的碎片
         │
         ▼
┌─────────────────┐
│ asr_dual        │  faster-whisper large-v3 转写
└────────┬────────┘     whisper_only=True 时跳过 Qwen 双模型校验
         │
         ▼
┌─────────────────┐
│ llm_correct     │  (可选) OpenRouter LLM 批量纠错
└────────┬────────┘
         │
         ▼
  manifest.json / manifest_rejected.json
```

---

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **注意**：本项目使用 **PyAV** 替代系统 `ffmpeg` CLI 进行音频解码，无需安装 `ffmpeg` 可执行文件。

---

## 用法

### 完整 pipeline（CLI）

```bash
fspipeline run \
  --video  source_audio.m4a \
  --reference target_audio.mp3 \
  --config config/default.yaml \
  --output output/
```

从指定阶段续跑：

```bash
fspipeline run \
  --video source_audio.m4a \
  --reference target_audio.mp3 \
  --from-stage vad_segment
```

运行单个 stage：

```bash
fspipeline stage audio_extract \
  --input source_audio.m4a \
  --output output/
```

### Python API

```python
from pathlib import Path
from fspipeline.config import PipelineConfig
from fspipeline.pipeline import run_pipeline

cfg = PipelineConfig()
cfg.output_dir = "output"
cfg.speaker_extract.similarity_threshold = 0.40   # 说话人过滤阈值
cfg.asr.whisper_only = True                        # 只用 Whisper，跳过 Qwen
cfg.asr.whisper_model = "large-v3"

ctx = run_pipeline(
    video_path=Path("source_audio.m4a"),
    reference_audio_path=Path("target_audio.mp3"),
    config=cfg,
)

valid = [s for s in ctx.segments if s.is_valid]
print(f"有效片段: {len(valid)}")
for seg in valid[:3]:
    print(seg.id, seg.speaker_score, seg.transcript_final)
```

### 分段逐步运行（Python）

```python
from fspipeline.config import PipelineConfig
from fspipeline.models import PipelineContext
from fspipeline.stages.audio_extract import AudioExtractStage
from fspipeline.stages.vad_segment import VadSegmentStage
from fspipeline.stages.speaker_filter import SpeakerFilterStage
from fspipeline.stages.denoise import DenoiseStage
from fspipeline.stages.asr_dual import DualAsrStage

cfg = PipelineConfig()
cfg.output_dir = "output"

ctx = PipelineContext(
    video_path=Path("source_audio.m4a"),
    reference_audio_path=Path("target_audio.mp3"),
    output_dir=Path("output"),
)

ctx = AudioExtractStage(cfg).execute(ctx)    # → output/full_audio.wav
ctx = VadSegmentStage(cfg).execute(ctx)      # → output/segments/
ctx = SpeakerFilterStage(cfg).execute(ctx)   # 过滤非目标说话人
ctx = DenoiseStage(cfg).execute(ctx)         # 降噪
ctx = DualAsrStage(cfg).execute(ctx)         # Whisper 转写
```

---

## 配置说明（config/default.yaml）

```yaml
speaker_extract:
  embedding_model: "pyannote/wespeaker-voxceleb-resnet34-LM"
  similarity_threshold: 0.40   # 说话人余弦相似度阈值，建议 0.40~0.60
  hf_token: "your_hf_token"

vad:
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 300
  padding_ms: 250

denoise:
  model_source: "speechbrain/sepformer-dns4-16k-enhancement"
  chunk_seconds: 30.0

asr:
  whisper_model: "large-v3"
  whisper_device: "auto"          # auto / cuda / cpu
  whisper_compute_type: "float16"
  language: "auto"
  whisper_only: true              # false = 启用 Qwen3-ASR 双模型交叉校验

output_dir: "output"
```

---

## 输出格式

`output/manifest.json`（有效片段）:

```json
[
  {
    "id": "seg_00008",
    "audio_path": "output/segments_denoised/seg_00008.wav",
    "start_time": 123.4,
    "end_time": 135.6,
    "duration": 12.2,
    "sample_rate": 16000,
    "speaker_score": 0.82,
    "transcript_whisper": "这是转写文本",
    "transcript_final": "这是转写文本",
    "similarity_score": 1.0,
    "is_valid": true
  }
]
```

---

## 测试

```bash
# 快速测试（无需模型下载）
pytest tests/ -m "not slow" -v

# 完整测试（包含 Silero VAD 下载 ~2MB）
pytest tests/ -v
```
