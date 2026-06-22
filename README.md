# omniASR Streaming Server

An OpenAI-compatible ASR (Automatic Speech Recognition) API server powered by Meta's [omniASR](https://github.com/facebookresearch/omnilingual-asr) model. Supports real-time streaming via WebSocket and batch transcription via REST API.

## Features

- **Batch Transcription** - Process thousands of files with resume support (`docker compose run --rm omniASR-batch`)
- **4-bit Quantized** - NF4 quantization with bitsandbytes — runs on 4 GB VRAM, no 29 GB download
- **Batch Inference** - Processes 8 audio files per GPU pass for maximum throughput
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint
- **Real-time Streaming** - WebSocket support for live transcription
- **Long Audio Support** - Automatically handles files **longer than 40 seconds**
- **Multi-device Support** - CUDA (NVIDIA), MPS (Apple Silicon), CPU
- **Voice Agent Ready** - Works with Pipecat, LiveKit, and other frameworks
- **Docker Support** - One-command deployment with GPU support

## Quick Start

### Option 1: Local Installation (Supports CUDA/MPS/CPU)

> **Note:** In your system, audio support requires [libsndfile](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#system-dependencies) (Mac: `brew install libsndfile`; Windows may need an additional [setup](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#installing-on-windows))

```bash
# Clone the repository
git clone https://github.com/ARahim3/omniASR-server.git
cd omniASR-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

Server starts at `http://localhost:8000`

### Option 2: Docker with Quantization (Recommended — No 29 GB Download)

Pre-quantized NF4 checkpoint is available on HuggingFace. No need to download the original 29 GB model.

**Requirements:** NVIDIA GPU with 12+ GB VRAM, Docker Desktop with GPU support.

```bash
# 1. Clone
git clone https://github.com/mufradhossain/omniASR-server.git
cd omniASR-server

# 2. Download the NF4 checkpoint (3.9 GB, ~4 GB VRAM)
mkdir -p checkpoints
# Download omniASR_LLM_7B_v2_nf4_full.pt from:
#   https://huggingface.co/sheikhmufrad/omniASR-LLM-7B-v2-quantized
# Place it in ./checkpoints/

# 3. Start the server
docker compose up -d
```

Server loads in ~32 seconds at `http://localhost:8000`. No `.env` file needed — defaults to NF4 with batch_size=8.

Test it:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

**Performance:**

| Config | Load time | VRAM (idle) | VRAM (peak) | RTF |
|--------|-----------|-------------|-------------|-----|
| NF4 + batch 8 | 32s | 4.2 GB | ~12.3 GB | 0.23 |

## Batch Transcription (Thousands of Files with Resume)

Transcribe large audio corpora — e.g. 350 hours of Bangla audio across 100k+ short files. Processes files in GPU batches of 8 and resumes automatically after crashes.

### Quick Start

```bash
# 1. Put audio files in ./audio/  (wav, mp3, flac, m4a, ogg, opus)
mkdir -p audio transcripts
cp /path/to/bangla_corpus/*.wav audio/

# 2. Run batch transcription
docker compose run --rm omniASR-batch

# 3. Check output
ls transcripts/
# clip_001.txt  clip_002.txt  ...  manifest.json
```

That's it. All defaults are pre-configured: `ben_Beng`, NF4 quant, `batch_size=8`.

### How It Works

```
batch_transcribe.py
  ├── Scans ./audio/ for all audio files (recursive)
  ├── Groups into batches of 8
  ├── Sends each batch in one GPU pass (not one-by-one)
  ├── Writes .txt per file to ./transcripts/
  └── Saves manifest.json for resume
```

### Resume — Kill Anytime, Restart Anytime

The `manifest.json` in the output directory tracks every file's status. If the process crashes, runs out of memory, or you Ctrl-C it:

```bash
# Just re-run the same command
docker compose run --rm omniASR-batch
```

It reads the manifest, skips all completed files, and continues from where it stopped. You lose at most one batch of 8 files (~96 seconds of work).

### Options

```bash
# Custom language
docker compose run --rm omniASR-batch /audio /output --lang ben_Beng

# Different batch size
docker compose run --rm omniASR-batch /audio /output --batch-size 16

# Only specific file extensions
docker compose run --rm omniASR-batch /audio /output --ext wav mp3

# Full options
docker compose run --rm omniASR-batch --help
```

### Expected Throughput

| GPU | Batch Size | ~Files/sec | 350hr (~105k files) |
|-----|-----------|------------|---------------------|
| RTX 5090 / A100 | 8 | ~12-15 | ~2-2.5 hours |
| RTX 4090 | 8 | ~8-10 | ~3-3.5 hours |
| RTX 3060 (12GB) | 8 | ~4-5 | ~6-7 hours |

### Output Structure

```
transcripts/
├── manifest.json                    # resume state (do not delete)
├── clip_001.txt                     # one .txt per audio file
├── clip_002.txt
├── subdir/                          # preserves input folder structure
│   └── clip_003.txt
```

### Custom Audio/Output Paths

The compose file mounts `./audio:/audio` and `./transcripts:/output` by default. To use different paths, edit `docker-compose.yml` under the `omniASR-batch` service:

```yaml
volumes:
  - /data/my_corpus:/audio
  - /data/my_transcripts:/output
```

### Server + Batch Simultaneously

The server (`omniASR-gpu`) and batch tool (`omniASR-batch`) are separate containers. You can run the API server while batch processing, but both load the model into VRAM separately:

```bash
docker compose up -d                                    # server on :8000
docker compose run --rm omniASR-batch                   # batch (needs its own VRAM)
```

If you don't have enough VRAM for both, stop the server first:

```bash
docker compose stop
docker compose run --rm omniASR-batch
docker compose start   # restart server after batch is done
```

### Option 3: Docker (Original — Downloads 29 GB)

```bash
# With NVIDIA GPU
QUANT_ENABLED=false docker compose up -d

# CPU only
docker compose --profile cpu up -d
```

## Usage

### REST API (OpenAI Compatible)

```bash
# Transcribe an audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=omniASR_CTC_300M_v2

# Response
{"text": "Hello world, this is a test."}
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="omniASR_CTC_300M_v2",
        file=f
    )
print(transcript.text)
```

### WebSocket Streaming

```python
import asyncio
import websockets
import json

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/v1/audio/transcriptions") as ws:
        # Wait for ready
        ready = await ws.recv()
        print(f"Server ready: {ready}")

        # Send audio chunks (16kHz, 16-bit PCM, mono)
        with open("audio.raw", "rb") as f:
            while chunk := f.read(3200):  # 100ms chunks
                await ws.send(chunk)

        # Send end signal
        await ws.send(json.dumps({"type": "end"}))

        # Receive transcriptions
        async for message in ws:
            data = json.loads(message)
            print(f"Transcript: {data['text']}")
            if data.get("is_final"):
                break

asyncio.run(stream_audio())
```

### Voice Agent Integration (Pipecat)

```python
from pipecat.services.openai.stt import OpenAISTTService

stt = OpenAISTTService(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/audio/transcriptions/stream` | POST | SSE streaming for long files |
| `/v1/audio/transcriptions` | WebSocket | Real-time streaming |
| `/health` | GET | Health check + connection stats |

### POST /v1/audio/transcriptions

**Request:**
```
Content-Type: multipart/form-data

file: <audio file>
model: omniASR_CTC_300M_v2 (optional)
language: eng_Latn (optional)
response_format: json | text | verbose_json (optional)
```

**Example 1: Basic transcription**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav
```
```json
{"text": "transcribed text here"}
```

**Example 2: With metrics (for developers)**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=verbose_json | jq
```
```json
{
  "text": "transcribed text here",
  "language": "eng_Latn",
  "duration": 1080.854,
  "model": "omniASR_CTC_300M_v2",
  "processing_time": 10.216,
  "rtf": 0.0095
}
```

### POST /v1/audio/transcriptions/stream (SSE)

Stream transcription progress for long audio files using Server-Sent Events.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions/stream \
  -F file=@long_audio.wav
```

**Response (SSE stream):**
```
data: {"text": "first chunk...", "chunk": "1/3", "is_final": false, ...}

data: {"text": "first chunk second chunk...", "chunk": "2/3", "is_final": false, ...}

data: {"text": "first chunk second chunk third chunk", "chunk": "3/3", "is_final": true, ...}

data: [DONE]
```

### WebSocket /v1/audio/transcriptions

**Protocol:**
1. Connect to WebSocket
2. Receive `{"type": "ready", ...}` message
3. Send raw PCM audio (16kHz, 16-bit, mono)
4. Receive transcription updates: `{"text": "...", "is_final": false}`
5. Send `{"type": "end"}` to finish
6. Receive final transcription with `"is_final": true`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CARD` | `omniASR_LLM_7B_v2` | Model to use |
| `DEFAULT_LANG` | `ben_Beng` | Default language (empty for auto-detect) |
| `DEVICE` | auto | `cuda`, `mps`, `cpu`, or auto-detect |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CHUNK_DURATION` | `5.0` | Streaming chunk size (seconds) |
| `VAD_ENABLED` | `true` | Voice activity detection |
| `MAX_CONCURRENT_REQUESTS` | `100` | Max simultaneous REST requests |
| `MAX_WEBSOCKET_CONNECTIONS` | `50` | Max simultaneous WebSocket connections |
| `QUANT_ENABLED` | `true` | Enable NF4 quantization (no 29 GB download) |
| `QUANT_TYPE` | `nf4` | Quantization type (`nf4` = 4-bit) |
| `BATCH_SIZE` | `8` | Files per GPU batch (higher = faster, more VRAM) |

### Using .env file

```bash
cp .env.example .env
# Edit .env with your settings
```

### Available Models

This server supports all models from [Meta's omniASR](https://github.com/facebookresearch/omnilingual-asr), some of these are:

| Model | Type | Parameters | Speed (RTF) | Use Case |
|-------|------|------------|-------------|----------|
| `omniASR_CTC_300M_v2` | CTC | 300M | 0.001 | Fast, good for streaming |
| `omniASR_CTC_1B_v2` | CTC | 1B | 0.003 | Better accuracy |
| `omniASR_CTC_7B_v2` | CTC | 7B | 0.006 | Best accuracy |
| `omniASR_LLM_1B_v2` | LLM | 1B | 0.09 | Language conditioning |
| `omniASR_LLM_7B_v2` | LLM | 7B | 0.10 | Best with context |

*RTF (Real-Time Factor) measured on A100 GPU with batch=1, 30s audio*

> **Note:** CTC models are faster and recommended for streaming. LLM models support language conditioning but are slower. See [omniASR docs](https://github.com/facebookresearch/omnilingual-asr) for full model list.

## Deployment

### Docker Compose

```yaml
# docker-compose.yml is included
# GPU server deployment
docker compose up -d

# Batch transcription (with resume)
docker compose run --rm omniASR-batch

# CPU deployment
docker compose --profile cpu up -d

# Custom configuration
MODEL_CARD=omniASR_CTC_1B_v2 docker compose up -d
```

### Manual Docker Build

```bash
docker build -t omniasr-server .
docker run -d -p 8000:8000 --gpus all omniasr-server
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: omniASR-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: omniASR-server
  template:
    metadata:
      labels:
        app: omniASR-server
    spec:
      containers:
      - name: omniASR
        image: omniASR-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CARD
          value: "omniASR_CTC_300M_v2"
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 omniASR Streaming Server                    │
├──────────────────────┬──────────────────────────────────────┤
│   REST API           │   WebSocket API                      │
│   (OpenAI compat)    │   (real-time streaming)              │
├──────────────────────┴──────────────────────────────────────┤
│                   StreamingTranscriber                      │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Audio    │→ │   Chunked    │→ │   LocalAgreement     │ │
│  │  Buffer   │  │   Inference  │  │   (stable output)    │ │
│  └───────────┘  └──────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              omniASR CTC Model (CUDA/MPS/CPU)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **AudioBuffer** - Ring buffer with overlap chunking for streaming
- **AudioChunker** - VAD-based segmentation for long files
- **LocalAgreement** - Stabilizes streaming output (prevents flickering)
- **ASRModel** - Wrapper with auto device detection and long audio support

## Performance

### Latency (Streaming)

| Device | Chunk Size | Latency |
|--------|------------|---------|
| A100 GPU | 5s | ~50ms |
| RTX 5090 GPU | 5s | ~50ms |
| M4 Pro (MPS) | 5s | ~50ms |
| CPU | 5s | ~5s |

### Throughput

| Device | RTF | 1 hour audio processed in |
|--------|-----|---------------------------|
| A100 GPU | 0.001 | 3.6 seconds |
| RTX 5090 GPU | 0.001 | 3.6 seconds |
| RTX 4050 (laptop) | 0.05 | 3 minutes |
| M4 Pro (MPS) | 0.0095 | ~35 seconds |
| CPU | ~1.0 | 1 hour |

## Troubleshooting

### Model not loading

```bash
# Check if model is downloading
docker compose logs -f

# Manually download model
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; ASRInferencePipeline('omniASR_CTC_300M_v2')"
```

### CUDA out of memory

```bash
# Use smaller model
MODEL_CARD=omniASR_CTC_300M_v2 docker compose up -d

# Or reduce batch size
BATCH_SIZE=1 docker compose up -d
```

### Poor transcription quality in streaming

1. Increase chunk duration: `CHUNK_DURATION=5.0`
2. Force language: `DEFAULT_LANG=eng_Latn`
3. Use larger model: `MODEL_CARD=omniASR_CTC_1B_v2`

### WebSocket connection issues

```bash
# Check server health
curl http://localhost:8000/health

# Test WebSocket
python test_streaming.py websocket
```

## Development

### Project Structure

```
omniASR_server/
├── server.py            # FastAPI app
├── batch_transcribe.py  # Batch transcription CLI (thousands of files + resume)
├── config.py            # Configuration (env vars)
├── model.py             # ASR model wrapper
├── streaming.py         # Streaming transcriber
├── audio_buffer.py      # Audio buffering
├── audio_chunker.py     # Long audio chunking
├── local_agreement.py   # Output stabilization
├── schemas.py           # API schemas
├── test_streaming.py    # Test script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Running Tests

```bash
# Test microphone (local, no server)
python test_streaming.py mic

# Test REST API
python test_streaming.py rest --file audio.wav

# Test WebSocket
python test_streaming.py websocket
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Meta's omniASR](https://github.com/facebookresearch/omnilingual-asr) - The underlying ASR model
- [faster-whisper-server](https://github.com/etalab-ia/faster-whisper-server) - Inspiration for streaming architecture
- [whisper_streaming](https://github.com/ufal/whisper_streaming) - LocalAgreement algorithm

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
