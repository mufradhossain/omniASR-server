"""omniASR model wrapper with device detection and caching."""

import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from bitsandbytes.nn import Linear8bitLt, LinearNF4

from config import config


@dataclass
class TranscribeResult:
    """Result from transcription."""
    text: str
    duration: float      # audio duration in seconds
    latency: float       # inference time in seconds
    rtf: float           # real-time factor


def get_device() -> str:
    """Get best available device (can be overridden via config/env)."""
    # Check if device is explicitly set in config
    if config.model.device:
        return config.model.device

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ASRModel:
    """
    Wrapper for omniASR pipeline with lazy loading and device management.
    """

    _instance: "ASRModel | None" = None

    def __init__(
        self,
        model_card: str = None,
        device: str = None,
        lang: str = None,
    ):
        self.model_card = model_card or config.model.model_card
        self.device = device or get_device()
        self.lang = lang if lang is not None else config.model.default_lang

        self._pipeline = None
        self._load_time: float = 0

    @classmethod
    def get_instance(cls) -> "ASRModel":
        """Get singleton instance of ASRModel."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def load(self) -> None:
        """Load the model (lazy loading)."""
        if self._pipeline is not None:
            return

        if config.quant.enabled:
            self._load_quantized()
        else:
            self._load_standard()

    def _load_standard(self) -> None:
        """Standard pipeline load (no quantization)."""
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        print(f"Loading ASR model '{self.model_card}' on {self.device}...")
        start = time.perf_counter()
        self._pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=self.device,
        )
        self._load_time = time.perf_counter() - start
        print(f"Model loaded in {self._load_time:.2f}s")

    def _count_fs2_linears(self, module) -> int:
        """Count fairseq2 Linear layers in module tree."""
        try:
            from fairseq2.nn import Linear as Fs2Linear
        except ImportError:
            from fairseq2.nn.projection import Linear as Fs2Linear

        count = 0
        for _, child in module.named_children():
            if isinstance(child, Fs2Linear):
                count += 1
            else:
                count += self._count_fs2_linears(child)
        return count

    def _get_quant_layer_class(self):
        """Return the bnb quantized Linear class based on config."""
        qtype = config.quant.quant_type.lower()
        if qtype == "int8":
            return Linear8bitLt, {"has_fp16_weights": False}
        elif qtype in ("nf4", "int4", "fp4"):
            return LinearNF4, {}
        else:
            raise ValueError(f"Unsupported quant_type: {qtype}. Use 'int8' or 'nf4'.")

    def _replace_fs2_linears_streaming(self, module, layer_idx: list, total: int, device: str) -> None:
        """
        Replace fairseq2.nn.Linear with bnb quantized layers, one at a time.

        For each layer:
            1. Clone weight from mmap to RAM (one layer only, ~50-200 MB)
            2. Create quantized layer, load weight (quantization converts to int8/nf4)
            3. Move quantized layer to GPU (small VRAM footprint)
            4. Free CPU weight (mmap reclaims, RAM stays low)
        """
        import torch
        import torch.nn as nn
        try:
            from fairseq2.nn import Linear as Fs2Linear
        except ImportError:
            from fairseq2.nn.projection import Linear as Fs2Linear

        QuantClass, extra_kwargs = self._get_quant_layer_class()

        for name, child in list(module.named_children()):
            if isinstance(child, Fs2Linear):
                layer_idx[0] += 1
                idx = layer_idx[0]

                in_dim = child.input_dim
                out_dim = child.output_dim
                has_bias = child.bias is not None

                # Step 1: Load this single weight to RAM from mmap
                w = child.weight.data.clone()
                b = child.bias.data.clone() if has_bias else None

                # Step 2: Create standard nn.Linear, then quantized layer
                tmp = nn.Linear(in_dim, out_dim, bias=has_bias)
                tmp.weight = nn.Parameter(w)
                if has_bias:
                    tmp.bias = nn.Parameter(b)

                quant_lin = QuantClass(in_dim, out_dim, bias=has_bias, **extra_kwargs)
                quant_lin.load_state_dict(tmp.state_dict())
                del tmp, w
                if b is not None:
                    del b

                # Step 3: Move to GPU (triggers .cuda() quantization for nf4)
                quant_lin = quant_lin.to(device)

                # Step 4: Free original CPU weight reference
                child.weight = nn.Parameter(torch.empty(0))
                if has_bias:
                    child.bias = None

                setattr(module, name, quant_lin)

                # Progress logging
                if idx % 25 == 0 or idx == total:
                    vram = torch.cuda.memory_allocated() / 1e9 if 'cuda' in str(device) else 0
                    pct = 100 * idx / total
                    print(f"    [{idx}/{total}] {pct:.0f}% — VRAM: {vram:.2f} GB", flush=True)

                if idx % 50 == 0:
                    torch.cuda.empty_cache()
            else:
                self._replace_fs2_linears_streaming(child, layer_idx, total, device)

    def _restore_quant_to_device(self, model) -> None:
        """Move all model weights to GPU after loading from pickle.

        bitsandbytes Params4bit/Int8Params lose their CUDA context during
        torch.save()/torch.load(). This re-establishes the device mapping
        for quantized layers and moves regular layers normally.
        """
        device = self.device
        fixed = 0
        # First move non-quantized layers normally
        for module in model.modules():
            if not isinstance(module, (LinearNF4, Linear8bitLt)):
                for param in module.parameters(recurse=False):
                    param.data = param.data.to(device)
                for buf in module.buffers(recurse=False):
                    buf.data = buf.data.to(device)

        # Then fix quantized layers individually
        for module in model.modules():
            if isinstance(module, (LinearNF4, Linear8bitLt)):
                w = module.weight
                w.data = w.data.to(device)
                if hasattr(w, 'quant_state') and w.quant_state is not None:
                    w.quant_state = w.quant_state.to(device)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(device)
                # Linear8bitLt has a separate state object with CB/SCB tensors
                if hasattr(module, 'state') and module.state is not None:
                    for attr in ['CB', 'SCB', 'CxB', 'SB']:
                        tensor = getattr(module.state, attr, None)
                        if tensor is not None:
                            setattr(module.state, attr, tensor.to(device))
                fixed += 1
                if fixed % 200 == 0:
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    def _load_quantized(self) -> None:
        """
        Quantized model load with checkpoint caching.

        Supports int8 (Linear8bitLt) and nf4 (LinearNF4) via QUANT_TYPE config.

        First run (no checkpoint):
            1. mmap model on CPU (weights on disk, ~0 RAM)
            2. Stream each Linear: RAM → quantize → GPU → free
            3. Save full-module checkpoint (architecture + quantized weights)
        Subsequent runs (checkpoint exists):
            Load checkpoint directly — no model download, no fairseq2, no re-quantization
        """
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, load_tokenizer
        from fairseq2.models import load_model as fairseq2_load_model
        from fairseq2.device import Device
        from pathlib import Path

        qtype = config.quant.quant_type.lower()
        model_slug = self.model_card.replace('/', '_')
        checkpoint_dir = Path(config.quant.checkpoint_path)
        checkpoint = checkpoint_dir / f"{model_slug}_{qtype}.pt"
        full_pickle = checkpoint_dir / f"{model_slug}_{qtype}_full.pt"

        print(f"[1/4] Loading tokenizer for '{self.model_card}'...", flush=True)
        tokenizer = load_tokenizer(self.model_card)

        start = time.perf_counter()

        if full_pickle.exists():
            # Ultra-fast path: load full-module pickle directly — no fairseq2 at all
            pickle_size = full_pickle.stat().st_size / 1e9
            print(f"[2/4] Loading full-module pickle ({pickle_size:.1f} GB)...", flush=True)
            model = torch.load(full_pickle, weights_only=False, map_location='cpu')

            # Fix quantized layer device states (lost during pickle)
            print(f"[3/4] Restoring quantized states to GPU...", flush=True)
            self._restore_quant_to_device(model)

            self._load_time = time.perf_counter() - start
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"Done! Model ready in {self._load_time:.1f}s, VRAM: {vram:.2f} GB", flush=True)
            self._pipeline = ASRInferencePipeline(
                model_card=None,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            return

        if checkpoint.exists():
            # Fast path: load state_dict checkpoint
            # Still needs model architecture from fairseq2, but weights come from checkpoint
            checkpoint_size = checkpoint.stat().st_size / 1e9
            print(f"[2/4] Loading model architecture from cache...", flush=True)
            model = fairseq2_load_model(self.model_card, device=Device('cpu'), dtype=torch.bfloat16, mmap=True)

            total = self._count_fs2_linears(model)
            print(f"[3/4] Replacing {total} layers + loading {qtype} weights...", flush=True)
            self._replace_fs2_linears_streaming(model, [0], total, self.device)
            state = torch.load(checkpoint, weights_only=False, mmap=True)
            model.load_state_dict(state, strict=False)
            del state
            model = model.to(self.device)

        else:
            # Slow path: quantize from scratch (first time only)
            print(f"[2/4] No checkpoint found. Loading model on CPU (mmap)...", flush=True)
            model = fairseq2_load_model(
                self.model_card, device=Device('cpu'), dtype=torch.bfloat16, mmap=True
            )

            total = self._count_fs2_linears(model)
            print(f"[3/4] Quantizing {total} layers to {qtype} (streaming to GPU)...", flush=True)
            print(f"      Each layer: RAM load → {qtype} convert → GPU move → RAM free", flush=True)
            self._replace_fs2_linears_streaming(model, [0], total, self.device)
            model = model.to(self.device)

            print(f"[4/4] Saving state_dict checkpoint...", flush=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            checkpoint_size = checkpoint.stat().st_size / 1e9
            print(f"      Saved: {checkpoint.name} ({checkpoint_size:.1f} GB)", flush=True)

            # Save full-module pickle for zero-download future loads
            print(f"      Saving full-module pickle...", flush=True)
            model_cpu = model.cpu()
            torch.cuda.empty_cache()
            torch.save(model_cpu, full_pickle)
            del model_cpu
            full_size = full_pickle.stat().st_size / 1e9
            print(f"      Saved: {full_pickle.name} ({full_size:.1f} GB)", flush=True)
            model = model.to(self.device)

        self._pipeline = ASRInferencePipeline(
            model_card=None,
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )
        self._load_time = time.perf_counter() - start

        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"Done! Model ready in {self._load_time:.1f}s, VRAM: {vram:.2f} GB", flush=True)

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._pipeline is None:
            self.load()

    def transcribe_file(self, audio_path: str | Path, lang: str = None) -> TranscribeResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            lang: Language code (e.g., "eng_Latn") or None for auto-detect

        Returns:
            TranscribeResult with text and timing info
        """
        self.ensure_loaded()

        # Get audio duration
        import soundfile as sf
        info = sf.info(str(audio_path))
        duration = info.duration

        # Transcribe
        lang_param = lang if lang is not None else self.lang
        start = time.perf_counter()
        result = self._pipeline.transcribe(
            [str(audio_path)],
            lang=[lang_param] if lang_param else None,
            batch_size=config.model.batch_size,
        )
        latency = time.perf_counter() - start

        text = result[0] if result else ""

        return TranscribeResult(
            text=text,
            duration=duration,
            latency=latency,
            rtf=latency / duration if duration > 0 else 0,
        )

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        lang: str = None,
    ) -> TranscribeResult:
        """
        Transcribe audio array.

        Args:
            audio: Audio samples (numpy array)
            sample_rate: Sample rate (default from config)
            lang: Language code or None for auto-detect

        Returns:
            TranscribeResult with text and timing info
        """
        self.ensure_loaded()

        sample_rate = sample_rate or config.audio.sample_rate
        duration = len(audio) / sample_rate

        # Save to temp file (omniASR requires file path)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), sample_rate)
            temp_path = f.name

        try:
            # Transcribe
            lang_param = lang if lang is not None else self.lang
            start = time.perf_counter()
            result = self._pipeline.transcribe(
                [temp_path],
                lang=[lang_param] if lang_param else None,
                batch_size=config.model.batch_size,
            )
            latency = time.perf_counter() - start

            text = result[0] if result else ""

            return TranscribeResult(
                text=text,
                duration=duration,
                latency=latency,
                rtf=latency / duration if duration > 0 else 0,
            )
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_long_file(
        self,
        audio_path: str | Path,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ) -> TranscribeResult:
        """
        Transcribe a long audio file by chunking.

        Automatically handles files longer than the model's limit (40s)
        by splitting at natural boundaries (silences) and concatenating results.

        Args:
            audio_path: Path to audio file
            lang: Language code or None for auto-detect
            max_chunk_duration: Maximum chunk duration (default 35s for safety)

        Returns:
            TranscribeResult with concatenated text and total timing info
        """
        self.ensure_loaded()

        import soundfile as sf
        from audio_chunker import AudioChunker, ChunkerConfig, TranscriptionSegment, merge_transcriptions

        # Load audio file
        audio, sample_rate = sf.read(str(audio_path), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        total_duration = len(audio) / sample_rate

        # If short enough, use regular transcription
        if total_duration <= max_chunk_duration:
            return self.transcribe_file(audio_path, lang=lang)

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)

        print(f"Long audio ({total_duration:.1f}s) split into {len(segments)} chunks")

        # Write all chunks to temp files and batch-transcribe
        import tempfile
        import os
        temp_files = []
        batch_size = config.model.batch_size

        for i, segment in enumerate(segments):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, segment.audio.astype(np.float32), segment.sample_rate)
            temp_files.append(tmp.name)

        try:
            lang_param = lang if lang is not None else self.lang
            print(f"  Batch transcribing {len(temp_files)} chunks (batch_size={batch_size})...")

            start = time.perf_counter()
            results = self._pipeline.transcribe(
                temp_files,
                lang=[lang_param] * len(temp_files) if lang_param else None,
                batch_size=batch_size,
            )
            total_latency = time.perf_counter() - start

            transcription_segments = []
            for i, (segment, text) in enumerate(zip(segments, results)):
                text = text if text else ""
                transcription_segments.append(TranscriptionSegment(
                    text=text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                ))
                print(f"  Chunk {i+1}/{len(segments)} done")

        finally:
            for f in temp_files:
                os.unlink(f)

        # Merge transcriptions
        merged_text = merge_transcriptions(transcription_segments)

        return TranscribeResult(
            text=merged_text,
            duration=total_duration,
            latency=total_latency,
            rtf=total_latency / total_duration if total_duration > 0 else 0,
        )

    def transcribe_long_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ) -> TranscribeResult:
        """
        Transcribe a long audio array by chunking.

        Args:
            audio: Audio samples (numpy array)
            sample_rate: Sample rate (default from config)
            lang: Language code or None for auto-detect
            max_chunk_duration: Maximum chunk duration

        Returns:
            TranscribeResult with concatenated text
        """
        self.ensure_loaded()

        from audio_chunker import AudioChunker, ChunkerConfig, TranscriptionSegment, merge_transcriptions

        sample_rate = sample_rate or config.audio.sample_rate

        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        total_duration = len(audio) / sample_rate

        # If short enough, use regular transcription
        if total_duration <= max_chunk_duration:
            return self.transcribe_audio(audio, sample_rate=sample_rate, lang=lang)

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)

        # Transcribe each chunk
        transcription_segments = []
        total_latency = 0.0

        for segment in segments:
            result = self.transcribe_audio(
                segment.audio,
                sample_rate=segment.sample_rate,
                lang=lang,
            )
            total_latency += result.latency

            transcription_segments.append(TranscriptionSegment(
                text=result.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            ))

        # Merge transcriptions
        merged_text = merge_transcriptions(transcription_segments)

        return TranscribeResult(
            text=merged_text,
            duration=total_duration,
            latency=total_latency,
            rtf=total_latency / total_duration if total_duration > 0 else 0,
        )

    def transcribe_long_file_streaming(
        self,
        audio_path: str | Path,
        lang: str = None,
        max_chunk_duration: float = 35.0,
    ):
        """
        Generator that yields transcription results as chunks are processed.

        Useful for SSE streaming - yields progress for long files.

        Yields:
            dict with keys: text, chunk_index, total_chunks, is_final, duration, processing_time
        """
        self.ensure_loaded()

        import soundfile as sf
        from audio_chunker import AudioChunker, ChunkerConfig

        # Load audio file
        audio, sample_rate = sf.read(str(audio_path), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        total_duration = len(audio) / sample_rate

        # If short enough, yield single result
        if total_duration <= max_chunk_duration:
            result = self.transcribe_file(audio_path, lang=lang)
            yield {
                "text": result.text,
                "chunk_index": 1,
                "total_chunks": 1,
                "is_final": True,
                "duration": result.duration,
                "processing_time": result.latency,
                "rtf": result.rtf,
            }
            return

        # Chunk the audio
        chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_chunk_duration))
        segments = chunker.chunk_audio(audio, sample_rate)
        total_chunks = len(segments)

        accumulated_text = []
        total_processing_time = 0.0

        for i, segment in enumerate(segments):
            result = self.transcribe_audio(
                segment.audio,
                sample_rate=segment.sample_rate,
                lang=lang,
            )
            total_processing_time += result.latency

            if result.text.strip():
                accumulated_text.append(result.text.strip())

            is_final = (i == total_chunks - 1)

            yield {
                "text": " ".join(accumulated_text),
                "chunk_index": i + 1,
                "total_chunks": total_chunks,
                "is_final": is_final,
                "duration": segment.end_time,
                "processing_time": total_processing_time,
                "rtf": total_processing_time / segment.end_time if segment.end_time > 0 else 0,
            }

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipeline is not None

    @property
    def load_time(self) -> float:
        """Get model load time."""
        return self._load_time

    @property
    def device_name(self) -> str:
        """Get device name."""
        return self.device
