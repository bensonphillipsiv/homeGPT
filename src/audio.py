"""
Local and PulseAudio network streaming for audio I/O.
"""
import pyaudio
import asyncio
import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import AsyncIterator
from faster_whisper import WhisperModel

import numpy as np
import webrtcvad
from openwakeword.model import Model as OWWModel

logger = logging.getLogger(__name__)

# ========== Raw Audio Classes ==========
@dataclass
class AudioConfig:
    audio_type: str = "local"  # "local" or "remote"
    sample_rate: int = 16000
    channels: int = 1
    input_frames: int = 1280  # 80ms at 16kHz

    # Remote settings (for PulseAudio)
    device_ip: str = "192.168.1.132"
    listener_port: int = 4712
    speaker_port: int = 4713


class LocalAudio:
    """Local audio using PyAudio"""
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.pa = None
        self.input_stream = None
        self.output_stream = None
    
    async def connect(self):
        # PyAudio is blocking, run in thread
        def _connect():
            self.pa = pyaudio.PyAudio()
            
            self.input_stream = self.pa.open(
                input=True,
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                frames_per_buffer=self.config.input_frames,
            )
            
            self.output_stream = self.pa.open(
                output=True,
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
            )
        
        await asyncio.to_thread(_connect)
        logger.info("Local audio connected")
    
    async def read(self) -> bytes:
        """Read one frame of audio"""
        if self.input_stream is None:
            raise RuntimeError("Input stream not connected")
        
        # read() is blocking, run in thread
        return await asyncio.to_thread(
            self.input_stream.read, 
            self.config.input_frames,
            exception_on_overflow=False
        )
    
    
    async def write(self, audio: bytes) -> None:
        """Write audio to output"""
        if self.output_stream is None:
            raise RuntimeError("Output stream not connected")
        
        await asyncio.to_thread(self.output_stream.write, audio)
    
    async def close(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.pa:
            self.pa.terminate()
        logger.info("Local audio closed")


class Audio:
    """Unified audio interface for local or remote audio"""
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._backend = None
    
    async def connect(self) -> None:
        if self.config.audio_type == "local":
            self._backend = LocalAudio(self.config)
        elif self.config.audio_type == "remote":
            # TODO: Add PulseAudio network backend
            raise NotImplementedError("Remote audio not yet implemented")
        else:
            raise ValueError(f"Unsupported audio type: {self.config.audio_type}")
        
        await self._backend.connect()
    
    async def read(self) -> bytes:
        """Read one frame of audio (80ms)"""
        if self._backend is None:
            raise RuntimeError("Audio not connected")
        return await self._backend.read()
    
    async def write(self, audio: bytes) -> None:
        """Write audio to speakers"""
        if self._backend is None:
            raise RuntimeError("Audio not connected")
        await self._backend.write(audio)

    async def play_alert(self):
        """Play wake word acknowledgment sound"""
        import numpy as np
        
        duration = 0.15
        t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
        
        # Simple bell sound
        freq = 800
        sound = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t) * 0.3
        
        audio_bytes = (sound * 32767).astype(np.int16).tobytes()
        await self.write(audio_bytes)
    
    async def close(self) -> None:
        if self._backend:
            await self._backend.close()


# ========== Wake Word ==========
@dataclass
class OpenWakeWordConfig:
    """Configuration for OpenWakeWord detector"""
    # Model settings (None uses default bundled models)
    model_paths: list[str] | None = None
    
    # Detection settings
    threshold: float = 0.9
    
    # Audio settings
    sample_rate: int = 16000
    frame_size_ms: int = 80  # OpenWakeWord needs 80ms frames


class OpenWakeWordDetector:
    """
    OpenWakeWord implementation.
    """
    
    def __init__(self, config: OpenWakeWordConfig | None = None):
        self.config = config or OpenWakeWordConfig()
        
        if self.config.model_paths:
            self._model = OWWModel(wakeword_models=self.config.model_paths)
        else:
            self._model = OWWModel()
        
        # Get available keywords
        self._keywords = list(self._model.models.keys())
        logger.info(f"Wake word detector initialized with keywords: {self._keywords}")
    
    
    def process_frame(self, audio: bytes):
        """
        Process an audio frame for wake word detection.
        
        Args:
            audio: Raw PCM audio (80ms frame, 16-bit signed, mono)
            
        Returns:
            WakeWordEvent if detected, None otherwise
        """
        # Convert bytes to samples
        samples_per_frame = self.config.sample_rate * self.config.frame_size_ms // 1000
        frame = struct.unpack(f"{samples_per_frame}h", audio)
        
        # Get predictions
        scores = self._model.predict(frame)
        
        # Check each keyword
        for keyword, score in scores.items():
            if score > self.config.threshold:
                logger.debug(f"Wake word '{keyword}' detected with score {score:.3f}")
                return keyword, float(score), time.time()
        
        return None
    
    def reset(self) -> None:
        """Reset detector state"""
        self._model.reset()


# ========== VAD ==========
@dataclass
class WebRTCVADConfig:
    """Configuration for WebRTC VAD"""
    # Aggressiveness mode (0-3, higher = more aggressive filtering)
    aggressiveness: int = 3
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    FRAME_SIZE_MS: int = 20  # WebRTC VAD needs 10, 20, or 30ms frames
    SAMPLES_PER_80MS = SAMPLE_RATE * 80 // 1000  # 1280 samples
    SAMPLES_PER_20MS = SAMPLE_RATE * 20 // 1000  # 320 samples
    BYTES_PER_20MS = SAMPLES_PER_20MS * 2  # 640 bytes (16-bit audio)


class WebRTCVADDetector:
    """
    Adapter: WebRTC VAD implementation of VADDetector.
    """
    
    def __init__(self, config: WebRTCVADConfig | None = None):
        self.config = config or WebRTCVADConfig()
        self._vad = webrtcvad.Vad(self.config.aggressiveness)
    
    def is_speech(self, audio: bytes) -> bool:
        """
        Check if audio frame contains speech.
        
        Args:
            audio: Raw PCM audio (20ms frame, 16-bit signed, mono)
            
        Returns:
            True if speech detected
        """

        for subframe in self._split_20ms_frames(audio):
            if not self._vad.is_speech(subframe, self.config.SAMPLE_RATE):
                return False
    
        return True
    
    def _split_20ms_frames(self, frame_80ms: bytes):
        """Split an 80ms frame into four 20ms frames for VAD"""
        for i in range(4):
            start = i * self.config.BYTES_PER_20MS
            end = start + self.config.BYTES_PER_20MS
            yield frame_80ms[start:end]

# ========== ASR ==========
@dataclass
class OpenWhisperASRConfig:
    """Configuration for OpenWhisper ASR"""
    model_size: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"


class OpenWhisperASR:
    """
    Adapter: WebRTC VAD implementation of VADDetector.
    """
    
    def __init__(self, config: OpenWhisperASRConfig | None = None):
        self.config = config or OpenWhisperASRConfig()
        self.asr = WhisperModel(self.config.model_size, device=self.config.device, compute_type=self.config.compute_type)
    
    def transcribe(self, request_audio):
        """
        Process audio data with ASR and return the transcribed text.
        """
        pcm_f32 = np.frombuffer(request_audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.asr.transcribe(pcm_f32, beam_size=5, vad_filter=False)
        request_text = " ".join(s.text for s in segments).strip()

        return request_text


# ========== TTS ==========