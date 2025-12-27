"""
Local and remote TCP streaming for audio I/O.
"""
import pyaudio
import asyncio
import logging
import socket
import struct
import time
from dataclasses import dataclass

from faster_whisper import WhisperModel
from piper.voice import PiperVoice
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

    # Remote settings (TCP streaming to/from Raspberry Pi)
    device_ip: str = "192.168.1.131"
    listener_port: int = 4712   # Server listens for mic audio from Pi
    speaker_port: int = 4713    # Server connects to Pi's speaker listener
    
    # Retry settings
    retry_delay: float = 5.0


class LocalAudio:
    """Local audio using PyAudio"""
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.pa = None
        self.input_stream = None
        self.output_stream = None
    
    async def connect(self):
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
    
    def write_sync(self, audio: bytes) -> None:
        """Write audio to output (synchronous)"""
        if self.output_stream is None:
            raise RuntimeError("Output stream not connected")
        self.output_stream.write(audio)
    
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


class RemoteAudio:
    """
    Remote audio via TCP sockets to a Raspberry Pi running pi_audio_streamer.py
    
    Architecture:
    - Microphone: Pi connects to Server:4712, sends audio
                  Server listens on 4712, receives raw PCM
    - Speaker:    Pi listens on 4713
                  Server connects to Pi:4713, sends raw PCM
    """
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        
        # Sockets
        self._mic_server: socket.socket | None = None
        self._mic_conn: socket.socket | None = None
        self._speaker_socket: socket.socket | None = None
        
        # Buffer for incomplete frames
        self._read_buffer = b""
        
        # Frame size in bytes (16-bit mono)
        self._frame_bytes = self.config.input_frames * 2
        
        self._connected = False
    
    async def connect(self):
        """Establish connections for mic and speaker with retry"""
        await asyncio.to_thread(self._connect_sync)
    
    def _connect_sync(self):
        """Synchronous connection setup with retry"""
        # === Microphone: We listen, Pi connects to us ===
        self._connect_mic_with_retry()
        
        # === Speaker: We connect to Pi's listener ===
        self._connect_speaker_with_retry()
        
        self._connected = True
        logger.info("Remote audio connected")
    
    def _connect_mic_with_retry(self):
        """Connect microphone with fixed retry delay"""
        attempt = 0
        
        while True:
            try:
                # Clean up any existing socket
                if self._mic_server:
                    try:
                        self._mic_server.close()
                    except:
                        pass
                
                self._mic_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._mic_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._mic_server.bind(("0.0.0.0", self.config.listener_port))
                self._mic_server.listen(1)
                self._mic_server.settimeout(30.0)  # 30 second timeout per attempt
                
                logger.info(f"Waiting for Pi microphone on port {self.config.listener_port}...")
                
                self._mic_conn, addr = self._mic_server.accept()
                self._mic_conn.settimeout(5.0)
                logger.info(f"Microphone connected from {addr}")
                
                # Success
                return
                
            except socket.timeout:
                attempt += 1
                logger.warning(
                    f"Timeout waiting for Pi microphone (attempt {attempt}). "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                attempt += 1
                logger.error(
                    f"Mic connection error: {e} (attempt {attempt}). "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                time.sleep(self.config.retry_delay)
    
    def _connect_speaker_with_retry(self):
        """Connect speaker with fixed retry delay"""
        attempt = 0
        
        while True:
            try:
                # Clean up any existing socket
                if self._speaker_socket:
                    try:
                        self._speaker_socket.close()
                    except:
                        pass
                
                self._speaker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._speaker_socket.settimeout(10.0)
                
                logger.info(
                    f"Connecting to Pi speaker at "
                    f"{self.config.device_ip}:{self.config.speaker_port}..."
                )
                
                self._speaker_socket.connect((self.config.device_ip, self.config.speaker_port))
                self._speaker_socket.settimeout(None)  # No timeout for sends
                logger.info("Speaker connected")
                
                # Success
                return
                
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                attempt += 1
                logger.warning(
                    f"Speaker connection failed: {e} (attempt {attempt}). "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                attempt += 1
                logger.error(
                    f"Speaker connection error: {e} (attempt {attempt}). "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                time.sleep(self.config.retry_delay)
    
    async def read(self) -> bytes:
        """Read one frame of audio from Pi's microphone"""
        if not self._connected or self._mic_conn is None:
            raise RuntimeError("Remote audio not connected")
        
        return await asyncio.to_thread(self._read_sync)
    
    def _read_sync(self) -> bytes:
        """Synchronous read with buffering and auto-reconnect"""
        while len(self._read_buffer) < self._frame_bytes:
            try:
                if self._mic_conn is None:
                    self._reconnect_mic()
                
                chunk = self._mic_conn.recv(4096)
                if not chunk:
                    logger.warning("Microphone connection closed by Pi, reconnecting...")
                    self._reconnect_mic()
                    continue
                self._read_buffer += chunk
                
            except socket.timeout:
                logger.warning("Mic read timeout, returning silence")
                return b"\x00" * self._frame_bytes
            
            except (ConnectionError, OSError, BrokenPipeError) as e:
                logger.warning(f"Mic connection lost: {e}, reconnecting...")
                self._reconnect_mic()
                continue
        
        # Extract one frame
        frame = self._read_buffer[:self._frame_bytes]
        self._read_buffer = self._read_buffer[self._frame_bytes:]
        return frame
    
    def _reconnect_mic(self):
        """Reconnect microphone with retry"""
        # Close existing connections
        if self._mic_conn:
            try:
                self._mic_conn.close()
            except:
                pass
            self._mic_conn = None
        
        if self._mic_server:
            try:
                self._mic_server.close()
            except:
                pass
            self._mic_server = None
        
        # Clear buffer
        self._read_buffer = b""
        
        # Reconnect
        self._connect_mic_with_retry()
    
    async def write(self, audio: bytes) -> None:
        """Write audio to Pi's speaker"""
        if not self._connected or self._speaker_socket is None:
            raise RuntimeError("Remote audio not connected")
        
        await asyncio.to_thread(self._write_sync, audio)
    
    def _write_sync(self, audio: bytes) -> None:
        """Synchronous write with auto-reconnect"""
        if self._speaker_socket is None:
            logger.warning("Speaker not connected, attempting reconnect...")
            self._reconnect_speaker()
        
        try:
            self._speaker_socket.sendall(audio)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Speaker write failed: {e}, reconnecting...")
            self._reconnect_speaker()
            # Try once more after reconnect
            try:
                if self._speaker_socket:
                    self._speaker_socket.sendall(audio)
            except Exception as e2:
                logger.error(f"Speaker write failed after reconnect: {e2}")
    
    def _reconnect_speaker(self):
        """Reconnect speaker with retry"""
        if self._speaker_socket:
            try:
                self._speaker_socket.close()
            except:
                pass
            self._speaker_socket = None
        
        # Reconnect
        self._connect_speaker_with_retry()
        
        # Play reconnect beep
        self._play_alert_sync()
    
    def _play_alert_sync(self):
        """Play alert sound synchronously"""
        try:
            duration = 0.15
            t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
            freq = 800
            sound = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t) * 0.3
            audio_bytes = (sound * 32767).astype(np.int16).tobytes()
            self._write_sync(audio_bytes)
        except Exception as e:
            logger.warning(f"Failed to play reconnect beep: {e}")
    
    def write_sync(self, audio: bytes) -> None:
        """Write audio synchronously (used by TTS)"""
        if not self._connected:
            return
        self._write_sync(audio)
    
    async def close(self):
        """Close all connections"""
        self._connected = False
        
        for sock in [self._mic_conn, self._mic_server, self._speaker_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        
        self._mic_conn = None
        self._mic_server = None
        self._speaker_socket = None
        self._read_buffer = b""
        
        logger.info("Remote audio closed")


class Audio:
    """Unified audio interface for local or remote audio"""
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._backend: LocalAudio | RemoteAudio | None = None
    
    async def connect(self) -> None:
        if self.config.audio_type == "local":
            self._backend = LocalAudio(self.config)
        elif self.config.audio_type == "remote":
            self._backend = RemoteAudio(self.config)
        else:
            raise ValueError(f"Unsupported audio type: {self.config.audio_type}")
        
        await self._backend.connect()
        
        # Play connection beep
        await self.play_alert()
    
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

    def write_sync(self, audio: bytes) -> None:
        """Write audio to speakers (synchronous)"""
        if self._backend is None:
            raise RuntimeError("Audio not connected")
        self._backend.write_sync(audio)

    async def play_alert(self):
        """Play wake word acknowledgment sound"""
        duration = 0.15
        t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
        
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
    model_paths: list[str] | None = None
    threshold: float = 0.9
    sample_rate: int = 16000
    frame_size_ms: int = 80


class OpenWakeWordDetector:
    """OpenWakeWord implementation."""
    
    def __init__(self, config: OpenWakeWordConfig | None = None):
        self.config = config or OpenWakeWordConfig()
        
        if self.config.model_paths:
            self._model = OWWModel(wakeword_models=self.config.model_paths)
        else:
            self._model = OWWModel()
        
        self._keywords = list(self._model.models.keys())
        logger.info(f"Wake word detector initialized with keywords: {self._keywords}")
    
    def process_frame(self, audio: bytes):
        """Process an audio frame for wake word detection."""
        samples_per_frame = self.config.sample_rate * self.config.frame_size_ms // 1000
        frame = struct.unpack(f"{samples_per_frame}h", audio)
        scores = self._model.predict(frame)
        
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
    aggressiveness: int = 3
    SAMPLE_RATE: int = 16000
    FRAME_SIZE_MS: int = 20
    SAMPLES_PER_80MS = SAMPLE_RATE * 80 // 1000
    SAMPLES_PER_20MS = SAMPLE_RATE * 20 // 1000
    BYTES_PER_20MS = SAMPLES_PER_20MS * 2


class WebRTCVADDetector:
    """WebRTC VAD implementation."""
    
    def __init__(self, config: WebRTCVADConfig | None = None):
        self.config = config or WebRTCVADConfig()
        self._vad = webrtcvad.Vad(self.config.aggressiveness)
    
    def is_speech(self, audio: bytes) -> bool:
        """Check if audio frame contains speech."""
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
    """Whisper ASR implementation."""
    
    def __init__(self, config: OpenWhisperASRConfig | None = None):
        self.config = config or OpenWhisperASRConfig()
        self.asr = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type
        )
    
    def transcribe(self, request_audio):
        """Process audio data with ASR and return the transcribed text."""
        pcm_f32 = np.frombuffer(request_audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.asr.transcribe(pcm_f32, beam_size=5, vad_filter=False)
        request_text = " ".join(s.text for s in segments).strip()
        return request_text


# ========== TTS ==========
@dataclass
class PiperTTSConfig:
    """Configuration for Piper TTS"""
    model_path: str = "./models/lessac_low_model.onnx"


class PiperTTS:
    """Piper TTS with streaming audio output."""
    
    def __init__(self, config: PiperTTSConfig | None = None):
        self.config = config or PiperTTSConfig()
        self._voice = PiperVoice.load(self.config.model_path)
    
    def synthesize_stream(self, text: str):
        """Generator that yields audio chunks as they're synthesized."""
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes
    
    def speak(self, text: str, audio: "Audio") -> None:
        """Speak text through the audio output in streaming mode."""
        if not text.strip():
            return
        
        logger.info(f"TTS speaking: {text[:50]}...")
        
        for audio_bytes in self.synthesize_stream(text):
            audio.write_sync(audio_bytes)
        
        logger.info("TTS finished speaking")