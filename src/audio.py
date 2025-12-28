"""
Local and WebSocket streaming for audio I/O.
"""
import pyaudio
import asyncio
import logging
import struct
import time
from dataclasses import dataclass

from faster_whisper import WhisperModel
from piper.voice import PiperVoice
import numpy as np
import webrtcvad
from openwakeword.model import Model as OWWModel
import websockets
from websockets.server import serve

logger = logging.getLogger(__name__)


# ========== Raw Audio Classes ==========
@dataclass
class AudioConfig:
    audio_type: str = "local"  # "local" or "remote"
    sample_rate: int = 16000
    channels: int = 1
    input_frames: int = 1280  # 80ms at 16kHz

    # WebSocket settings (used when audio_type="remote")
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10


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
    WebSocket-based remote audio I/O for Raspberry Pi.
    
    Single WebSocket connection handles both mic input and speaker output.
    Messages are prefixed with a type byte:
      - 0x01: Mic audio (Pi -> Server)
      - 0x02: Speaker audio (Server -> Pi)
      - 0x03: Control message (JSON)
    """
    
    MSG_MIC = 0x01
    MSG_SPEAKER = 0x02
    MSG_CONTROL = 0x03
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._server = None
        self._client_ws = None
        self._connected = asyncio.Event()
        self._read_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._frame_bytes = self.config.input_frames * 2  # 16-bit audio
        self._read_buffer = b""
        self._running = False
        self._handler_task = None
        self._loop = None
    
    async def connect(self):
        """Start WebSocket server and wait for client connection."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        
        self._server = await serve(
            self._handle_client,
            self.config.ws_host,
            self.config.ws_port,
            ping_interval=self.config.ws_ping_interval,
            ping_timeout=self.config.ws_ping_timeout,
        )
        
        logger.info(f"WebSocket audio server listening on ws://{self.config.ws_host}:{self.config.ws_port}")
        logger.info("Waiting for Pi to connect...")
        
        # Wait for client to connect
        await self._connected.wait()
        
        logger.info("WebSocket audio connected")
    
    async def _handle_client(self, websocket):
        """Handle incoming WebSocket connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected from {client_addr}")
        
        # Only allow one client at a time
        if self._client_ws is not None:
            logger.warning(f"Rejecting connection from {client_addr} - already have a client")
            await websocket.close(1008, "Only one client allowed")
            return
        
        self._client_ws = websocket
        self._connected.set()
        
        try:
            async for message in websocket:
                if isinstance(message, bytes) and len(message) > 0:
                    msg_type = message[0]
                    payload = message[1:]
                    
                    if msg_type == self.MSG_MIC:
                        # Queue mic audio for reading
                        await self._read_queue.put(payload)
                    elif msg_type == self.MSG_CONTROL:
                        # Handle control messages (future use)
                        logger.debug(f"Control message: {payload.decode()}")
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Client disconnected: {e}")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self._client_ws = None
            self._connected.clear()
            # Clear any pending audio
            while not self._read_queue.empty():
                try:
                    self._read_queue.get_nowait()
                except:
                    pass
            self._read_buffer = b""
            logger.info(f"Client {client_addr} cleaned up, waiting for reconnection...")
            
            # Wait for reconnection if still running
            if self._running:
                logger.info("Waiting for Pi to reconnect...")
                await self._connected.wait()
    
    async def read(self) -> bytes:
        """Read one frame of audio from the client."""
        # Wait for connection if not connected
        if self._client_ws is None:
            await self._connected.wait()
        
        # Buffer until we have a full frame
        while len(self._read_buffer) < self._frame_bytes:
            try:
                chunk = await self._read_queue.get()
                self._read_buffer += chunk
            except Exception as e:
                logger.warning(f"Read queue error: {e}")
                await asyncio.sleep(0.01)
        
        # Extract one frame
        frame = self._read_buffer[:self._frame_bytes]
        self._read_buffer = self._read_buffer[self._frame_bytes:]
        return frame
    
    async def write(self, audio: bytes) -> None:
        """Write audio to the client's speaker."""
        if self._client_ws is None:
            return
        
        try:
            # Prefix with message type
            message = bytes([self.MSG_SPEAKER]) + audio
            await self._client_ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Cannot write - client disconnected")
        except Exception as e:
            logger.warning(f"Write error: {e}")
    
    def write_sync(self, audio: bytes) -> None:
        """Synchronous write wrapper for TTS compatibility."""
        if self._client_ws is None or self._loop is None:
            return
        
        try:
            # Schedule the async write from sync context
            future = asyncio.run_coroutine_threadsafe(self.write(audio), self._loop)
            # Wait for completion with timeout
            future.result(timeout=1.0)
        except Exception as e:
            logger.warning(f"Sync write failed: {e}")
    
    async def close(self):
        """Close the server."""
        self._running = False
        if self._client_ws:
            try:
                await self._client_ws.close()
            except:
                pass
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("WebSocket audio server closed")


class Audio:
    """Unified audio interface for local or remote (WebSocket) audio"""
    
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