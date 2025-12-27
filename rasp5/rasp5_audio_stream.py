#!/usr/bin/env python3
"""
HomeGPT Raspberry Pi Audio Streamer

Streams microphone audio to server and receives TTS audio back.
Uses PyAudio for audio I/O and TCP sockets for network transport.

Usage:
    python pi_audio_streamer.py --server 192.168.1.100
    python pi_audio_streamer.py --server 192.168.1.100 --mic-port 4712 --speaker-port 4713
"""

import argparse
import logging
import socket
import threading
import time
import signal
import sys

import pyaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pi-audio")

# Audio settings (must match server)
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1280  # 80ms at 16kHz


class AudioStreamer:
    """Handles bidirectional audio streaming to/from server."""
    
    def __init__(self, server_ip: str, mic_port: int = 4712, speaker_port: int = 4713):
        self.server_ip = server_ip
        self.mic_port = mic_port
        self.speaker_port = speaker_port
        
        self.pa = None
        self.mic_stream = None
        self.speaker_stream = None
        
        self.mic_socket = None
        self.speaker_server = None
        self.speaker_conn = None
        
        self._running = False
        self._mic_thread = None
        self._speaker_thread = None
    
    def start(self):
        """Start audio streaming."""
        logger.info(f"Starting audio streamer")
        logger.info(f"  Server: {self.server_ip}")
        logger.info(f"  Mic port: {self.mic_port} (Pi → Server)")
        logger.info(f"  Speaker port: {self.speaker_port} (Server → Pi)")
        
        self._running = True
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self._log_audio_devices()
        
        # Start speaker listener first (server will connect to us)
        self._start_speaker_listener()
        
        # Start mic sender (we connect to server)
        self._start_mic_sender()
        
        logger.info("Audio streamer running. Press Ctrl+C to stop.")
    
    def _log_audio_devices(self):
        """Log available audio devices."""
        logger.info("Available audio devices:")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            direction = []
            if info['maxInputChannels'] > 0:
                direction.append("IN")
            if info['maxOutputChannels'] > 0:
                direction.append("OUT")
            logger.info(f"  [{i}] {info['name']} ({'/'.join(direction)})")
    
    def _start_speaker_listener(self):
        """Start thread that listens for incoming TTS audio."""
        self._speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._speaker_thread.start()
    
    def _speaker_loop(self):
        """Listen for server connection and play received audio."""
        while self._running:
            try:
                # Create server socket
                self.speaker_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.speaker_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.speaker_server.bind(("0.0.0.0", self.speaker_port))
                self.speaker_server.listen(1)
                self.speaker_server.settimeout(1.0)  # Check _running every second
                
                logger.info(f"Speaker listening on port {self.speaker_port}...")
                
                # Wait for server to connect
                while self._running:
                    try:
                        self.speaker_conn, addr = self.speaker_server.accept()
                        logger.info(f"Speaker connected from {addr}")
                        break
                    except socket.timeout:
                        continue
                
                if not self._running:
                    break
                
                # Open speaker stream
                self.speaker_stream = self.pa.open(
                    output=True,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=CHUNK_SIZE,
                )
                
                # Receive and play audio
                while self._running:
                    try:
                        data = self.speaker_conn.recv(4096)
                        if not data:
                            logger.info("Speaker connection closed by server")
                            break
                        self.speaker_stream.write(data)
                    except socket.error as e:
                        logger.warning(f"Speaker socket error: {e}")
                        break
                
            except Exception as e:
                logger.error(f"Speaker error: {e}")
            
            finally:
                # Cleanup for reconnect
                if self.speaker_stream:
                    self.speaker_stream.stop_stream()
                    self.speaker_stream.close()
                    self.speaker_stream = None
                if self.speaker_conn:
                    self.speaker_conn.close()
                    self.speaker_conn = None
                if self.speaker_server:
                    self.speaker_server.close()
                    self.speaker_server = None
            
            if self._running:
                logger.info("Speaker reconnecting in 2s...")
                time.sleep(2)
    
    def _start_mic_sender(self):
        """Start thread that sends microphone audio to server."""
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()
    
    def _mic_loop(self):
        """Connect to server and send microphone audio."""
        while self._running:
            try:
                # Connect to server
                self.mic_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.mic_socket.settimeout(10.0)
                
                logger.info(f"Mic connecting to {self.server_ip}:{self.mic_port}...")
                self.mic_socket.connect((self.server_ip, self.mic_port))
                logger.info("Mic connected to server")
                
                # Open mic stream
                self.mic_stream = self.pa.open(
                    input=True,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=CHUNK_SIZE,
                )
                
                # Send audio
                while self._running:
                    try:
                        data = self.mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        self.mic_socket.sendall(data)
                    except socket.error as e:
                        logger.warning(f"Mic socket error: {e}")
                        break
                    except IOError as e:
                        logger.warning(f"Mic read error: {e}")
                        break
                
            except socket.timeout:
                logger.warning("Mic connection timeout")
            except ConnectionRefusedError:
                logger.warning(f"Mic connection refused - is server running?")
            except Exception as e:
                logger.error(f"Mic error: {e}")
            
            finally:
                # Cleanup for reconnect
                if self.mic_stream:
                    self.mic_stream.stop_stream()
                    self.mic_stream.close()
                    self.mic_stream = None
                if self.mic_socket:
                    self.mic_socket.close()
                    self.mic_socket = None
            
            if self._running:
                logger.info("Mic reconnecting in 2s...")
                time.sleep(2)
    
    def stop(self):
        """Stop audio streaming."""
        logger.info("Stopping audio streamer...")
        self._running = False
        
        # Wait for threads
        if self._mic_thread:
            self._mic_thread.join(timeout=3)
        if self._speaker_thread:
            self._speaker_thread.join(timeout=3)
        
        # Cleanup
        if self.pa:
            self.pa.terminate()
        
        logger.info("Audio streamer stopped")
    
    def wait(self):
        """Wait for threads to finish."""
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="HomeGPT Pi Audio Streamer")
    parser.add_argument("--server", "-s", required=True, help="Server IP address")
    parser.add_argument("--mic-port", "-m", type=int, default=4712, help="Mic port (default: 4712)")
    parser.add_argument("--speaker-port", "-p", type=int, default=4713, help="Speaker port (default: 4713)")
    args = parser.parse_args()
    
    streamer = AudioStreamer(
        server_ip=args.server,
        mic_port=args.mic_port,
        speaker_port=args.speaker_port,
    )
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        streamer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    streamer.start()
    streamer.wait()


if __name__ == "__main__":
    main()
