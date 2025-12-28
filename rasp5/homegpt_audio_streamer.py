#!/usr/bin/env python3
"""
HomeGPT Raspberry Pi Audio Streamer

Streams microphone audio to server and receives TTS audio back.
Uses PyAudio for audio I/O and TCP sockets for network transport.
Controlled by Home Assistant switch.homegpt entity.

Usage:
    uv run pi_audio_streamer.py
    uv run pi_audio_streamer.py --server 192.168.1.100
"""

import argparse
import logging
import socket
import struct
import threading
import time
import signal
import sys
import os

import pyaudio
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

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

# Home Assistant settings
HA_CHECK_INTERVAL = 5.0  # seconds
HA_ENTITY_ID = os.environ.get("HA_ENTITY", "input_boolean.homegpt")


class HomeAssistantClient:
    """Client for checking Home Assistant switch state."""
    
    def __init__(self, ha_url: str, ha_token: str, entity_id: str = HA_ENTITY_ID):
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self.entity_id = entity_id
        self._headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }
    
    def is_switch_on(self) -> bool:
        """Check if the Home Assistant switch is on.
        
        Returns False if:
        - Switch is off
        - Request fails
        - Entity not found
        - Any error occurs
        """
        try:
            url = f"{self.ha_url}/api/states/{self.entity_id}"
            response = requests.get(url, headers=self._headers, timeout=5.0)
            
            if response.status_code == 200:
                state = response.json().get("state", "off")
                return state.lower() == "on"
            else:
                logger.warning(f"HA API returned status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning("HA API request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning("HA API connection failed")
            return False
        except Exception as e:
            logger.error(f"HA API error: {e}")
            return False


class AudioStreamer:
    """Handles bidirectional audio streaming to/from server with persistent connections."""
    
    def __init__(
        self,
        server_ip: str,
        mic_port: int = 4712,
        speaker_port: int = 4713,
        ha_client: HomeAssistantClient | None = None,
    ):
        self.server_ip = server_ip
        self.mic_port = mic_port
        self.speaker_port = speaker_port
        self.ha_client = ha_client
        
        self.pa = None
        self.mic_stream = None
        self.speaker_stream = None
        
        self.mic_socket = None
        self.speaker_server = None
        self.speaker_conn = None
        
        self._running = False
        self._streaming_enabled = False  # Controlled by HA switch
        self._mic_thread = None
        self._speaker_thread = None
        self._ha_thread = None
        
        # Lock for thread-safe access to streaming state
        self._state_lock = threading.Lock()
    
    def start(self):
        """Start audio streamer with persistent connections."""
        logger.info(f"Starting audio streamer (persistent connection mode)")
        logger.info(f"  Server: {self.server_ip}")
        logger.info(f"  Mic port: {self.mic_port} (Pi → Server)")
        logger.info(f"  Speaker port: {self.speaker_port} (Server → Pi)")
        
        if self.ha_client:
            logger.info(f"  HA control: {self.ha_client.entity_id}")
        else:
            logger.info("  HA control: disabled (always streaming)")
            self._streaming_enabled = True
        
        self._running = True
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self._log_audio_devices()
        
        # Start HA monitor if configured
        if self.ha_client:
            self._start_ha_monitor()
        
        # Start speaker listener (persistent connection)
        self._start_speaker_listener()
        
        # Start mic sender (persistent connection)
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
    
    def _start_ha_monitor(self):
        """Start thread that monitors Home Assistant switch state."""
        self._ha_thread = threading.Thread(target=self._ha_monitor_loop, daemon=True)
        self._ha_thread.start()
    
    def _ha_monitor_loop(self):
        """Monitor Home Assistant switch and update streaming state."""
        while self._running:
            try:
                new_state = self.ha_client.is_switch_on()
                
                with self._state_lock:
                    if new_state != self._streaming_enabled:
                        self._streaming_enabled = new_state
                        if new_state:
                            logger.info("HA switch ON - audio streaming resumed")
                        else:
                            logger.info("HA switch OFF - audio streaming paused (connection maintained)")
                
            except Exception as e:
                logger.error(f"HA monitor error: {e}")
                # Don't change streaming state on error - maintain current state
            
            time.sleep(HA_CHECK_INTERVAL)
    
    def _is_streaming_enabled(self) -> bool:
        """Thread-safe check of streaming state."""
        with self._state_lock:
            return self._streaming_enabled
    
    def _start_speaker_listener(self):
        """Start thread that listens for incoming TTS audio."""
        self._speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._speaker_thread.start()
    
    def _speaker_loop(self):
        """Maintain persistent speaker connection, always receive and play audio."""
        while self._running:
            try:
                # Create server socket
                self.speaker_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.speaker_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.speaker_server.bind(("0.0.0.0", self.speaker_port))
                self.speaker_server.listen(1)
                self.speaker_server.settimeout(1.0)
                
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
                
                # Always receive and play audio (server controls what it sends)
                # Connection stays open regardless of HA switch state
                while self._running:
                    try:
                        data = self.speaker_conn.recv(4096)
                        if not data:
                            logger.info("Speaker connection closed by server")
                            break
                        # Always play received audio - server decides what to send
                        self.speaker_stream.write(data)
                    except socket.timeout:
                        continue
                    except socket.error as e:
                        logger.warning(f"Speaker socket error: {e}")
                        break
                
            except Exception as e:
                logger.error(f"Speaker error: {e}")
            
            finally:
                self._cleanup_speaker()
            
            if self._running:
                logger.info("Speaker reconnecting in 2s...")
                time.sleep(2)
    
    def _cleanup_speaker(self):
        """Clean up speaker resources."""
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except:
                pass
            self.speaker_stream = None
        if self.speaker_conn:
            try:
                self.speaker_conn.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.speaker_conn.close()
            except:
                pass
            self.speaker_conn = None
        if self.speaker_server:
            try:
                self.speaker_server.close()
            except:
                pass
            self.speaker_server = None
    
    def _start_mic_sender(self):
        """Start thread that sends microphone audio to server."""
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()
    
    def _mic_loop(self):
        """Maintain persistent mic connection, send audio only when enabled."""
        while self._running:
            try:
                # Connect to server
                self.mic_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.mic_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.mic_socket.settimeout(10.0)
                
                logger.info(f"Mic connecting to {self.server_ip}:{self.mic_port}...")
                self.mic_socket.connect((self.server_ip, self.mic_port))
                self.mic_socket.settimeout(None)  # No timeout once connected
                logger.info("Mic connected to server (persistent connection)")
                
                # Open mic stream
                self.mic_stream = self.pa.open(
                    input=True,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=CHUNK_SIZE,
                )
                
                # Persistent connection loop
                while self._running:
                    try:
                        # Always read from mic to keep buffer clear
                        data = self.mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Only send if streaming is enabled
                        if self._is_streaming_enabled():
                            self.mic_socket.sendall(data)
                        # else: discard audio data but keep connection alive
                        
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
                self._cleanup_mic()
            
            if self._running:
                logger.info("Mic reconnecting in 2s...")
                time.sleep(2)
    
    def _cleanup_mic(self):
        """Clean up mic resources."""
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
            self.mic_stream = None
        if self.mic_socket:
            try:
                self.mic_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.mic_socket.close()
            except:
                pass
            self.mic_socket = None
    
    def stop(self):
        """Stop audio streaming."""
        logger.info("Stopping audio streamer...")
        self._running = False
        
        # Wait for threads
        if self._mic_thread:
            self._mic_thread.join(timeout=3)
        if self._speaker_thread:
            self._speaker_thread.join(timeout=3)
        if self._ha_thread:
            self._ha_thread.join(timeout=3)
        
        # Cleanup
        self._cleanup_mic()
        self._cleanup_speaker()
        
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
    parser.add_argument("--server", "-s", help="Server IP address")
    parser.add_argument("--mic-port", "-m", type=int, help="Mic port")
    parser.add_argument("--speaker-port", "-p", type=int, help="Speaker port")
    parser.add_argument("--ha-url", help="Home Assistant URL")
    parser.add_argument("--ha-token", help="Home Assistant long-lived access token")
    parser.add_argument("--ha-entity", help="HA entity ID")
    args = parser.parse_args()
    
    # Load from args, then env vars, then defaults
    server_ip = args.server or os.environ.get("SERVER_IP")
    mic_port = args.mic_port or int(os.environ.get("MIC_PORT", "4712"))
    speaker_port = args.speaker_port or int(os.environ.get("SPEAKER_PORT", "4713"))
    ha_url = args.ha_url or os.environ.get("HOMEASSISTANT_URL")
    ha_token = args.ha_token or os.environ.get("HOMEASSISTANT_TOKEN")
    ha_entity = args.ha_entity or os.environ.get("HA_ENTITY", "input_boolean.homegpt")
    
    if not server_ip:
        logger.error("Server IP required. Set SERVER_IP in .env or use --server")
        sys.exit(1)
    
    # Create HA client if configured
    ha_client = None
    if ha_url and ha_token:
        ha_client = HomeAssistantClient(ha_url, ha_token, ha_entity)
        logger.info(f"Home Assistant control enabled: {ha_url}")
    else:
        logger.warning("No HA credentials provided - streaming will always be enabled")
    
    streamer = AudioStreamer(
        server_ip=server_ip,
        mic_port=mic_port,
        speaker_port=speaker_port,
        ha_client=ha_client,
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