#!/usr/bin/env python3
"""
HomeGPT Raspberry Pi Audio Streamer - WebSocket Version
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
import queue

import pyaudio
import requests
import websockets
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pi-audio-ws")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1280  # 80ms at 16kHz

# Message types (must match server)
MSG_MIC = 0x01
MSG_SPEAKER = 0x02
MSG_CONTROL = 0x03

# Home Assistant settings
HA_CHECK_INTERVAL = 5.0
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
        """Check if the Home Assistant switch is on."""
        try:
            url = f"{self.ha_url}/api/states/{self.entity_id}"
            response = requests.get(url, headers=self._headers, timeout=5.0)
            if response.status_code == 200:
                state = response.json().get("state", "off")
                return state.lower() == "on"
            return False
        except Exception as e:
            logger.warning(f"HA check failed: {e}")
            return False


class WebSocketAudioStreamer:
    """WebSocket-based audio streamer with automatic reconnection."""
    
    def __init__(
        self,
        server_url: str,
        ha_client: HomeAssistantClient | None = None,
    ):
        self.server_url = server_url
        self.ha_client = ha_client
        
        self.pa = None
        self.mic_stream = None
        self.speaker_stream = None
        self.websocket = None
        
        self._running = False
        self._connected = False
        self._streaming_enabled = True
        self._state_lock = threading.Lock()
        
        # Queues for thread-safe audio transfer
        self._mic_queue = queue.Queue(maxsize=100)
        self._speaker_queue = queue.Queue(maxsize=100)
        
        # Silence frame
        self._silence_frame = b'\x00' * (CHUNK_SIZE * 2)
    
    def _is_streaming_enabled(self) -> bool:
        with self._state_lock:
            return self._streaming_enabled
    
    def _set_streaming_enabled(self, value: bool):
        with self._state_lock:
            if value != self._streaming_enabled:
                self._streaming_enabled = value
                if value:
                    logger.info("HA switch ON - sending real audio")
                else:
                    logger.info("HA switch OFF - sending silence")
    
    def _mic_reader_thread(self):
        """Thread that reads from microphone and queues audio."""
        logger.info("Mic reader thread started")
        while self._running:
            try:
                if self.mic_stream:
                    data = self.mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    
                    if self._is_streaming_enabled():
                        audio = data
                    else:
                        audio = self._silence_frame
                    
                    try:
                        self._mic_queue.put_nowait(audio)
                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            self._mic_queue.get_nowait()
                            self._mic_queue.put_nowait(audio)
                        except:
                            pass
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Mic reader error: {e}")
                time.sleep(0.1)
        logger.info("Mic reader thread ended")
    
    def _speaker_writer_thread(self):
        """Thread that plays audio from queue to speaker."""
        logger.info("Speaker writer thread started")
        while self._running:
            try:
                audio = self._speaker_queue.get(timeout=0.5)
                if self.speaker_stream:
                    self.speaker_stream.write(audio)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speaker writer error: {e}")
                time.sleep(0.1)
        logger.info("Speaker writer thread ended")
    
    async def _ha_monitor(self):
        """Monitor Home Assistant switch state."""
        while self._running:
            try:
                if self.ha_client:
                    new_state = self.ha_client.is_switch_on()
                    self._set_streaming_enabled(new_state)
            except Exception as e:
                logger.error(f"HA monitor error: {e}")
            await asyncio.sleep(HA_CHECK_INTERVAL)
    
    async def _send_audio(self):
        """Send mic audio from queue to server."""
        logger.info("Send audio task started")
        try:
            while self._running and self.websocket and self._connected:
                try:
                    # Non-blocking get from queue
                    try:
                        audio = self._mic_queue.get_nowait()
                        message = bytes([MSG_MIC]) + audio
                        await self.websocket.send(message)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Send: Connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    await asyncio.sleep(0.1)
        finally:
            logger.info("Send audio task ended")
    
    async def _receive_audio(self):
        """Receive audio from server and queue for playback."""
        logger.info("Receive audio task started")
        try:
            while self._running and self.websocket and self._connected:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0
                    )
                    
                    if isinstance(message, bytes) and len(message) > 0:
                        msg_type = message[0]
                        payload = message[1:]
                        
                        if msg_type == MSG_SPEAKER:
                            logger.info(f"Received speaker audio: {len(payload)} bytes")
                            try:
                                self._speaker_queue.put_nowait(payload)
                            except queue.Full:
                                logger.warning("Speaker queue full, dropping audio")
                                
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Receive: Connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Receive error: {e}")
                    await asyncio.sleep(0.1)
        finally:
            logger.info("Receive audio task ended")
    
    def _init_audio(self):
        """Initialize PyAudio streams."""
        logger.info("Initializing audio...")
        
        if self.pa is None:
            self.pa = pyaudio.PyAudio()
        
        # Close existing streams
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
        
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except:
                pass
        
        # Open new streams
        self.mic_stream = self.pa.open(
            input=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("Mic stream opened")
        
        self.speaker_stream = self.pa.open(
            output=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("Speaker stream opened")
    
    async def _connect(self):
        """Connect to WebSocket server."""
        try:
            logger.info(f"Connecting to {self.server_url}...")
            
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            
            self._connected = True
            logger.info("WebSocket connected!")
            return True
            
        except Exception as e:
            logger.warning(f"Connection failed: {e}")
            self._connected = False
            return False
    
    async def run(self):
        """Main run loop with automatic reconnection."""
        self._running = True
        
        # Initialize audio
        self._init_audio()
        
        # Start audio threads
        mic_thread = threading.Thread(target=self._mic_reader_thread, daemon=True)
        mic_thread.start()
        
        speaker_thread = threading.Thread(target=self._speaker_writer_thread, daemon=True)
        speaker_thread.start()
        
        # Start HA monitor
        ha_task = None
        if self.ha_client:
            ha_task = asyncio.create_task(self._ha_monitor())
        else:
            self._streaming_enabled = True
        
        logger.info("Audio streamer running...")
        
        # Main loop
        while self._running:
            try:
                if await self._connect():
                    # Run send and receive concurrently
                    send_task = asyncio.create_task(self._send_audio())
                    recv_task = asyncio.create_task(self._receive_audio())
                    
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    
                    for task in done:
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Task failed: {exc}")
                        except:
                            pass
                    
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    self._connected = False
                    if self.websocket:
                        try:
                            await self.websocket.close()
                        except:
                            pass
                        self.websocket = None
                
                if self._running:
                    logger.info("Disconnected, reconnecting in 2s...")
                    await asyncio.sleep(2)
                        
            except Exception as e:
                logger.error(f"Run loop error: {e}")
                await asyncio.sleep(2)
        
        # Cleanup
        if ha_task:
            ha_task.cancel()
            try:
                await ha_task
            except asyncio.CancelledError:
                pass
        
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self._running = False
        
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
            self.mic_stream = None
        
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except:
                pass
            self.speaker_stream = None
        
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None
        
        logger.info("Audio streamer stopped")
    
    def stop(self):
        """Stop the streamer."""
        logger.info("Stop requested")
        self._running = False


async def main_async(server_url: str, ha_client: HomeAssistantClient | None):
    streamer = WebSocketAudioStreamer(server_url, ha_client)
    
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Signal received, stopping...")
        streamer.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    await streamer.run()


def main():
    parser = argparse.ArgumentParser(description="HomeGPT Pi Audio Streamer (WebSocket)")
    parser.add_argument("--server", "-s", help="WebSocket server URL")
    parser.add_argument("--ha-url", help="Home Assistant URL")
    parser.add_argument("--ha-token", help="Home Assistant token")
    parser.add_argument("--ha-entity", help="HA entity ID")
    args = parser.parse_args()
    
    server_ip = os.environ.get("SERVER_IP")
    server_port = os.environ.get("WS_PORT", "8765")
    server_url = args.server or os.environ.get("WS_SERVER_URL") or (f"ws://{server_ip}:{server_port}" if server_ip else None)
    
    if not server_url:
        logger.error("Server URL required")
        sys.exit(1)
    
    ha_url = args.ha_url or os.environ.get("HOMEASSISTANT_URL")
    ha_token = args.ha_token or os.environ.get("HOMEASSISTANT_TOKEN")
    ha_entity = args.ha_entity or os.environ.get("HA_ENTITY", "input_boolean.homegpt")
    
    ha_client = None
    if ha_url and ha_token:
        ha_client = HomeAssistantClient(ha_url, ha_token, ha_entity)
        logger.info("Home Assistant control enabled")
    else:
        logger.info("No HA credentials - always streaming")
    
    logger.info(f"Starting WebSocket audio streamer")
    logger.info(f"  Server: {server_url}")
    
    asyncio.run(main_async(server_url, ha_client))


if __name__ == "__main__":
    main()