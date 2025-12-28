#!/usr/bin/env python3
"""
HomeGPT Raspberry Pi Audio Streamer - WebSocket Version

More reliable than raw TCP sockets through K8s/MetalLB.

Usage:
    uv run pi_audio_ws.py
    uv run pi_audio_ws.py --server ws://192.168.1.100:8765
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time

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
        self._streaming_enabled = False
        self._state_lock = threading.Lock()
        
        # Silence frame for when streaming is disabled
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
        """Read from mic and send to server."""
        while self._running:
            try:
                if self.mic_stream and self.websocket:
                    # Always read to keep buffer clear
                    data = self.mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    
                    if self._is_streaming_enabled():
                        # Send real audio
                        message = bytes([MSG_MIC]) + data
                    else:
                        # Send silence
                        message = bytes([MSG_MIC]) + self._silence_frame
                    
                    await self.websocket.send(message)
                else:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.warning(f"Send error: {e}")
                await asyncio.sleep(0.1)
    
    async def _receive_audio(self):
        """Receive audio from server and play."""
        while self._running:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    
                    if isinstance(message, bytes) and len(message) > 0:
                        msg_type = message[0]
                        payload = message[1:]
                        
                        if msg_type == MSG_SPEAKER and self.speaker_stream:
                            self.speaker_stream.write(payload)
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed during receive")
                break
            except Exception as e:
                logger.warning(f"Receive error: {e}")
                await asyncio.sleep(0.1)
    
    async def _connect(self):
        """Connect to WebSocket server with retry."""
        while self._running:
            try:
                logger.info(f"Connecting to {self.server_url}...")
                
                self.websocket = await websockets.connect(
                    self.server_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                )
                
                logger.info("WebSocket connected!")
                return True
                
            except Exception as e:
                logger.warning(f"Connection failed: {e}, retrying in 2s...")
                await asyncio.sleep(2)
        
        return False
    
    async def run(self):
        """Main run loop with automatic reconnection."""
        self._running = True
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        
        self.mic_stream = self.pa.open(
            input=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        self.speaker_stream = self.pa.open(
            output=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        # Start HA monitor if configured
        if self.ha_client:
            asyncio.create_task(self._ha_monitor())
        else:
            self._streaming_enabled = True
        
        logger.info("Audio streamer running...")
        
        # Main loop with reconnection
        while self._running:
            try:
                if await self._connect():
                    # Run send and receive concurrently
                    send_task = asyncio.create_task(self._send_audio())
                    recv_task = asyncio.create_task(self._receive_audio())
                    
                    # Wait for either to complete (usually due to disconnect)
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    
                    # Cancel the other task
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Close websocket
                    if self.websocket:
                        await self.websocket.close()
                        self.websocket = None
                    
                    if self._running:
                        logger.info("Disconnected, reconnecting in 2s...")
                        await asyncio.sleep(2)
                        
            except Exception as e:
                logger.error(f"Run loop error: {e}")
                await asyncio.sleep(2)
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if self.speaker_stream:
            self.speaker_stream.stop_stream()
            self.speaker_stream.close()
        if self.pa:
            self.pa.terminate()
        logger.info("Audio streamer stopped")
    
    def stop(self):
        """Stop the streamer."""
        self._running = False


async def main_async(server_url: str, ha_client: HomeAssistantClient | None):
    streamer = WebSocketAudioStreamer(server_url, ha_client)
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        streamer.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    await streamer.run()


def main():
    parser = argparse.ArgumentParser(description="HomeGPT Pi Audio Streamer (WebSocket)")
    parser.add_argument("--server", "-s", help="WebSocket server URL (e.g., ws://192.168.1.100:8765)")
    parser.add_argument("--ha-url", help="Home Assistant URL")
    parser.add_argument("--ha-token", help="Home Assistant token")
    parser.add_argument("--ha-entity", help="HA entity ID")
    args = parser.parse_args()
    
    server_ip = os.environ.get("SERVER_IP")
    server_port = os.environ.get("WS_PORT", "8765")
    server_url = args.server or os.environ.get("WS_SERVER_URL") or f"ws://{server_ip}:{server_port}"
    
    if not server_url or server_url == f"ws://None:{server_port}":
        logger.error("Server URL required. Set SERVER_IP or WS_SERVER_URL in .env or use --server")
        sys.exit(1)
    
    ha_url = args.ha_url or os.environ.get("HOMEASSISTANT_URL")
    ha_token = args.ha_token or os.environ.get("HOMEASSISTANT_TOKEN")
    ha_entity = args.ha_entity or os.environ.get("HA_ENTITY", "input_boolean.homegpt")
    
    ha_client = None
    if ha_url and ha_token:
        ha_client = HomeAssistantClient(ha_url, ha_token, ha_entity)
        logger.info(f"Home Assistant control enabled")
    
    logger.info(f"Starting WebSocket audio streamer")
    logger.info(f"  Server: {server_url}")
    
    asyncio.run(main_async(server_url, ha_client))


if __name__ == "__main__":
    main()