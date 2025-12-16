"""
Simple test script to record audio and play it back.
"""
import asyncio
import logging

import numpy as np
import pyaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioConfig:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.input_frames = 1280  # 80ms at 16kHz


class LocalAudio:
    """Local audio using PyAudio"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
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
        logger.info("Audio connected")
    
    async def read(self) -> bytes:
        """Read one 80ms frame of audio"""
        return await asyncio.to_thread(
            self.input_stream.read, 
            self.config.input_frames
        )
    
    async def write(self, audio: bytes) -> None:
        """Write audio to speakers"""
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
        logger.info("Audio closed")


async def main():
    config = AudioConfig()
    audio = LocalAudio(config)
    
    await audio.connect()
    
    # Calculate how many frames for desired duration
    record_seconds = 3
    frame_duration_ms = 80
    num_frames = int(record_seconds * 1000 / frame_duration_ms)
    
    print(f"\nRecording {record_seconds} seconds... Speak now!")
    
    # Record
    recorded_frames = []
    for i in range(num_frames):
        frame = await audio.read()
        recorded_frames.append(frame)
        
        # Progress indicator
        if i % 5 == 0:
            print(".", end="", flush=True)
    
    print(f"\n\nRecorded {len(recorded_frames)} frames")
    print("Playing back...")
    
    # Playback
    for frame in recorded_frames:
        await audio.write(frame)
    
    print("Done!")
    
    await audio.close()


if __name__ == "__main__":
    asyncio.run(main())