"""Assistant module for homeGPT"""
import asyncio
import logging
import time
import struct
from dataclasses import dataclass

from audio import Audio, OpenWakeWordDetector, WebRTCVADDetector, OpenWhisperASR
from agent import HomeAgent

logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """Configuration for the assistant"""
    audio: Audio
    wakeword: OpenWakeWordDetector
    vad: WebRTCVADDetector
    asr: OpenWhisperASR
    agent: HomeAgent
    
    # Timing settings
    vad_delay_ms: int = 800  # Delay after wake word before listening
    vad_silence_ms: int = 600  # Silence duration to end utterance
    max_utterance_seconds: float = 30.0


class Assistant:
    """Main assistant class"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.audio = config.audio
        self.wakeword = config.wakeword
        self.vad = config.vad
        self.asr = config.asr
        self.agent = config.agent
        
        self._running = False
    
    async def run(self):
        """Run the assistant main loop"""
        self._running = True
        
        try:
            await self.audio.connect()
            logger.info("Assistant running, listening for wake word...")
            
            # Audio processing loop
            async for complete_utterance in self._audio_pipeline():
                logger.info(f"Got complete_utterance: {len(complete_utterance)} bytes")
                
                logger.info("Processing utterance with ASR...")
                start_time = time.time()
                request_text = await asyncio.to_thread(self.asr.transcribe, complete_utterance)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"Transcription ({elapsed_ms:.0f}ms): {request_text}")

                start_time = time.time()
                response = self.agent.process(request_text)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"Agent Response ({elapsed_ms:.0f}ms): {response}")

                # await self.speak(response)
                
        finally:
            await self.audio.close()
            self._running = False
    
    async def stop(self):
        """Stop the assistant"""
        self._running = False
    
    async def _audio_pipeline(self):
        """
        Audio processing pipeline.
        Yields complete utterances (bytes) after wake word detection.
        """
        # State
        collecting = False
        audio_buffer = []
        silence_frames = 0
        delay_frames = 0
        waiting_for_speech = True
        
        # Calculate frame counts
        vad_delay_frames = self.config.vad_delay_ms // 80  
        vad_silence_frames = self.config.vad_silence_ms // 80
        max_frames = int(self.config.max_utterance_seconds * 1000 / 80)
        
        while self._running:
            # Read 80ms frame
            frame = await self.audio.read()
            
            # Check for wake word
            wakeword_detected = self.wakeword.process_frame(frame)
            
            if not collecting and wakeword_detected:
                collecting = True
                audio_buffer = []
                silence_frames = 0
                delay_frames = 0
                waiting_for_speech = True
                
                logger.info("🔔 Wake word detected")
                await self.audio.play_alert()
                continue
            
            if collecting:
                # Buffer the audio
                audio_buffer.append(frame)
                
                # Run VAD on frames
                is_speech = self.vad.is_speech(frame)
                
                if waiting_for_speech:
                    delay_frames += 1
                    
                    # Still in delay period after wake word
                    if delay_frames < vad_delay_frames:
                        continue
                    
                    # Delay over, look for speech
                    if is_speech:
                        waiting_for_speech = False
                        silence_frames = 0
                        logger.debug("Speech started")
                else:
                    # Normal VAD processing
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                
                # Check if utterance is complete
                too_long = len(audio_buffer) > max_frames
                silence_timeout = not waiting_for_speech and silence_frames > vad_silence_frames
                
                if silence_timeout or too_long:
                    collecting = False
                    
                    # Combine buffered audio
                    utterance = b"".join(audio_buffer)
                    audio_buffer = []
                    
                    self.wakeword.reset()
                    
                    logger.info("📨 Utterance complete")
                    yield utterance
