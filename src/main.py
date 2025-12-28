"""
Main entry point with audio test.
"""
import asyncio
import logging
import signal
import os

from config import load_config, setup_logging
from audio import AudioConfig, Audio, OpenWakeWordDetector, WebRTCVADDetector, OpenWhisperASR, PiperTTS, PiperTTSConfig
from assistant import Assistant, AssistantConfig
from agent import AgentConfig, HomeAgent, MCPServerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    config = load_config()
    setup_logging(config.log_level)

    logger.info("Starting HomeGPT..")
    logger.info(f"API enabled: {config.enable_api}")
    logger.info(f"Audio type: {config.audio_type}")
    logger.info(f"Model provider: {config.model_provider}")
    
    # Handle shutdown signals
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Initialize audio with config
    audio_config = AudioConfig(
        audio_type=config.audio_type,
        ws_port=config.audio_ws_port
    )
    audio = Audio(audio_config)
    
    if config.audio_type == "remote":
        logger.info(f"Remote audio config:")
        logger.info(f"  Mic port: {config.audio_listener_port} (server listens)")
        logger.info(f"  Speaker port: {config.audio_speaker_port} (Pi listens)")

    wakeword = OpenWakeWordDetector()

    vad = WebRTCVADDetector()

    asr = OpenWhisperASR()

    # Initialize TTS if enabled
    tts = None
    if config.tts_enabled:
        tts_config = PiperTTSConfig(model_path=config.tts_model_path)
        tts = PiperTTS(tts_config)
        logger.info(f"TTS enabled: {config.tts_model_path}")

    # Build MCP server configs
    mcp_servers = []
    for mcp_module in config.mcps:
        mcp_servers.append(
            MCPServerConfig(
                name=mcp_module.split('.')[-1],
                command="uv",
                args=["run", "-m", mcp_module],
            )
        )
    
    # Build agent config based on provider
    agent_config = AgentConfig(
        model_provider=config.model_provider,
        # OpenAI settings
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_model,
        # Bedrock settings
        bedrock_model=config.bedrock_model,
        bedrock_region=config.bedrock_region,
        # MCP servers
        mcp_servers=mcp_servers,
    )
    
    agent = HomeAgent(agent_config)
    agent.start()

    # Build assistant
    assistant_config = AssistantConfig(
        audio=audio,
        wakeword=wakeword,
        vad=vad,
        asr=asr,
        agent=agent,
        tts=tts,
    )
    assistant = Assistant(assistant_config)
    
    # Run until shutdown
    assistant_task = asyncio.create_task(assistant.run())
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    
    # Shutdown
    done, pending = await asyncio.wait(
        [assistant_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    assistant_task.cancel()
    agent.stop()
    try:
        await assistant_task
    except asyncio.CancelledError:
        pass
    await assistant.stop()
    
    logger.info("Shutdown complete")
    os._exit(0)


def run():
    """Entry point for command line"""
    asyncio.run(main())


if __name__ == "__main__":
    run()