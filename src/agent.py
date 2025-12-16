"""Agent module for homeGPT using Strands with OpenAI and MCPs"""
import os
import logging
from contextlib import ExitStack
from dataclasses import dataclass, field

from strands import Agent as StrandsAgent 
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a Home Assistant operator for my house. 
Keep replies brief and action-focused (optimize for TTS). 
When I ask for something, do this policy:

1) Resolve intent:
   - Action (on/off/toggle/set/scene) + Targets (entities/areas) + Options (brightness, temp, %).
   - If the entity_id is unknown, search by friendly name/area; prefer best-effort resolution.
   - Avoid asking for clarification unless there is true ambiguity in what the request is. 
   - Ask 1 short clarification if the action is risky (unlock/door/drain).

2) Grounding:
   - Never invent entity_ids. Use tools to find them:
     • `list_entities`
     • `get_entity` for exact state/attributes
     • When searching for devices only use one key word (e.g., "light" or "office").
   - Use your best judgment to resolve ambiguities (e.g., multiple "closet light", "upstairs living room" is the "living room").

3) No-ops & batching:
   - If the device is already in the desired state, say so and don't call a service.
   - If a request clearly affects multiple devices ("turn off office lights"), batch them.

4) Act & verify:
   - Use `entity_action` (on/off/toggle) or `call_service_tool` for custom services.

5) Safety & confirmations:
   - For doors/locks/garage/openers/alarms/critical HVAC changes: require explicit confirmation.
   - Do not execute destructive actions without confirmation.

6) Errors:
   - If Home Assistant is unreachable or returns an error, run a quick health check if available (e.g., `ha_ping`) and report a short actionable message.

Style:
- Be concise. Do not explain reasoning for how you arrived to an answer unless requested. Examples: 
  "Turned on office closet light… (100%)."
  "Already off."
  "Multiple 'closet light' found: office, hallway. Which?"
  "12 inches by 12 inches by 12 inches equals 7.48 gallons."
"""


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    command: str
    args: list[str]
    name: str = "mcp-server"
    env: dict | None = None


@dataclass
class AgentConfig:
    """Configuration for the agent"""
    openai_api_key: str
    openai_model: str = "gpt-4.1"
    temperature: float = 0.7
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)


class HomeAgent:
    """
    Home Assistant agent using Strands with OpenAI and MCPs.
    
    Usage:
        agent = HomeAgent(config)
        agent.start()  # Initialize MCP connections once
        
        # For single requests:
        response = agent.process("turn on the lights")
        
        # For async usage in the assistant:
        response = await agent.process_async("turn on the lights")
        
        agent.stop()  # Cleanup on shutdown
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._model = self._init_model()
        self._mcp_clients = self._init_mcp_clients()
        self._exit_stack: ExitStack | None = None
        self._tools: list = []
        self._started = False
        
        logger.info(f"HomeAgent initialized with model {self.config.openai_model}")
        logger.info(f"Configured {len(self._mcp_clients)} MCP servers")
    
    def _init_model(self) -> OpenAIModel:
        """Initialize the OpenAI model"""
        return OpenAIModel(
            client_args={"api_key": self.config.openai_api_key},
            model_id=self.config.openai_model,
            params={"temperature": self.config.temperature},
        )
    
    def _init_mcp_clients(self) -> list[MCPClient]:
        """Initialize MCP clients for each configured server"""
        clients = []
        
        for server_config in self.config.mcp_servers:
            env = server_config.env if server_config.env is not None else dict(os.environ)
            
            client = MCPClient(lambda sc=server_config, e=env: stdio_client(
                StdioServerParameters(
                    command=sc.command,
                    args=sc.args,
                    env=e,
                )
            ))
            clients.append(client)
            logger.debug(f"Configured MCP client: {server_config.name}")
        
        return clients
    
    def start(self):
        """
        Start MCP servers and load tools.
        Call this once at application startup.
        """
        if self._started:
            logger.warning("HomeAgent already started")
            return
        
        self._exit_stack = ExitStack()
        self._tools = []
        
        for client in self._mcp_clients:
            opened_client = self._exit_stack.enter_context(client)
            client_tools = opened_client.list_tools_sync()
            self._tools.extend(client_tools)
            logger.debug(f"Loaded {len(client_tools)} tools from MCP server")
        
        self._started = True
        logger.info(f"HomeAgent started with {len(self._tools)} tools available")
    
    def stop(self):
        """
        Stop MCP servers and cleanup.
        Call this at application shutdown.
        """
        if not self._started:
            return
        
        if self._exit_stack:
            self._exit_stack.close()
            self._exit_stack = None
        
        self._tools = []
        self._started = False
        logger.info("HomeAgent stopped")
    
    def process(self, request_text: str) -> str:
        """
        Process a request synchronously.
        
        Args:
            request_text: The user's request
            
        Returns:
            The agent's response text
        """
        if not self._started:
            raise RuntimeError("HomeAgent not started. Call start() first.")
        
        if not request_text.strip():
            return "I didn't catch that. Could you repeat?"
        
        logger.info(f"Processing request: {request_text}")
        
        # Create agent with pre-loaded tools
        agent = StrandsAgent(
            model=self._model,
            tools=self._tools,
            system_prompt=self.config.system_prompt,
        )
        
        result = agent(request_text)
        response_text = result.message["content"][0]["text"]
        
        logger.info(f"Response: {response_text}")
        return response_text
    
    async def process_async(self, request_text: str) -> str:
        """
        Process a request asynchronously (runs sync code in thread pool).
        
        Args:
            request_text: The user's request
            
        Returns:
            The agent's response text
        """
        import asyncio
        return await asyncio.to_thread(self.process, request_text)