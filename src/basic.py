import os

from haystack.dataclasses import ChatMessage

from haystack.components.agents import Agent
from haystack.components.websearch import SerperDevWebSearch

from haystack.tools import ComponentTool
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

# check ENV_VARS are loaded
# print(os.environ["GOOGLE_API_KEY"])
# print(os.environ["SERPERDEV_API_KEY"])

search_tool = ComponentTool(component=SerperDevWebSearch())
system_prompt="You are a helpful web agent."
tools=[search_tool]

gemini_agent = Agent(
    chat_generator=GoogleGenAIChatGenerator(model="gemini-2.5-flash"),
    system_prompt=system_prompt,
    tools=tools,
)

ollama_agent = Agent(
    chat_generator=OllamaChatGenerator(model="qwen3:14b", url = os.environ["OLLAMA_URL"]),
    system_prompt=system_prompt,
    tools=tools,
)

basic_agent = ollama_agent

result = basic_agent.run(
    messages=[ChatMessage.from_user("When was the first version of Haystack (by deepset) released?")]
)

# print(result)
print(result['last_message'].text)
