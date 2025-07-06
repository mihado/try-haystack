# https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent
import os

from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.agents import Agent
from haystack.components.websearch import SerperDevWebSearch
from haystack.components.generators.utils import print_streaming_chunk

from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.fetchers.link_content import LinkContentFetcher



# Simple Agent

chat_generator = OllamaChatGenerator(model="qwen3:14b", url = os.environ["OLLAMA_URL"])

web_tool = ComponentTool(component=SerperDevWebSearch(), name="web_tool")

agent = Agent(
    chat_generator=chat_generator,
    tools=[web_tool],
    streaming_callback=print_streaming_chunk)

# result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])
# print(result["messages"][-1].text)



# Deep Research Agent

search_pipeline = Pipeline()
search_pipeline.add_component("search", SerperDevWebSearch(top_k=10))
search_pipeline.add_component("fetcher", LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2))
search_pipeline.add_component("converter", HTMLToDocument())

output_template="""
{%- for doc in docs -%}
  {%- if doc.content -%}
  <search-result url="{{ doc.meta.url }}">
  {{ doc.content|truncate(25000) }}
  </search-result>
  {%- endif -%}
{%- endfor -%}
"""
search_pipeline.add_component("output_adapter", OutputAdapter(template=output_template, output_type=str))

search_pipeline.connect("search.links", "fetcher.urls")
search_pipeline.connect("fetcher.streams", "converter.sources")
search_pipeline.connect("converter.documents", "output_adapter.docs")

search_component = SuperComponent(
    pipeline=search_pipeline,
    input_mapping={"query": ["search.query"]},
    output_mapping={"output_adapter.output": "search_result"}
)

search_tool = ComponentTool(
    name="search",
    description="Internet search.",
    component=search_component,
    outputs_to_string={"source": "search_result"}
)

search_agent = Agent(
    chat_generator=chat_generator,
    tools=[search_tool],
    system_prompt="""
    You are a deep research assistant.
    You create comprehensive research reports to answer the user's questions.
    You use the 'search'-tool to answer any questions.
    You perform multiple searches until you have the information you need to answer the question.
    Make sure you research different aspects of the question.
    Use markdown to format your response.
    When you use information from the websearch results, cite your sources using markdown links.
    It is important that you cite accurately.
    """,
    exit_conditions=["text"],
    max_agent_steps=20,
    streaming_callback=print_streaming_chunk,
)
search_agent.warm_up()


query = "What are the latest updates on the Artemis moon mission?"

messages = [ChatMessage.from_user(query)]

agent_output = search_agent.run(messages=messages)
print(agent_output["messages"][-1].text)
