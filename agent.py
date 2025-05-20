import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    print("Api key not set for Tavily.")

if not OPENAI_API_KEY:
    print("Api key is not set for Openai.")

search = TavilySearchResults(max_results=2)
# search_results = search.invoke("What's the weather in Budapest?")

tools = [search]

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", model_provider="openai")

from langchain_core.messages import HumanMessage

# response = model.invoke([HumanMessage(content="Hi!")])

model_with_tools = model.bind_tools(tools)

""" response = model_with_tools.invoke(
    [HumanMessage(content="What's the weather in Budapest?")]
) """

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="What's the weather in Budapest?")]}
)

# print(response["messages"])

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="What's the weather in Budapest?")]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
