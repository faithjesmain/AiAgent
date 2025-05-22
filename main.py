# set up simple agent

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

# define class that specifies type of content generated


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# set up llm & give it to agent to give agent tools to generate output
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# create parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# set up prompt template for future queries
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper. 
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# create the agent
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

# output parsing
try:
    structure_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structure_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
