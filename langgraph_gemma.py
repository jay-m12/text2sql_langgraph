import os
import re
from typing_extensions import TypedDict, Annotated

import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langgraph.graph import START, StateGraph

from dotenv import load_dotenv
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

DB_PATH = "/home/sql/people/doyoung/langgraph/Chinook.db"
HF_API_PATH = "/home/sql/people/doyoung/langgraph/huggingface_api.env"



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"


db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}") 
print(db.dialect)
print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")


model_id = "google/gemma-2-2b-it"

quantization_config = BitsAndBytesConfig(      
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


load_dotenv(f"{HF_API_PATH}")

hf_token = os.getenv("hf_token")


tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    token=hf_token
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    top_k=50,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)


query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")


class State(TypedDict): 
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def extract_sql_query(response_text: str) -> str:
    match = re.search(r"SELECT[\s\S]+?;", response_text, re.IGNORECASE)
    return match.group(0).strip() if match else response_text.strip()


def write_query(state: State):
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    response = llm.invoke(prompt)   
    query = response if isinstance(response, str) else response.content
    query = extract_sql_query(query)
    return {"query": query}

def execute_query(state: State):
    execute_query_tool = QuerySQLDatabaseTool(db=db)    
    return {"result": execute_query_tool.invoke(state["query"])}   

def generate_answer(state: State):   
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question in korean.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.strip() if isinstance(response, str) else response.content}


graph_builder = StateGraph(State).add_sequence(    
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")   
graph = graph_builder.compile()

# # EX1
# print('=================EX1=================')
# for step in graph.stream(
#     {"question": "2023년에 두 번 이상 주문한 캐나다 고객이 있어?"}, stream_mode="updates"
# ):
#     print(step)
#     if "generate_answer" in step:
#         print("정답:", step["generate_answer"]["answer"])

# print('=================EX2=================')
# for step in graph.stream(
#     {"question": "총 고객 수는 몇 명이야?"}, stream_mode="updates"
# ):
#     print(step)
#     if "generate_answer" in step:
#         print("정답:", step["generate_answer"]["answer"])

# # EX2
# print('=================EX3=================')
# for step in graph.stream(
#     {"question": "전체 직원 중에서 5명의 직원을 알려줘."}, stream_mode="updates"
# ):
#     print(step)
#     if "generate_answer" in step:
#         print("정답:", step["generate_answer"]["answer"])

# # EX3
# print('=================EX4=================')
# for step in graph.stream(
#     {"question": "독일에서는 몇개의 결제가 진행됐어?"}, stream_mode="updates"
# ):
#     print(step)
#     if "generate_answer" in step:
#         print("정답:", step["generate_answer"]["answer"])


# print('=================EX5=================')
# for step in graph.stream(
#     {"question": "캐나다에 거주하면서 2009년에 주문한 고객들의 이름과 이메일 알려줘."}, stream_mode="updates"
# ):
#     print(step)
#     if "generate_answer" in step:
#         print("정답:", step["generate_answer"]["answer"])



# SQL 쿼리 실행
db_path = f"{DB_PATH}"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT FirstName, LastName FROM Customer WHERE Country = 'Canada'")
print(cursor.fetchall())

cursor.close()
conn.close()