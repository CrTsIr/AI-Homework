from openai import OpenAI
from colorama import Fore
from camel.utils import print_text_animated

from camel.societies import RolePlaying
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

from knowledge_generator import LocalKnowledgeBase

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='./.env')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    url=os.getenv('SILICONFLOW_BASE_URL'),
    api_key=os.getenv('SILICONFLOW_API_KEY'),
    model_config_dict={
        # 核心参数
        "temperature": 0.1,        # 控制随机性 (0-1)
        "max_tokens": 16384,         # 最大生成长度
    }
)

print("请输入两行，第一行为题面的文件名，第二行为题解的文件名\n")

statement_file=input()
tutorial_file=input()

with open(statement_file, 'r',encoding='utf-8', errors='ignore') as file:
    statement_text = file.read()

with open(tutorial_file, 'r',encoding='utf-8', errors='ignore') as file:
    tutorial_text = file.read()

agent = ChatAgent(
    model=model,
    output_language='中文'
)


knowledge_base = LocalKnowledgeBase(model,
"你是一个专业耐心的信息学竞赛教师，根据供的上下文以及聊天的上下文引导学生做出题目"
"尽量在一次回复中提供有意义而尽可能少的信息，以引导学生思考。"
"题面为{"+statement_text+"}，题解为{"+tutorial_text+"}。"
"如果你对用户对题目的了解情况有较大程度的疑问，可以向用户询问。"
"如果用户的要求不恰当，你可以回绝。"
"在提示时尽量保证自己的回答简洁有效，比如一个小性质或者做法的下一步。"
"在提示时如果供的上下文和聊天相关性不够高，则不要提到上下文的存在。"
"回答一定要包含关键信息。"
"尽量让自己的回答不包含复杂的术语，或者说使回答尽量初等。"
)
# knowledge_base.import_from_json("knowledge_export.json")

import time

while True:
    print("\n" + "="*50)
    query = input("对话中 (输入 'exit' 退出): ")
    
    if query.lower() in ["exit", "quit"]:
        break
    query="在本题中{"+query+"}"
    response = knowledge_base.query_knowledge(query, show_context=False)
    print("\n" + response)
