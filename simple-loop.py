"""
简单claude code实现
"""
import os

import dotenv
from langchain.chat_models import init_chat_model
import subprocess

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage

dotenv.load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY')
api_url = os.getenv('DEEPSEEK_API_URL')
MODEL_ID = os.getenv('MODULE_ID')
llm = init_chat_model(
    model=MODEL_ID,
    api_key=api_key,
    base_url=api_url,
)

@tool
def run_bash(command:str)-> str:
    """
    运行bash命令

    Args:
        command: 要执行的 bash 命令

    Returns:
        命令执行结果字符串

    Raises:
        无显式抛出，但会捕获并返回超时和文件错误
    """
    print("运行命令:", command)
    #定义危险操作，默认不允许
    dangerous_operations = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous_operations):
        return "危险操作，不允许执行"
    try:
        r = subprocess.run(command,shell= True, cwd=os.getcwd(),
                        capture_output=True, text=True,timeout=60)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "无输出"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"



llm_with_tools = llm.bind_tools([run_bash])


def agent_loop(messages: list) -> str:
    """
    执行 Agent 循环，处理 LLM 响应和工具调用

    Args:
        messages: 消息历史列表

    Returns:
        最终的 LLM 响应内容
    """
    while True:
        response = llm_with_tools.invoke(messages)
        # 如果没有工具调用，直接返回响应
        if not response.tool_calls:
            messages.append(AIMessage(content=response.content))
            return response.content

        # 添加包含 tool_calls 的 assistant 消息
        messages.append(AIMessage(
            content=response.content,
            tool_calls=response.tool_calls
        ))


        tool_messages = []

        # 处理工具调用
        for tool_call in response.tool_calls:
            print(f"\n> 运行工具: {tool_call['name']}")
            tool_output = run_bash.invoke(tool_call['args'])

            # 添加工具结果消息
            tool_messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call['id']
            ))

        # 添加所有工具结果消息
        messages.extend(tool_messages)










if __name__ == '__main__':
    history = []
    while True:
    #获取输入
        user_input = input()
        if user_input == 'exit':
             print('bye')
             exit()
        history.append({"role": "user", "content": user_input})
        response = agent_loop(history)
        # 获取最后一条 AI 消息的内容
        last_message = history[-1]
        if isinstance(last_message, AIMessage):
            print(last_message.content)
        elif isinstance(last_message, dict) and "content" in last_message:
            print(last_message["content"])
        print()




