"""
简单claude code实现
"""
import os

from pathlib import Path
import dotenv
from langchain.chat_models import init_chat_model
import subprocess

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage

dotenv.load_dotenv()
WORKDIR = Path.cwd()

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

#路径沙箱防止逃逸工作区
def safe_path(path: str) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return path

@tool
def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容

    Args:
        path: 文件路径
        limit: 读取行数限制，默认为 None

    Returns:
        文件内容字符串

    Raises:
        无显式抛出，但会捕获并返回文件错误
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

@tool
def run_write(path: str, content: str) -> str:
    """
    写入文件内容

    Args:
        path: 文件路径
        content: 文件内容字符串

    Returns:
        写入的字节数

    Raises:
        无显式抛出，但会捕获并返回文件错误
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件内容

    Args:
        path: 文件路径
        old_text: 要替换的旧文本
        new_text: 替换的新文本

    Returns:
        编辑后的文件内容字符串

    Raises:
        无显式抛出，但会捕获并返回文件错误
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "run_bash":  run_bash,
    "run_read":  run_read,
    "run_write": run_write,
    "run_edit":  run_edit,
}


llm_with_tools = llm.bind_tools([run_bash, run_read, run_write, run_edit])


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
            handler = TOOL_HANDLERS.get(tool_call['name'])
            if handler is None:
                tool_output = f"Error: Unknown tool '{tool_call['name']}'"
            else:
                tool_output = handler.invoke(tool_call['args'])
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
        user_input = input("请输入: ")
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




