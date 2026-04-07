"""
简单claude code实现
上下文压缩
第一层压缩micro_compact
每次 LLM 调用前, 将旧的 tool result 替换为占位符。
"""
import os
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
import dotenv
import yaml
from langchain.chat_models import init_chat_model
import subprocess

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage, system

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_requests.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
WORKDIR = Path.cwd()
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."
THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}
api_key = os.getenv('DEEPSEEK_API_KEY')
api_url = os.getenv('DEEPSEEK_API_URL')
MODEL_ID = os.getenv('MODULE_ID')
llm = init_chat_model(
    model=MODEL_ID,
    api_key=api_key,
    base_url=api_url
)
SKILLS_DIR = WORKDIR / "skills"
def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    """
    压缩旧的 ToolMessage 内容，保留最近 KEEP_RECENT 个
    """
    # Collect (msg_index, tool_msg) for all ToolMessage entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_results.append((msg_idx, msg))

    if len(tool_results) <= KEEP_RECENT:
        return messages

    # Find tool_name for each result by matching tool_call_id in prior assistant messages
    tool_name_map = {}  # tool_call_id -> tool_name
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_name_map[tc.get("id")] = tc.get("name")

    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for msg_idx, tool_msg in to_clear:
        content = tool_msg.content

        # 只处理长文本内容
        if not isinstance(content, str) or len(content) <= 100:
            continue

        # 获取工具名称
        tool_id = getattr(tool_msg, "tool_call_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")

        # 保留特定工具的结果（如 read_file，避免重复读取文件）
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        # 替换内容为占位符
        tool_msg.content = f"[Previous {tool_name} result: {len(content)} chars]"

    return messages

# -- auto_compact**: token 超过阈值时, 保存完整对话到磁盘, 让 LLM 做摘要。
def auto_compact(messages: list) -> list:
    """
    自动压缩上下文
    :param messages:
    :return: compressed messages
    """
    # Save transcript for recovery
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    #创建一个摘要大模型，将对话压缩保存到本地
    summary_llm = init_chat_model(
        model=MODEL_ID,
        api_key=api_key,
        base_url=api_url
    )
    summary = summary_llm.invoke([HumanMessage(content="Summarize this conversation for continuity..."
            + json.dumps(messages, default=str)[:80000])])
    return HumanMessage(content=summary.content)

# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)
# Layer 1: skill metadata injected into system prompt
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """更新待办事项列表（内部方法，非工具）"""
        validated, in_progress_count = [], 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item["id"], "text": item["text"],
                              "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        print(items)
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


@tool("todo")
def todo_tool(items: list) -> str:
    """
    更新待办事项列表

    Args:
        items: 待办事项列表，每项包含 id、text 和 status (pending/in_progress/completed)

    Returns:
        更新后的待办事项列表字符串

    Example:
        items = [
            {"id": "1", "text": "读取文件", "status": "in_progress"},
            {"id": "2", "text": "修改代码", "status": "pending"}
        ]
    """
    return TODO.update(items)

@tool(description="Load specialized knowledge by name.")
def load_skill(name:str):
    SkillLoader.get_content(name)

def log_messages(messages: list, prefix: str = ""):
    """记录消息列表到日志"""
    log_data = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            log_data.append(
                {"role": "system", "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content})
        elif isinstance(msg, HumanMessage):
            log_data.append(
                {"role": "user", "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content})
        elif isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": msg.content}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            log_data.append(entry)
        elif isinstance(msg, ToolMessage):
            log_data.append(
                {"role": "tool", "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                 "tool_call_id": msg.tool_call_id})
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            log_data.append({"role": msg.get("role", "unknown"), "content": content})

    logger.info(f"{prefix}\n{json.dumps(log_data, ensure_ascii=False, indent=2)}")


@tool
def run_bash(command: str) -> str:
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
    # 定义危险操作，默认不允许
    dangerous_operations = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous_operations):
        return "危险操作，不允许执行"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "无输出"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# 路径沙箱防止逃逸工作区
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


def subagent_loop(messages: list) -> str:

    system_message = SystemMessage(content=SUBAGENT_SYSTEM)
    messages.append(system_message)
    sub_llm = init_chat_model(
        model=MODEL_ID,
        api_key=api_key,
        base_url=api_url
    )
    subllm_with_tool =sub_llm.bind_tools([run_bash, run_read, run_write, run_edit, todo_tool,load_skill])
    for _ in range(30):
        micro_compact(messages)
        response = subllm_with_tool.invoke(messages)
        # 记录响应
        response_data = {
            "content": response.content,
            "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else [],
            "usage": response.usage_metadata if hasattr(response, 'usage_metadata') else None
        }
        logger.info(f"=== SUB_LLM Response ===\n{json.dumps(response_data, ensure_ascii=False, indent=2)}")

        if not response.tool_calls:
            return response.content
        messages.append(AIMessage(
            content=response.content,
            tool_calls=response.tool_calls
        ))
        tool_messages = []
        for tool_call in response.tool_calls:
            print(f"\n>sub_agent 运行工具: {tool_call['name']}")
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


@tool(description="Spawn a subagent with fresh context. It shares the filesystem but not conversation history.")
def task(messages: list) -> str:
    return subagent_loop(messages)


@tool(description="Trigger manual conversation compression.")
def compact():
    return


TOOL_HANDLERS = {
    "run_bash": run_bash,
    "run_read": run_read,
    "run_write": run_write,
    "run_edit": run_edit,
    "todo": todo_tool,
    "task": task
}

llm_with_tools = llm.bind_tools([run_bash, run_read, run_write, run_edit, todo_tool, task,load_skill,compact])


# 添加系统提示词

def agent_loop(messages: list) -> str:
    """
    执行 Agent 循环，处理 LLM 响应和工具调用

    Args:
        messages: 消息历史列表

    Returns:
        最终的 LLM 响应内容
    """
    system_message = SystemMessage(content=SYSTEM)
    messages.append(system_message)

    rounds_since_todo = 0
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        # 记录请求
        log_messages(messages, prefix="=== LLM Request ===")

        response = llm_with_tools.invoke(messages)

        # 记录响应
        response_data = {
            "content": response.content,
            "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else [],
            "usage": response.usage_metadata if hasattr(response, 'usage_metadata') else None
        }
        logger.info(f"=== LLM Response ===\n{json.dumps(response_data, ensure_ascii=False, indent=2)}")
        # 如果没有工具调用，直接返回响应
        if not response.tool_calls:
            return response.content

        # 添加包含 tool_calls 的 assistant 消息
        messages.append(AIMessage(
            content=response.content,
            tool_calls=response.tool_calls
        ))

        tool_messages = []

        # 处理工具调用
        used_todo = False
        manual_compact = False
        for tool_call in response.tool_calls:
            print(f"\n> 运行工具: {tool_call['name']}")
            handler = TOOL_HANDLERS.get(tool_call['name'])
            if(tool_call['name'] == 'compact'):
                manual_compact = True
                output = "Compressing..."
            elif(tool_call['name'] == 'task'):
                prompt = tool_call['args']['messages']
                tool_output = handler.invoke({'messages': prompt})
                tool_messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                ))
            else:
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


        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == '__main__':
    history = []
    while True:
        # 获取输入
        user_input = input("请输入: ")
        if user_input == 'exit':
            print('bye')
            exit()

        history.append(HumanMessage(content=user_input))
        response = agent_loop(history)
        # 获取最后一条 AI 消息的内容
        last_message = history[-1]
        if isinstance(last_message, AIMessage):
            print(last_message.content)
        elif isinstance(last_message, dict) and "content" in last_message:
            print(last_message["content"])
        print()
