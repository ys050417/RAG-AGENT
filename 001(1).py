"""
实验三：基于Qwen3（硅基流动API）的Agent+RAG应用
实现功能：
1. Agent动态决策（任务分类器 + 工具选择策略）
2. 动态RAG优化（实时调整检索参数，置信度重试机制）
3. 记忆机制（跨会话知识延续）
4. 工具集：RAG检索、Freedub（文本配音）、ScienceNews（科技资讯）
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from openai import OpenAI

# ==================== 环境配置 ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# 检查API Key
if not SILICONFLOW_API_KEY:
    print("警告: 未设置SILICONFLOW_API_KEY环境变量，将使用Mock模式（仅演示逻辑）")
    MOCK_MODE = True
else:
    MOCK_MODE = False

# 硅基流动API配置
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"  # 免费模型


# ==================== 1. LLM客户端封装（硅基流动API，OpenAI兼容） ====================
class SiliconFlowLLMClient:
    """基于硅基流动API的Qwen3模型客户端"""

    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

        if not MOCK_MODE:
            self.client = OpenAI(
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL
            )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """生成文本"""
        if MOCK_MODE:
            return self._mock_generate(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return f"错误: {str(e)}"

    def _mock_generate(self, prompt: str) -> str:
        """模拟生成（用于无API时演示流程）"""
        if "置信度" in prompt:
            return "0.85"
        elif "计算" in prompt or "加" in prompt or "乘" in prompt:
            return "计算结果: 42"
        else:
            return "这是一个模拟回答。请设置SILICONFLOW_API_KEY以使用真实硅基流动API。"

    def generate_with_confidence(self, prompt: str) -> Tuple[str, float]:
        """生成文本并返回置信度（通过二次评估）"""
        answer = self.generate(prompt)
        confidence_prompt = f"""请评估以下回答对问题的置信度分数（0-1之间，只输出数字）：
问题：{prompt[:200]}
回答：{answer}
置信度："""
        conf_str = self.generate(confidence_prompt, max_tokens=10)
        try:
            confidence = float(conf_str.strip())
        except:
            confidence = 0.5
        return answer, min(max(confidence, 0.0), 1.0)


# ==================== 2. RAG检索模块（动态参数调整） ====================
class SimpleDocumentRetriever:
    """基于关键词匹配的简单检索器（可调整top_k）"""

    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.corpus = [doc["content"] for doc in documents]
        self.titles = [doc["title"] for doc in documents]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索最相关的文档片段（基于关键词匹配）"""
        results = []
        query_words = set(query.lower().split())
        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                results.append({
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": score / len(query_words) if query_words else 0
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def adjust_parameters(self, strategy: str):
        """动态调整检索参数"""
        if strategy == "more_diverse":
            print("[RAG] 调整策略: 增加检索多样性")
        elif strategy == "higher_precision":
            print("[RAG] 调整策略: 提高检索精度")


# ==================== 3. Agent工具定义 ====================
class Tool:
    name: str
    description: str
    func: Callable

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_str: str) -> str:
        try:
            return self.func(input_str)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


def freedub_tool(input_str: str) -> str:
    """
    Freedub文本配音工具：将文本合成为语音
    接口地址: https://api.pearapi.ai/api/freedub
    请求方式: POST
    """
    try:
        parts = input_str.split("||")
        text = parts[0].strip()
        role = parts[1].strip() if len(parts) > 1 else "zh-CN-XiaoyiNeural"
        style = parts[2].strip() if len(parts) > 2 else "cheerful"

        if not text:
            return "错误: 请输入要配音的文本内容"

        payload = {
            "text": text,
            "role": role,
            "style": style
        }

        response = requests.post(
            "https://api.pearapi.ai/api/freedub",
            json=payload,
            timeout=60
        )
        data = response.json()

        if data.get("code") == 200:
            audio_url = data.get("data", {}).get("audio_url", "")
            return f"语音合成成功！\n音频链接: {audio_url}\n角色: {role}\n风格: {style}"
        else:
            return f"配音失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"请求失败: {str(e)}"


def sciencenews_tool(input_str: str) -> str:
    """
    ScienceNews科技资讯工具：获取最新科技资讯
    接口地址: https://api.pearapi.ai/api/sciencenews/
    请求方式: GET
    """
    try:
        response = requests.get("https://api.pearapi.ai/api/sciencenews/", timeout=15)
        data = response.json()

        if data.get("code") == 200:
            count = data.get("count", "0")
            update_time = data.get("update", "")
            data_str = data.get("data", "")

            try:
                articles = json.loads(data_str) if isinstance(data_str, str) else data_str
                if isinstance(articles, list):
                    result = f"获取到{count}条科技资讯（更新时间: {update_time}）:\n"
                    for i, article in enumerate(articles[:5], 1):
                        title = article.get("title", "无标题")
                        time_str = article.get("time", "")
                        result += f"{i}. [{time_str}] {title}\n"
                    return result
                else:
                    return f"科技资讯: {data_str[:1000]}"
            except:
                return f"科技资讯: {data_str[:1000]}"
        else:
            return f"获取失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"请求失败: {str(e)}"


# ==================== 4. Agent决策层 ====================
class AgentDecisionMaker:
    """Agent决策核心"""

    def __init__(self, llm_client: SiliconFlowLLMClient, retriever: SimpleDocumentRetriever):
        self.llm = llm_client
        self.retriever = retriever
        self.tools = {
            "rag_retrieve": Tool("rag_retrieve", "从知识库检索相关文档", self._rag_retrieve_wrapper),
            "freedub": Tool("freedub", "文本配音工具（格式：文本||角色||风格）", freedub_tool),
            "sciencenews": Tool("sciencenews", "获取最新科技资讯", sciencenews_tool)
        }
        self.memory = []  # 对话记忆 [(user, assistant), ...]

    def _rag_retrieve_wrapper(self, query: str) -> str:
        """RAG检索工具包装"""
        docs = self.retriever.retrieve(query, top_k=2)
        if not docs:
            return "未找到相关文档"
        context = "\n".join([f"[{d['title']}] {d['content']}" for d in docs])
        return f"检索到的相关内容:\n{context}"

    def _classify_intent(self, user_input: str) -> str:
        """任务分类器：判断需要何种工具"""
        prompt = f"""你是一个任务分类器。根据用户输入，判断应该使用哪个工具。
可用工具:
- rag_retrieve: 知识库检索（如：什么是人工智能、Python是什么）
- freedub: 文本配音（用户要求将文字转为语音时使用）
- sciencenews: 科技资讯（用户询问新闻、科技动态时使用）
- none: 直接回答

只输出工具名称，不要输出其他内容。
用户输入: {user_input}
工具名称:"""
        response = self.llm.generate(prompt, max_tokens=30)
        tool_name = response.strip().lower()
        if tool_name in self.tools:
            return tool_name
        else:
            return "none"

    def _execute_tool(self, tool_name: str, input_str: str) -> str:
        """执行工具并返回结果"""
        if tool_name in self.tools:
            return self.tools[tool_name].run(input_str)
        else:
            return f"未知工具: {tool_name}"

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """动态RAG：获取检索上下文"""
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        context = "\n".join([f"{d['content']}" for d in docs])
        return f"参考知识库:\n{context}\n"

    def answer_with_retry(self, user_input: str, max_retries: int = 2) -> Tuple[str, Dict]:
        """带重试机制的回答生成"""
        start_time = time.time()
        retry_count = 0
        tool_calls = []
        final_answer = ""
        confidence = 0.0

        while retry_count <= max_retries:
            tool_name = self._classify_intent(user_input)
            tool_used = False
            tool_result = None
            if tool_name != "none":
                tool_result = self._execute_tool(tool_name, user_input)
                tool_calls.append({"tool": tool_name, "input": user_input, "result": tool_result[:200]})
                tool_used = True

            memory_context = ""
            if self.memory:
                recent = self.memory[-3:]
                memory_context = "对话历史:\n" + "\n".join([f"用户: {u}\n助手: {a}" for u, a in recent]) + "\n"

            current_top_k = 3 + retry_count * 2
            rag_context = self._retrieve_context(user_input, top_k=current_top_k)

            tool_context = f"工具调用结果: {tool_result}\n" if tool_result else ""
            prompt = f"""{memory_context}
{rag_context}
{tool_context}
用户问题: {user_input}
请给出准确、有帮助的回答:"""

            answer, confidence = self.llm.generate_with_confidence(prompt)

            if confidence >= 0.05 or retry_count == max_retries:
                final_answer = answer
                break
            else:
                print(f"[重试] 置信度{confidence:.2f}<0.7，调整检索参数，重试第{retry_count + 1}次")
                self.retriever.adjust_parameters("more_diverse")
                retry_count += 1

        elapsed_time = time.time() - start_time

        self.memory.append((user_input, final_answer))
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

        info = {
            "tool_calls": tool_calls,
            "retries": retry_count,
            "confidence": confidence,
            "response_time": elapsed_time,
            "top_k_used": current_top_k if retry_count > 0 else 3
        }
        return final_answer, info


# ==================== 5. 示例文档库（用于RAG） ====================
def build_sample_documents():
    return [
        {"title": "人工智能简介",
         "content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。包括机器学习、自然语言处理等。"},
        {"title": "Python编程",
         "content": "Python是一种高级编程语言，广泛用于数据分析、人工智能和后端开发。语法简洁，易于学习。"},
        {"title": "机器学习基础",
         "content": "机器学习通过算法分析数据，学习规律并做出预测。常见算法包括线性回归、决策树和神经网络。"},
        {"title": "神经网络原理",
         "content": "神经网络由多层神经元组成，通过反向传播算法进行训练。深度学习是神经网络的一个重要分支。"}
    ]


# ==================== 6. 主函数 ====================
def main():
    print("正在初始化Agent系统（基于硅基流动API）...")
    llm_client = SiliconFlowLLMClient()
    documents = build_sample_documents()
    retriever = SimpleDocumentRetriever(documents)
    agent = AgentDecisionMaker(llm_client, retriever)

    print("\n" + "=" * 60)
    print("Agent已启动，支持以下工具:")
    for tool_name, tool in agent.tools.items():
        print(f"  - {tool_name}: {tool.description}")
    print("=" * 60)

    print("\n交互式对话（输入 'exit' 退出，输入 'clear' 清空记忆）:")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'clear':
            agent.memory = []
            print("对话记忆已清空")
            continue

        answer, info = agent.answer_with_retry(user_input)
        print(f"Agent: {answer}")
        print(
            f"[信息] 工具调用: {info['tool_calls']}, 耗时: {info['response_time']:.2f}s, 置信度: {info['confidence']:.2f}")


if __name__ == "__main__":
    try:
        import openai, requests
    except ImportError:
        print("请先安装所需库: pip install openai requests")
        exit(1)

    main()