import os
import json
import requests
import vlc
import time
from urllib.parse import unquote
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()  # 可从 .env 读取 DASHSCOPE_API_KEY
os.add_dll_directory(r"D:\VLC")
# ---- LangChain 基础 ----
from langchain_community.chat_models import ChatTongyi  # 通义千问（DashScope）在 LangChain 的封装
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings  # 通义千问的向量化模型
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes

# ------------------ 加载文档 ------------------
loader = PyPDFLoader("入职指引.pdf")  # 替换为你的 PDF 文件路径
documents = loader.load_and_split()

# ------------------ 文档向量化存储 ------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文档分割成 1000 字符的块
    chunk_overlap=200,  # 每个块之间重叠 200 字符
)
split_documents = text_splitter.split_documents(documents)  # 分割文档
# 使用 QwenEmbeddings 进行向量化
embeddings = DashScopeEmbeddings()
# 创建 FAISS 向量存储 本地存储
vectorstore = FAISS.from_documents(split_documents, embeddings)
# 保存向量存储到本地
vectorstore.save_local("vector_store")  # 保存到本地目录 vector_store

# ------------------ 天气工具类 ------------------
class WeatherTool:
    """天气查询工具类，使用和风天气 API 获取城市天气。"""
    def __init__(self):
        self.api_key = os.getenv("HE_WEATHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("未发现 HE_WEATHER_API_KEY, 请先配置和风天气 API Key。")
    
    """获取城市编码"""
    def _get_city_code(self, city: str) -> str:
        """根据城市名称获取城市编码。"""
        url = f"https://ma7aaq5d6y.re.qweatherapi.com/geo/v2/city/lookup?location={city}&key={self.api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"获取城市编码失败，状态码：{response.status_code}")
        data = response.json()
        if "location" not in data or not data["location"]:
            raise ValueError(f"未找到城市：{city} 的编码信息。")
        return data["location"][0]["id"]  # 返回第一个匹配的城市编码
    
    def get_weather_by_city(self, city: str) -> str:
        """查询指定城市的当前天气。"""
        data = json.loads(city)  # 尝试解析 JSON 格式的城市名
        city = data.get("city", city)  # 获取城市名，如果没有则使用原始输入
        if not city:
            return "请输入城市名称。"
        try:
            city_code = self._get_city_code(city)  # 获取城市编码
            url = f"https://ma7aaq5d6y.re.qweatherapi.com/v7/weather/now?location={city_code}&key={self.api_key}"
            response = requests.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"天气查询失败，状态码：{response.status_code}")
            data = response.json()
            if "now" not in data:
                raise ValueError(f"未找到城市：{city} 的天气信息。")
            weather = data["now"]
            return f"天气：{weather.get('text','未知')}，温度：{weather.get('temp','未知')}°C, 风速：{weather.get('windSpeed','未知')}km/h, 湿度：{weather.get('humidity','未知')}%\n"
        except Exception as e:
            return f"查询天气失败：{str(e)}。请检查城市名称是否正确或网络连接是否正常。"
    

# ------------------ 音乐播放工具类 ------------------
class MusicPlayerTool:
    
    def __init__(self):
        # 初始化音乐播放器相关设置

        self.instance = None  # VLC 实例
        self.playList = None  # 媒体列表
        self.player = None
        self.music_folder = r"D:\Music"
        self.current_track = None  # 当前播放的音乐

    def open_music_player(self):
        """打开音乐播放器并打印音乐目录和加载的音乐列表"""
        self.instance = vlc.Instance()  # 创建 VLC 实例
        self.playList = self.instance.media_list_new()  # 创建媒体列表
        self.player = self.instance.media_list_player_new()  # 创建媒体播放器

        if not os.path.exists(self.music_folder):
            raise FileNotFoundError(f"音乐目录不存在：{self.music_folder}")
 
        supported_extensions = (".mp3", ".wav", ".wma", ".flac", ".aac")
 
        for root, _, files in os.walk(self.music_folder):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        self.playList.add_media(self.instance.media_new(file_path))  # 添加媒体到列表
                    except Exception as e:
                        print(f"无法添加文件: {file_path}, 错误: {e}")
        if len(self.playList) == 0:
            print("没有找到支持的音乐文件。")
            return
        
        self.player.set_media_list(self.playList)  # 设置媒体列表到播放器

        print("\n加载的音乐列表：")
        for idx in range(len(self.playList)):
            media = self.playList.item_at_index(idx)
            if media:
                print(f"{idx + 1}. {unquote(media.get_mrl()).split('/')[-1]}")  # 打印音乐文件名
        return "音乐播放器已打开。请使用 '播放《歌曲名》' 来播放音乐。"

    def play_music(self, track_name: str) -> str:
        """播放指定音乐。"""    
        for i in range(len(self.playList)):
            name = unquote(self.playList.item_at_index(i).get_mrl().split('/')[-1].split('.')[0])  # 获取音乐文件名（不含扩展名）
            if name.lower() == track_name.lower():  # 忽略大小写
                self.player.play_item_at_index(i)  # 播放指定索引的音乐
                self.current_track = i
                self.player.play()
                return f"正在播放音乐：{track_name}"
        return f"未找到音乐：{track_name}。请检查音乐目录或音乐名称是否正确。"

    def quit_music_player(self) -> str:
        """退出音乐播放器。"""
        self.player.stop()
        self.player.release()  # 释放播放器
        self.playList.release()
        self.instance.release()  # 释放 VLC 实例
        self.playList = None
        return "音乐播放器已退出。"
        

    def pause_music(self) -> str:
        """暂停当前音乐播放。"""
        self.player.pause()
        return "音乐已暂停。"

    def resume_music(self) -> str:
        """恢复当前音乐播放。"""
        self.player.play()
        return "音乐已恢复播放。"

    def previous_track(self) -> str:
        """播放上一首音乐。"""
        if self.current_track is None or self.current_track == 0:
            self.current_track = len(self.playList) - 1  # 循环到最后一首
        else:
            self.current_track -= 1
        self.player.play_item_at_index(self.current_track)  # 播放上一首音乐
        self.player.play()
        return f"正在播放音乐：{unquote(self.playList.item_at_index(self.current_track).get_mrl().split('/')[-1].split('.')[0])}"

    def next_track(self) -> str:
        """播放下一首音乐。"""
        if self.current_track is None or self.current_track == len(self.playList) - 1:
            self.current_track = 0  # 循环到第一首
        else:
            self.current_track += 1
        self.player.play_item_at_index(self.current_track)  # 播放下一首音乐
        self.player.play()
        return f"正在播放音乐：{unquote(self.playList.item_at_index(self.current_track).get_mrl().split('/')[-1].split('.')[0])}"

    def music_control(self, action: str) -> str:
        """执行音乐播放器操作。"""
        data = json.loads(action)
        action = data.get("action", action)  # 获取操作指令
        action = action.strip().lower()
        response = ""
        if action == "打开播放器":
            # 如果音乐起实例已存在
            if self.instance is not None and self.playList is not None:
                return "播放器正在运行。"
            response = self.open_music_player()
        elif action.startswith("播放"):
            if self.instance is None or self.playList is None:
                return "请先打开音乐播放器。"
            track_name = action[3:-1].strip()
            if not track_name:
                return "请提供要播放的音乐名称。"
            response = self.play_music(track_name)
        elif action == "暂停":
            response = self.pause_music()

        elif action == "取消暂停":
            response = self.resume_music()

        elif action == "上一首":
            response = self.previous_track()

        elif action == "下一首":
            response = self.next_track()

        elif action == "退出":
            if self.instance is None or self.playList is None:
                return "请先打开音乐播放器。"
            response = self.quit_music_player()

        else:
            return f"未知音乐操作：{action}。请使用 '打开播放器'、'播放<歌曲名>'、'暂停'、'取消暂停'、'上一首'、 '下一首' 或 '退出'。\n"
        return f"已执行音乐操作, {response}。\n"


# ------------------ 音量控制工具类 ------------------
class VolumeControlTool:
    """音量控制工具类，使用 Pycaw 控制系统音量。"""
    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = ctypes.cast(self.interface, ctypes.POINTER(IAudioEndpointVolume))

    def _set_volume(self, level: float):
        """设置音量，范围 0.0 到 1.0。"""
        if not (0.0 <= level <= 1.0):
            raise ValueError("音量级别必须在 0.0 到 1.0 之间。")
        self.volume.SetMasterVolumeLevelScalar(level, None)

    def _get_volume(self) -> float:
        """获取当前音量，范围 0.0 到 1.0。"""
        return self.volume.GetMasterVolumeLevelScalar()

    def _mute(self):
        """静音。"""
        self.volume.SetMute(1, None)

    def _unmute(self):
        """取消静音。"""
        self.volume.SetMute(0, None)        

# ------------------ 音量控制工具函数 ------------------
    def volume_control(self, action: str) -> str:
        """执行音量控制操作。"""
        data = json.loads(action)
        action = data.get("action", action)  # 获取操作指令
        action = action.strip().lower() 
        volume = self._get_volume()
        if action == "增大音量":
            volume = min(1.0, volume + 0.1)
            self._set_volume(volume)
        elif action == "减小音量":
            volume = max(0.0, volume - 0.1)
            self._set_volume(volume)
        elif action == "静音":
            self._mute() 
        elif action == "取消静音":
            self._unmute()
        else:
            return f"未知音量操作：{action}。请使用 '增大音量'、'减小音量'、'静音' 或 '取消静音'。\n" 
        return f"已执行音量操作：{action}。\n"


#创建RAG工具类
class RAGTool:
    """RAG（Retrieval-Augmented Generation）工具类，使用 LangChain 进行检索式问答。"""
    def __init__(self, vector_store_path: str):
        self.embeddings = DashScopeEmbeddings()  # 使用通义千问的向量化模型
        self.vector_store = FAISS.load_local(
            vector_store_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatTongyi(model="qwen-plus"),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
        )

    def rag_query(self, query: str) -> Dict[str, Any]:
        """执行 RAG 查询，返回答案和相关文档。"""
        result = self.qa_chain.invoke({"query": query})
        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }

# ------------------ 构建通义千问 LLM ------------------
def build_llm(model_name: str = "qwen-plus"):
    """
    ChatTongyi 是通义千问 (DashScope) 的 LangChain 封装。
    需要环境变量 DASHSCOPE_API_KEY。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未发现 DASHSCOPE_API_KEY, 请先配置 DashScope API Key。")
    llm = ChatTongyi(model=model_name)  # 可换成 "qwen-turbo" / "qwen-max" / "qwen2.5-7b-instruct" 等
    return llm

# ------------------ 组装 Agent ------------------
def build_agent(model_name: str = "qwen-plus", verbose: bool = True):
    llm = build_llm(model_name)

    weather_tool = Tool(
        name="get_weather",
        func=WeatherTool().get_weather_by_city,
        description=(
            "查询指定城市的当前天气。"
            "输入参数：城市名（中文或英文），例如：'上海'、'Beijing'、'San Francisco'。"
            "返回为天气描述。"
        ),
    )
    volume_tool = Tool(
        name="volume_control",
        func=VolumeControlTool().volume_control,
        description=(
            "控制系统音量。"
            "输入参数：音量操作（'增大音量'、'减小音量'、'静音'、'取消静音'）。"
            "返回操作结果的描述。"
        ),
    )

    music_tool = Tool(
        name="music_player",
        func=MusicPlayerTool().music_control,
        description=(
            "打开音乐播放器，列出音乐目录下的所有音乐文件。"
            "输入参数： 播放控制（'打开播放器'、'播放《歌曲名》'、 '暂停' 、'取消暂停'、'上一首'、'下一首'、 '退出'）。"
            "返回操作描述。"
        ),
    )

    rag_tool = Tool(
        name="rag_query",
        func=RAGTool(vector_store_path="vector_store").rag_query,
        description=(
            "执行 RAG（检索增强生成）查询。"
            "输入参数：查询字符串，例如：'什么是量子计算？'。"
            "返回答案和相关文档内容。"
        ),
    )

    tools = [weather_tool, volume_tool, music_tool, rag_tool]


    # 使用 ConversationBufferWindowMemory 进行对话记忆
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,  # 保留最近 10 条对话
        return_messages=False,  # 返回消息格式
        output_key="output",  # 输出键
        input_key="input",  # 输入键
    )

    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
        template=(
            "你是一个智能助手，能够回答用户的问题并调用工具执行操作。"
            "你可以使用以下工具：\n"
            "{tools}\n"

            "你必须严格遵守以下工作流程，不能直接回答用户的问题：\n"

            "Thought: 分析用户输入以及历史对话{chat_history},决定是否需要调用工具\n"
            "Action: 选择一个工具,该工具必须包含在{tool_names}中\n"
            "Action Input: 提供工具所需的输入参数,必须为JSON格式\n"
            "Observation: 工具执行后的输出结果，由系统自动添加到{agent_scratchpad}中，你不应主动生成这部分内容。\n"
            "Thought: 基于工具返回的结果, 判断是否可以回答用户问题，或者是进一步调用工具\n"
            "Final Answer: 给出最终答案。\n"
    
            "对话历史\n"
            "{chat_history}\n"

            "开始处理用户输入：\n"
            "{input}\n"
            "{agent_scratchpad}\n"
        )
    )
    # 创建 ReAct Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt= prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,  # 是否打印详细日志
        return_intermediate_steps=True,  # 返回中间步骤
        handle_parsing_errors=True,
        temperature=0,  # 低温度更确定
    )
    # 返回 AgentExecutor 实例
    return agent_executor


def pretty_run(agent_executor, question: str):
    print("\n用户：", question)
    resp = agent_executor.invoke({
        "input": question,
    })
    print("\n智能助手：", resp["output"])

if __name__ == "__main__":
    agent_executor = build_agent(model_name="qwen-turbo", verbose=True)
    while True:
        q = input("\n请输入问题(输入 'exit' 退出）：").strip() 
        if q.lower() in {"exit", "quit"}:
            break
        pretty_run(agent_executor, q)
