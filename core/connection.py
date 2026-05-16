import os
import sys
import copy
import json
import uuid
import time
import re
import queue
import asyncio
import threading
import traceback
import subprocess
from pathlib import Path
import websockets

from core.utils.util import (
    extract_json_from_string,
    check_vad_update,
    check_asr_update,
    filter_sensitive_info,
)
from typing import Dict, Any
from collections import deque
from core.utils.modules_initialize import (
    initialize_modules,
    initialize_tts,
    initialize_asr,
)
from core.handle.reportHandle import report, enqueue_tool_report
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.handle.textHandle import handleTextMessage
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.loadplugins import auto_import_modules
from plugins_func.register import Action, ActionResponse
from core.auth import AuthenticationError
from core.clinical_safety import ClinicalSafetyInterceptor
from config.config_loader import get_private_config_from_api
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger
from config.manage_api_client import DeviceNotFoundException, DeviceBindException, generate_and_save_chat_title
from core.utils.prompt_manager import PromptManager
from core.utils.voiceprint_provider import VoiceprintProvider
from core.utils.conversation_history_store import (
    ConversationHistoryStore,
    build_session_preview,
    build_session_title,
    utc_now_iso,
)
from core.utils.util import get_system_error_response
from core.utils import textUtils
from core.utils.device_identity import normalize_device_user_id
from config.config_loader import get_project_dir


TAG = __name__

# 工具调用规则 - 用于动态注入提醒
TOOL_CALLING_RULES = """
<tool_calling>
【核心原则】你是拥有工具能力的智能助手。当用户请求需要实时信息或执行操作时，调用相应工具获取数据，禁止凭空编造答案。

- **何时必须调用工具：**
  1. 实时信息查询（新闻、非本地天气、股价、汇率等）
  2. 执行操作（播放音乐、控制设备、拍照、设置闹钟等）
  3. 知识库检索（当工具列表包含 search_clinical_rag 时，结合用户意图判断是否需要调用）
  4. 查询非今天的农历信息（明天农历、某日宜忌、节气等）
  5. 用户说"拍照"时调用 self_camera_take_photo，默认 question 参数为"描述一下看到的物品"

- **何时无需调用工具：**
  1. `<context>` 中已提供的信息（当前时间、今天日期、今天农历、本地天气等）
  2. 普通对话、问候、闲聊、情感交流、讲故事
  3. 通用知识问答（非实时信息）

- **调用规范：**
  1. 每次请求独立判断，不复用历史工具结果，需重新获取最新数据
  2. 多任务时依次调用所有需要的工具，并依次总结每个工具的结果，不得遗漏
  3. 严格遵循工具的参数要求，提供所有必要参数
  4. 不确定时引导用户澄清或告知能力限制，切勿猜测或编造
  5. 不调用未提供的工具，对话中提及的旧工具若不可用则忽略或说明

- **反偷懒机制（最高优先级）：**
  1. **每次独立判断：** 无论对话历史中是否调用过工具，当前请求必须根据当前需求独立判断是否需要调用
  2. **禁止模式模仿：** 即使之前的回复没有调用工具，也不代表本次可以不调用
  3. **自我检查：** 回复前必须自问："这个请求是否涉及实时信息或执行操作？如果是，我调用工具了吗？"
  4. **历史不等于现在：** 对话历史中的行为模式不影响当前判断，每个用户请求都是全新的开始
</tool_calling>
"""

auto_import_modules("plugins_func.functions")


class TTSException(RuntimeError):
    pass


class ConnectionHandler:
    def __init__(
            self,
            config: Dict[str, Any],
            _vad,
            _asr,
            _llm,
            _memory,
            _intent,
            server=None,
    ):
        self.common_config = config
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()
        self.server = server  # 保存server实例的引用

        self.need_bind = False  # 是否需要绑定设备
        self.bind_completed_event = asyncio.Event()
        self.bind_code = None  # 绑定设备的验证码
        self.last_bind_prompt_time = 0  # 上次播放绑定提示的时间戳(秒)
        self.bind_prompt_interval = 60  # 绑定提示播放间隔(秒)

        self.read_config_from_api = self.config.get("read_config_from_api", False)

        self.websocket: websockets.ServerConnection | None = None
        self.headers = None
        self.device_id = None
        self.user_id = None
        self.client_ip = None
        self.prompt = None
        self.welcome_msg = None
        self.max_output_size = 0
        self.chat_history_conf = 0
        self.audio_format = "opus"
        self.sample_rate = 24000  # 默认采样率，从客户端 hello 消息中动态更新

        # 客户端状态相关
        self.client_abort = False
        self.client_is_speaking = False
        self.client_listen_mode = "auto"

        # 线程任务相关
        self.loop = None  # 在 handle_connection 中获取运行中的事件循环
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)

        # 添加上报线程池
        self.report_queue = queue.Queue()
        self.report_thread = None
        # 未来可以通过修改此处，调节asr的上报和tts的上报，目前默认都开启
        self.report_asr_enable = self.read_config_from_api
        self.report_tts_enable = self.read_config_from_api

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr
        self._vad = _vad
        self.llm = _llm
        self.memory = _memory
        self.intent = _intent
        self.clinical_safety = self._build_clinical_safety_interceptor()

        self.is_exiting = False  # 标记是否正在执行退出流程

        # 为每个连接单独管理声纹识别
        self.voiceprint_provider = None

        # vad相关变量
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_window = deque(maxlen=5)
        self.first_activity_time = 0.0  # 记录首次活动的时间（毫秒）
        self.last_activity_time = 0.0  # 统一的活动时间戳（毫秒）
        self.vad_last_voice_time = 0.0  # 记录用户最后一次说话的时间（毫秒）
        self.client_voice_stop = False
        self.last_is_voice = False

        # asr相关变量
        # 因为实际部署时可能会用到公共的本地ASR，不能把变量暴露给公共ASR
        # 所以涉及到ASR的变量，需要在这里定义，属于connection的私有变量
        self.asr_audio = []
        self.asr_audio_queue = queue.Queue()
        self.current_speaker = None  # 存储当前说话人

        # llm相关变量
        self.dialogue = Dialogue()

        # 工具调用统计（用于监控和自动恢复）
        self.tool_call_stats = {
            'last_call_turn': -1,  # 上次调用工具的轮数
            'consecutive_no_call': 0,  # 连续未调用次数
        }

        # tts相关变量
        self.sentence_id = None
        # 处理TTS响应没有文本返回
        self.tts_MessageText = ""

        # iot相关变量
        self.iot_descriptors = {}
        self.func_handler = None

        self.cmd_exit = self.config["exit_commands"]

        # 是否在聊天结束后关闭连接
        self.close_after_chat = False
        self.load_function_plugin = False
        self.intent_type = "nointent"

        self.timeout_seconds = (
                int(self.config.get("close_connection_no_voice_time", 120)) + 60
        )  # 在原来第一道关闭的基础上加60秒，进行二道关闭
        self.timeout_task = None

        # {"mcp":true} 表示启用MCP功能
        self.features = None

        # 标记连接是否来自MQTT
        self.conn_from_mqtt_gateway = False

        # 初始化提示词管理器
        self.prompt_manager = PromptManager(self.config, self.logger)
        self.console_history_store = ConversationHistoryStore(
            Path(get_project_dir()) / "data" / "console_history.db"
        )

    async def handle_connection(self, ws: websockets.ServerConnection):
        try:
            # 获取运行中的事件循环（必须在异步上下文中）
            self.loop = asyncio.get_running_loop()

            # 获取并验证headers
            self.headers = dict(ws.request.headers)
            real_ip = self.headers.get("x-real-ip") or self.headers.get(
                "x-forwarded-for"
            )
            if real_ip:
                self.client_ip = real_ip.split(",")[0].strip()
            else:
                self.client_ip = ws.remote_address[0]
            self.logger.bind(tag=TAG).info(
                f"{self.client_ip} conn - Headers: {self.headers}"
            )

            self.device_id = self.headers.get("device-id", None)
            self.user_id = normalize_device_user_id(self.device_id)

            # 认证通过,继续处理
            self.websocket = ws

            # 检查是否来自MQTT连接
            request_path = ws.request.path
            self.conn_from_mqtt_gateway = request_path.endswith("?from=mqtt_gateway")
            if self.conn_from_mqtt_gateway:
                self.logger.bind(tag=TAG).info("连接来自:MQTT网关")

            # 初始化活动时间戳
            self.first_activity_time = time.time() * 1000
            self.last_activity_time = time.time() * 1000

            # 启动超时检查任务
            self.timeout_task = asyncio.create_task(self._check_timeout())

            self.welcome_msg = self.config["xiaozhi"]
            self.welcome_msg["session_id"] = self.session_id

            # 从配置中读取采样率
            self.sample_rate = self.welcome_msg["audio_params"]["sample_rate"]
            self.logger.bind(tag=TAG).info(f"配置输出音频采样率为: {self.sample_rate}")

            # 在后台初始化配置和组件（完全不阻塞主循环）
            asyncio.create_task(self._background_initialize())

            try:
                async for message in self.websocket:
                    await self._route_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.bind(tag=TAG).info("客户端断开连接")

        except AuthenticationError as e:
            self.logger.bind(tag=TAG).error(f"Authentication failed: {str(e)}")
            return
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.bind(tag=TAG).error(f"Connection error: {str(e)}-{stack_trace}")
            return
        finally:
            try:
                await self._save_and_close(ws)
            except Exception as final_error:
                self.logger.bind(tag=TAG).error(f"最终清理时出错: {final_error}")
                # 确保即使保存记忆失败，也要关闭连接
                try:
                    await self.close(ws)
                except Exception as close_error:
                    self.logger.bind(tag=TAG).error(
                        f"强制关闭连接时出错: {close_error}"
                    )

    async def _save_and_close(self, ws):
        """保存记忆并关闭连接"""
        try:
            # 守护线程1：独立生成标题（不依赖记忆模型）
            if self.session_id:
                def generate_title_task():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            generate_and_save_chat_title(self.session_id)
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"生成标题失败: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                threading.Thread(target=generate_title_task, daemon=True).start()

            try:
                self._persist_local_console_history()
            except Exception as history_error:
                self.logger.bind(tag=TAG).error(
                    f"淇濆瓨鏈湴鍘嗗彶瀵硅瘽澶辫触: {history_error}"
                )

            # 守护线程2：走老流程记忆保存（仅记忆，不含标题）
            if self.memory:
                # 使用线程池异步保存记忆
                def save_memory_task():
                    try:
                        # 创建新事件循环（避免与主循环冲突）
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.memory.save_memory(
                                self.dialogue.dialogue, self.session_id
                            )
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                # 启动线程保存记忆，不等待完成
                threading.Thread(target=save_memory_task, daemon=True).start()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
        finally:
            # 立即关闭连接，不等待记忆保存完成
            try:
                await self.close(ws)
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(
                    f"保存记忆后关闭连接失败: {close_error}"
                )

    def _persist_local_console_history(self):
        if not self.session_id or not self.user_id:
            return

        transcript = self._serialize_console_history_messages()
        if not transcript:
            return

        created_at = transcript[0].get("created_at") or utc_now_iso()
        updated_at = transcript[-1].get("created_at") or utc_now_iso()
        has_tool_calls = any(
            item.get("role") == "tool" or bool(item.get("tool_calls"))
            for item in transcript
        )
        has_vision = any(self._message_has_vision_signal(item) for item in transcript)

        self.console_history_store.upsert_session(
            session_id=self.session_id,
            user_id=self.user_id,
            device_id=self.device_id or "",
            title=build_session_title(transcript),
            preview=build_session_preview(transcript),
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(transcript),
            has_tool_calls=has_tool_calls,
            has_vision=has_vision,
            metadata={
                "saved_at": utc_now_iso(),
                "message_roles": [item.get("role", "") for item in transcript[-12:]],
            },
            transcript=transcript,
        )

    def _serialize_console_history_messages(self):
        transcript = []
        for msg in self.dialogue.dialogue:
            item = {
                "message_id": msg.uniq_id,
                "role": msg.role,
                "content": msg.content or "",
                "created_at": getattr(msg, "created_at", None) or utc_now_iso(),
            }
            if msg.tool_calls is not None:
                item["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if getattr(msg, "is_temporary", False):
                item["is_temporary"] = True
            transcript.append(item)
        return transcript

    def _message_has_vision_signal(self, item):
        keywords = ("camera", "photo", "vision", "image", "take_photo")
        raw = json.dumps(item, ensure_ascii=False)
        lowered = raw.lower()
        return any(keyword in lowered for keyword in keywords)

    async def _discard_message_with_bind_prompt(self):
        """丢弃消息并检查是否需要播放绑定提示"""
        current_time = time.time()
        # 检查是否需要播放绑定提示
        if current_time - self.last_bind_prompt_time >= self.bind_prompt_interval:
            self.last_bind_prompt_time = current_time
            # 复用现有的绑定提示逻辑
            from core.handle.receiveAudioHandle import check_bind_device

            asyncio.create_task(check_bind_device(self))

    async def _route_message(self, message):
        """消息路由"""
        # 退出状态丢弃所有消息
        if self.is_exiting:
           return

        # 检查是否已经获取到真实的绑定状态
        if not self.bind_completed_event.is_set():
            # 还没有获取到真实状态，等待直到获取到真实状态或超时
            try:
                await asyncio.wait_for(self.bind_completed_event.wait(), timeout=1)
            except asyncio.TimeoutError:
                # 超时仍未获取到真实状态，丢弃消息
                await self._discard_message_with_bind_prompt()
                return

        # 已经获取到真实状态，检查是否需要绑定
        if self.need_bind:
            # 需要绑定，丢弃消息
            await self._discard_message_with_bind_prompt()
            return

        # 不需要绑定，继续处理消息

        if isinstance(message, str):
            await handleTextMessage(self, message)
        elif isinstance(message, bytes):
            if self.vad is None or self.asr is None:
                return

            # 处理来自MQTT网关的音频包
            if self.conn_from_mqtt_gateway and len(message) >= 16:
                handled = await self._process_mqtt_audio_message(message)
                if handled:
                    return

            # 不需要头部处理或没有头部时，直接处理原始消息
            self.asr_audio_queue.put(message)

    async def _process_mqtt_audio_message(self, message):
        """
        处理来自MQTT网关的音频消息，解析16字节头部并提取音频数据

        Args:
            message: 包含头部的音频消息

        Returns:
            bool: 是否成功处理了消息
        """
        try:
            # 提取头部信息
            timestamp = int.from_bytes(message[8:12], "big")
            audio_length = int.from_bytes(message[12:16], "big")

            # 提取音频数据
            if audio_length > 0 and len(message) >= 16 + audio_length:
                # 有指定长度，提取精确的音频数据
                audio_data = message[16 : 16 + audio_length]
                # 基于时间戳进行排序处理
                self._process_websocket_audio(audio_data, timestamp)
                return True
            elif len(message) > 16:
                # 没有指定长度或长度无效，去掉头部后处理剩余数据
                audio_data = message[16:]
                self.asr_audio_queue.put(audio_data)
                return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"解析WebSocket音频包失败: {e}")

        # 处理失败，返回False表示需要继续处理
        return False

    def _process_websocket_audio(self, audio_data, timestamp):
        """处理WebSocket格式的音频包"""
        # 初始化时间戳序列管理
        if not hasattr(self, "audio_timestamp_buffer"):
            self.audio_timestamp_buffer = {}
            self.last_processed_timestamp = 0
            self.max_timestamp_buffer_size = 20

        # 如果时间戳是递增的，直接处理
        if timestamp >= self.last_processed_timestamp:
            self.asr_audio_queue.put(audio_data)
            self.last_processed_timestamp = timestamp

            # 处理缓冲区中的后续包
            processed_any = True
            while processed_any:
                processed_any = False
                for ts in sorted(self.audio_timestamp_buffer.keys()):
                    if ts > self.last_processed_timestamp:
                        buffered_audio = self.audio_timestamp_buffer.pop(ts)
                        self.asr_audio_queue.put(buffered_audio)
                        self.last_processed_timestamp = ts
                        processed_any = True
                        break
        else:
            # 乱序包，暂存
            if len(self.audio_timestamp_buffer) < self.max_timestamp_buffer_size:
                self.audio_timestamp_buffer[timestamp] = audio_data
            else:
                self.asr_audio_queue.put(audio_data)

    async def handle_restart(self, message):
        """处理服务器重启请求"""
        try:

            self.logger.bind(tag=TAG).info("收到服务器重启指令，准备执行...")

            # 发送确认响应
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "success",
                        "message": "服务器重启中...",
                        "content": {"action": "restart"},
                    }
                )
            )

            # 异步执行重启操作
            def restart_server():
                """实际执行重启的方法"""
                time.sleep(1)
                self.logger.bind(tag=TAG).info("执行服务器重启...")
                subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    start_new_session=True,
                )
                os._exit(0)

            # 使用线程执行重启避免阻塞事件循环
            threading.Thread(target=restart_server, daemon=True).start()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"重启失败: {str(e)}")
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "error",
                        "message": f"Restart failed: {str(e)}",
                        "content": {"action": "restart"},
                    }
                )
            )

    def _initialize_components(self):
        try:
            if self.tts is None:
                self.tts = self._initialize_tts()
            # 打开语音合成通道
            asyncio.run_coroutine_threadsafe(
                self.tts.open_audio_channels(self), self.loop
            )
            if self.need_bind:
                self.bind_completed_event.set()
                return
            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)

            """初始化组件"""
            if self.config.get("prompt") is not None:
                user_prompt = self.config["prompt"]
                # 使用快速提示词进行初始化
                prompt = self.prompt_manager.get_quick_prompt(user_prompt)
                self.change_system_prompt(prompt)
                self.logger.bind(tag=TAG).info(
                    f"快速初始化组件: prompt成功 {prompt[:50]}..."
                )

            """初始化本地组件"""
            if self.vad is None:
                self.vad = self._vad
            if self.asr is None:
                self.asr = self._initialize_asr()

            # 初始化声纹识别
            self._initialize_voiceprint()
            # 打开语音识别通道
            asyncio.run_coroutine_threadsafe(
                self.asr.open_audio_channels(self), self.loop
            )

            """加载记忆"""
            self._initialize_memory()
            """加载意图识别"""
            self._initialize_intent()
            """初始化上报线程"""
            self._init_report_threads()
            """更新系统提示词"""
            self._init_prompt_enhancement()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"实例化组件失败: {e}")

    def _init_prompt_enhancement(self):

        # 更新上下文信息
        self.prompt_manager.update_context_info(self, self.client_ip)
        enhanced_prompt = self.prompt_manager.build_enhanced_prompt(
            self.config["prompt"], self.device_id, self.client_ip
        )
        if enhanced_prompt:
            self.change_system_prompt(enhanced_prompt)
            self.logger.bind(tag=TAG).debug("系统提示词已增强更新")

    def _init_report_threads(self):
        """初始化ASR和TTS上报线程"""
        if not self.read_config_from_api or self.need_bind:
            return
        if self.chat_history_conf == 0:
            return
        if self.report_thread is None or not self.report_thread.is_alive():
            self.report_thread = threading.Thread(
                target=self._report_worker, daemon=True
            )
            self.report_thread.start()
            self.logger.bind(tag=TAG).info("TTS上报线程已启动")

    def _initialize_tts(self):
        """初始化TTS"""
        tts = None
        if not self.need_bind:
            tts = initialize_tts(self.config)

        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)

        return tts

    def _initialize_asr(self):
        """初始化ASR"""
        if (
                self._asr is not None
                and hasattr(self._asr, "interface_type")
                and self._asr.interface_type == InterfaceType.LOCAL
        ):
            # 如果公共ASR是本地服务，则直接返回
            # 因为本地一个实例ASR，可以被多个连接共享
            asr = self._asr
        else:
            # 如果公共ASR是远程服务，则初始化一个新实例
            # 因为远程ASR，涉及到websocket连接和接收线程，需要每个连接一个实例
            asr = initialize_asr(self.config)

        return asr

    def _initialize_voiceprint(self):
        """为当前连接初始化声纹识别"""
        try:
            voiceprint_config = self.config.get("voiceprint", {})
            if voiceprint_config:
                voiceprint_provider = VoiceprintProvider(voiceprint_config)
                if voiceprint_provider is not None and voiceprint_provider.enabled:
                    self.voiceprint_provider = voiceprint_provider
                    self.logger.bind(tag=TAG).info("声纹识别功能已在连接时动态启用")
                else:
                    self.logger.bind(tag=TAG).warning("声纹识别功能启用但配置不完整")
            else:
                self.logger.bind(tag=TAG).info("声纹识别功能未启用")
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"声纹识别初始化失败: {str(e)}")

    async def _background_initialize(self):
        """在后台初始化配置和组件（完全不阻塞主循环）"""
        try:
            # 异步获取差异化配置
            await self._initialize_private_config_async()
            # 在线程池中初始化组件
            self.executor.submit(self._initialize_components)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"后台初始化失败: {e}")

    async def _initialize_private_config_async(self):
        """从接口异步获取差异化配置（异步版本，不阻塞主循环）"""
        if not self.read_config_from_api:
            self.need_bind = False
            self.bind_completed_event.set()
            return
        try:
            begin_time = time.time()
            private_config = await get_private_config_from_api(
                self.config,
                self.headers.get("device-id"),
                self.headers.get("client-id", self.headers.get("device-id")),
            )
            private_config["delete_audio"] = bool(self.config.get("delete_audio", True))
            self.logger.bind(tag=TAG).info(
                f"{time.time() - begin_time} 秒，异步获取差异化配置成功: {json.dumps(filter_sensitive_info(private_config), ensure_ascii=False)}"
            )
            self.need_bind = False
            self.bind_completed_event.set()
        except DeviceNotFoundException as e:
            self.need_bind = True
            private_config = {}
        except DeviceBindException as e:
            self.need_bind = True
            self.bind_code = e.bind_code
            private_config = {}
        except Exception as e:
            self.need_bind = True
            self.logger.bind(tag=TAG).error(f"异步获取差异化配置失败: {e}")
            private_config = {}

        init_llm, init_tts, init_memory, init_intent = (
            False,
            False,
            False,
            False,
        )

        init_vad = check_vad_update(self.common_config, private_config)
        init_asr = check_asr_update(self.common_config, private_config)

        if init_vad:
            self.config["VAD"] = private_config["VAD"]
            self.config["selected_module"]["VAD"] = private_config["selected_module"][
                "VAD"
            ]
        if init_asr:
            self.config["ASR"] = private_config["ASR"]
            self.config["selected_module"]["ASR"] = private_config["selected_module"][
                "ASR"
            ]
        if private_config.get("TTS", None) is not None:
            init_tts = True
            self.config["TTS"] = private_config["TTS"]
            self.config["selected_module"]["TTS"] = private_config["selected_module"][
                "TTS"
            ]
        if private_config.get("LLM", None) is not None:
            init_llm = True
            self.config["LLM"] = private_config["LLM"]
            self.config["selected_module"]["LLM"] = private_config["selected_module"][
                "LLM"
            ]
        if private_config.get("VLLM", None) is not None:
            self.config["VLLM"] = private_config["VLLM"]
            self.config["selected_module"]["VLLM"] = private_config["selected_module"][
                "VLLM"
            ]
        if private_config.get("Memory", None) is not None:
            init_memory = True
            self.config["Memory"] = private_config["Memory"]
            self.config["selected_module"]["Memory"] = private_config[
                "selected_module"
            ]["Memory"]
        if private_config.get("Intent", None) is not None:
            init_intent = True
            self.config["Intent"] = private_config["Intent"]
            model_intent = private_config.get("selected_module", {}).get("Intent", {})
            self.config["selected_module"]["Intent"] = model_intent
            # 加载插件配置
            if model_intent != "Intent_nointent":
                plugin_from_server = private_config.get("plugins", {})
                for plugin, config_str in plugin_from_server.items():
                    plugin_from_server[plugin] = json.loads(config_str)
                self.config["plugins"] = plugin_from_server
                self.config["Intent"][self.config["selected_module"]["Intent"]][
                    "functions"
                ] = plugin_from_server.keys()
        if private_config.get("prompt", None) is not None:
            self.config["prompt"] = private_config["prompt"]
        # 获取声纹信息
        if private_config.get("voiceprint", None) is not None:
            self.config["voiceprint"] = private_config["voiceprint"]
        if private_config.get("summaryMemory", None) is not None:
            self.config["summaryMemory"] = private_config["summaryMemory"]
        if private_config.get("device_max_output_size", None) is not None:
            self.max_output_size = int(private_config["device_max_output_size"])
        if private_config.get("chat_history_conf", None) is not None:
            self.chat_history_conf = int(private_config["chat_history_conf"])
        if private_config.get("mcp_endpoint", None) is not None:
            self.config["mcp_endpoint"] = private_config["mcp_endpoint"]
        if private_config.get("context_providers", None) is not None:
            self.config["context_providers"] = private_config["context_providers"]

        # 使用 run_in_executor 在线程池中执行 initialize_modules，避免阻塞主循环
        try:
            modules = await self.loop.run_in_executor(
                None,  # 使用默认线程池
                initialize_modules,
                self.logger,
                private_config,
                init_vad,
                init_asr,
                init_llm,
                init_tts,
                init_memory,
                init_intent,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"初始化组件失败: {e}")
            modules = {}
        if modules.get("tts", None) is not None:
            self.tts = modules["tts"]
        if modules.get("vad", None) is not None:
            self.vad = modules["vad"]
        if modules.get("asr", None) is not None:
            self.asr = modules["asr"]
        if modules.get("llm", None) is not None:
            self.llm = modules["llm"]
        if modules.get("intent", None) is not None:
            self.intent = modules["intent"]
        if modules.get("memory", None) is not None:
            self.memory = modules["memory"]

    def _initialize_memory(self):
        if self.memory is None:
            return
        """初始化记忆模块"""
        self.memory.init_memory(
            role_id=self.user_id or normalize_device_user_id(self.device_id),
            llm=self.llm,
            summary_memory=self.config.get("summaryMemory", None),
            save_to_file=not self.read_config_from_api,
        )

        # 获取记忆总结配置
        memory_config = self.config["Memory"]
        selected_memory_name = self.config["selected_module"]["Memory"]
        selected_memory_config = memory_config[selected_memory_name]
        memory_type = selected_memory_config["type"]

        # 如果使用 nomem 或 mem_report_only，直接返回
        if memory_type == "nomem" or memory_type == "mem_report_only":
            return

        memory_llm_name = selected_memory_config.get("llm")
        if memory_llm_name and memory_llm_name in self.config["LLM"]:
            # 如果配置了专用LLM，则创建独立的LLM实例
            from core.utils import llm as llm_utils

            memory_llm_config = self.config["LLM"][memory_llm_name]
            memory_llm_type = memory_llm_config.get("type", memory_llm_name)
            memory_llm = llm_utils.create_instance(
                memory_llm_type, memory_llm_config
            )
            self.logger.bind(tag=TAG).info(
                f"为记忆模块创建了专用LLM: {memory_llm_name}, 类型: {memory_llm_type}"
            )
            self.memory.set_llm(memory_llm)
        else:
            # 否则使用主LLM
            self.memory.set_llm(self.llm)
            self.logger.bind(tag=TAG).info("使用主LLM作为记忆模型")

    def _initialize_intent(self):
        if self.intent is None:
            return
        self.intent_type = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ]["type"]
        if self.intent_type == "function_call" or self.intent_type == "intent_llm":
            self.load_function_plugin = True
        """初始化意图识别模块"""
        # 获取意图识别配置
        intent_config = self.config["Intent"]
        intent_type = self.config["Intent"][self.config["selected_module"]["Intent"]][
            "type"
        ]

        # 如果使用 nointent，直接返回
        if intent_type == "nointent":
            return
        # 使用 intent_llm 模式
        elif intent_type == "intent_llm":
            intent_llm_name = intent_config[self.config["selected_module"]["Intent"]][
                "llm"
            ]

            if intent_llm_name and intent_llm_name in self.config["LLM"]:
                # 如果配置了专用LLM，则创建独立的LLM实例
                from core.utils import llm as llm_utils

                intent_llm_config = self.config["LLM"][intent_llm_name]
                intent_llm_type = intent_llm_config.get("type", intent_llm_name)
                intent_llm = llm_utils.create_instance(
                    intent_llm_type, intent_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"为意图识别创建了专用LLM: {intent_llm_name}, 类型: {intent_llm_type}"
                )
                self.intent.set_llm(intent_llm)
            else:
                # 否则使用主LLM
                self.intent.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("使用主LLM作为意图识别模型")

        """加载统一工具处理器"""
        self.func_handler = UnifiedToolHandler(self)

        # 异步初始化工具处理器
        if hasattr(self, "loop") and self.loop:
            asyncio.run_coroutine_threadsafe(self.func_handler._initialize(), self.loop)

    def _build_clinical_safety_interceptor(self):
        if not self.config.get("clinical_safety_rules_enabled", True):
            return None

        rules_path = self.config.get("clinical_safety_rules_path")
        if not rules_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(
                project_root,
                "knowledge_base",
                "rules",
                "clinical_safety_rules.json",
            )

        try:
            interceptor = ClinicalSafetyInterceptor(rules_path)
            self.logger.bind(tag=TAG).info(f"临床安全规则拦截器初始化成功: {rules_path}")
            return interceptor
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(f"临床安全规则拦截器初始化失败: {exc}")
            return None

    def change_system_prompt(self, prompt):
        self.prompt = prompt
        # 更新系统prompt至上下文
        self.dialogue.update_system_message(self.prompt)

    def chat(self, query, depth=0):
        # 保存当前任务的sentence_id到局部变量，避免被新任务覆盖
        current_sentence_id = None

        if query is not None:
            self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")

        # 为最顶层时新建会话ID和发送FIRST请求
        if depth == 0:
            current_sentence_id = str(uuid.uuid4().hex)
            self.sentence_id = current_sentence_id  # 更新共享属性
            self.dialogue.put(Message(role="user", content=query))
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            if self._maybe_resolve_health_profile_voice_review(query, current_sentence_id):
                return True
            health_profile_stats = self._maybe_refresh_health_profile()
            if self._maybe_reply_health_profile_voice_review(health_profile_stats, current_sentence_id):
                return True

        else:
            # 递归调用时，使用当前的sentence_id
            current_sentence_id = self.sentence_id

        # 设置最大递归深度，避免无限循环，可根据实际需求调整
        MAX_DEPTH = 5
        force_final_answer = False  # 标记是否强制最终回答

        if depth >= MAX_DEPTH:
            self.logger.bind(tag=TAG).debug(
                f"已达到最大工具调用深度 {MAX_DEPTH}，将强制基于现有信息回答"
            )
            force_final_answer = True
            # 添加系统指令，要求 LLM 基于现有信息回答
            self.dialogue.put(
                Message(
                    role="user",
                    content="[系统提示] 已达到最大工具调用次数限制，请你基于目前已经获取的所有信息，直接给出最终答案。不要再尝试调用任何工具。",
                )
            )

        # 长对话工具调用提醒：当对话轮数较多时，提醒模型正确使用工具
        force_reminder = False  # 是否强制提醒

        if depth == 0 and query is not None:
            dialogue_length = len(self.dialogue.dialogue)
            current_turn = dialogue_length // 2

            # 检测距离上一次连续未调用工具的情况
            if self.tool_call_stats['last_call_turn'] >= 0:
                turns_since_last = current_turn - self.tool_call_stats['last_call_turn']
                if turns_since_last > 3:  # 超过3轮未调用
                    self.logger.bind(tag=TAG).warning(
                        f"检测到{turns_since_last}轮未调用工具，可能进入偷懒模式，将强制注入提醒"
                    )
                    force_reminder = True

            # 对话历史截断：防止历史过长导致模型"偷懒模式"扩散
            # 当对话历史超过阈值时，保留最近的 10 轮对话
            # max_dialogue_turns = 10
            # if dialogue_length > max_dialogue_turns * 2:
            #     removed = self.dialogue.trim_history(max_turns=max_dialogue_turns)
            #     if removed > 0:
            #         self.logger.bind(tag=TAG).info(
            #             f"对话历史过长({dialogue_length}条)，已智能截断保留最近{max_dialogue_turns}轮，移除{removed}条消息"
            #         )
            self._maybe_compact_short_term_memory()

        # Define intent functions
        functions = None
        # 达到最大深度时，禁用工具调用，强制 LLM 直接回答
        if (
                self.intent_type == "function_call"
                and hasattr(self, "func_handler")
                and not force_final_answer
        ):
            functions = self.func_handler.get_functions()

        if (
                depth == 0
                and query is not None
                and functions is not None
                and self._should_auto_trigger_camera(query)
        ):
            self.logger.bind(tag=TAG).info(
                f"检测到视觉指代问题，自动触发拍照工具: {query}"
            )
            if self._invoke_direct_tool_call(
                    "self_camera_take_photo",
                    {"question": self._build_camera_question(query)},
                    depth=depth,
            ):
                return True

        # 长对话工具调用规则强化：动态生成基于当前可用工具的提醒
        tool_call_reminder = None
        if depth == 0 and query is not None and functions is not None:
            dialogue_length = len(self.dialogue.dialogue)
            # 当对话历史超过4条消息时，注入规则强化
            if dialogue_length > 4:
                tool_summary = self._get_tool_summary(functions)
                if tool_summary:
                    # 根据对话长度和偷懒检测，使用不同强度的提醒
                    if force_reminder:
                        tool_call_reminder = self._build_tool_call_reminder(
                            tool_summary,
                            force_reminder=True,
                        )
                        reminder_level = "强"
                    else:
                        tool_call_reminder = self._build_tool_call_reminder(
                            tool_summary,
                            force_reminder=False,
                        )
                        reminder_level = "中"
                    self.logger.bind(tag=TAG).debug(
                        f"对话历史较长({dialogue_length}条)，已注入{reminder_level}等级工具调用规则强化，当前可用工具：{tool_summary}"
                    )

        response_message = []

        # 如果有工具调用提醒，临时添加到对话中（标记为临时消息）
        if tool_call_reminder:
            self.dialogue.put(Message(role="user", content=tool_call_reminder, is_temporary=True))

        try:
            # 使用带记忆的对话
            memory_str = None
            dialogue_messages = [
                msg for msg in self.dialogue.dialogue
                if not getattr(msg, "is_temporary", False)
            ]
            # 仅当query非空（代表用户询问）时查询记忆
            if self.memory is not None and query:
                future = asyncio.run_coroutine_threadsafe(
                    self.memory.build_memory_context(
                        query,
                        dialogue_messages=dialogue_messages,
                        session_id=self.session_id,
                    ),
                    self.loop,
                )
                memory_str = future.result()

            if depth == 0 and query and self.clinical_safety is not None:
                health_profile = None
                if (
                    getattr(self, "memory", None) is not None
                    and getattr(self.memory, "health_profile_store", None) is not None
                    and getattr(self.memory, "role_id", None)
                ):
                    try:
                        profile_future = asyncio.run_coroutine_threadsafe(
                            self.memory.health_profile_store.get_profile(self.memory.role_id),
                            self.loop,
                        )
                        health_profile = profile_future.result()
                    except Exception as profile_exc:
                        self.logger.bind(tag=TAG).warning(
                            f"结构化健康档案读取失败，临床安全将退回文本抽取: {profile_exc}"
                        )
                safety_result = self.clinical_safety.evaluate(
                    query=query,
                    memory_context=memory_str,
                    dialogue_messages=dialogue_messages,
                    health_profile=health_profile,
                )
                if safety_result.findings:
                    self.logger.bind(tag=TAG).warning(
                        "临床安全规则命中: "
                        f"context={safety_result.extracted_context}, "
                        f"rules={[item.rule_id for item in safety_result.findings]}"
                    )

                if safety_result.should_block:
                    if tool_call_reminder:
                        self.dialogue.dialogue = [
                            msg for msg in self.dialogue.dialogue
                            if not getattr(msg, "is_temporary", False)
                        ]
                    self._reply_clinical_safety_interception(
                        safety_result.response_text,
                        current_sentence_id,
                    )
                    return True

                if safety_result.prompt_context:
                    memory_str = (
                        f"{memory_str}\n\n{safety_result.prompt_context}"
                        if memory_str
                        else safety_result.prompt_context
                    )

            if self.intent_type == "function_call" and functions is not None:
                # 使用支持functions的streaming接口
                llm_responses = self.llm.response_with_functions(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {})
                    ),
                    functions=functions,
                )
            else:
                llm_responses = self.llm.response(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {})
                    ),
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM 处理出错 {query}: {e}")
            return None

        # 处理流式响应
        tool_call_flag = False
        # 支持多个并行工具调用 - 使用列表存储
        tool_calls_list = []  # 格式: [{"id": "", "name": "", "arguments": ""}]
        content_arguments = ""
        emotion_flag = True
        try:
            for response in llm_responses:
                if self.client_abort:
                    break
                if self.intent_type == "function_call" and functions is not None:
                    content, tools_call = response
                    if "content" in response:
                        content = response["content"]
                        tools_call = None
                    if content is not None and len(content) > 0:
                        content_arguments += content

                    if not tool_call_flag and content_arguments.startswith("<tool_call>"):
                        # print("content_arguments", content_arguments)
                        tool_call_flag = True

                    if tools_call is not None and len(tools_call) > 0:
                        tool_call_flag = True
                        self._merge_tool_calls(tool_calls_list, tools_call)
                else:
                    content = response

                # 在llm回复中获取情绪表情，一轮对话只在开头获取一次
                if emotion_flag and content is not None and content.strip():
                    asyncio.run_coroutine_threadsafe(
                        textUtils.get_emotion(self, content),
                        self.loop,
                    )
                    emotion_flag = False

                if content is not None and len(content) > 0:
                    if not tool_call_flag:
                        response_message.append(content)
                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=current_sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.TEXT,
                                content_detail=content,
                            )
                        )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM stream processing error: {e}")
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail=get_system_error_response(self.config),
                )
            )
            if depth == 0:
                self.tts.tts_text_queue.put(
                    TTSMessageDTO(
                        sentence_id=current_sentence_id,
                        sentence_type=SentenceType.LAST,
                        content_type=ContentType.ACTION,
                    )
                )
            return
        # 处理function call
        if tool_call_flag:
            bHasError = False
            # 处理基于文本的工具调用格式
            if len(tool_calls_list) == 0 and content_arguments:
                a = extract_json_from_string(content_arguments)
                if a is not None:
                    try:
                        content_arguments_json = json.loads(a)
                        tool_calls_list.append(
                            {
                                "id": str(uuid.uuid4().hex),
                                "name": content_arguments_json["name"],
                                "arguments": json.dumps(
                                    content_arguments_json["arguments"],
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    except Exception as e:
                        bHasError = True
                        response_message.append(a)
                else:
                    bHasError = True
                    response_message.append(content_arguments)
                if bHasError:
                    self.logger.bind(tag=TAG).error(
                        f"function call error: {content_arguments}"
                    )

            if not bHasError and len(tool_calls_list) > 0:
                self.logger.bind(tag=TAG).debug(
                    f"检测到 {len(tool_calls_list)} 个工具调用"
                )

                # 更新工具调用统计
                if depth == 0:
                    current_turn = len(self.dialogue.dialogue) // 2
                    self.tool_call_stats['last_call_turn'] = current_turn
                    self.tool_call_stats['consecutive_no_call'] = 0
                    self.logger.bind(tag=TAG).debug(
                        f"工具调用统计更新: 当前轮次={current_turn}"
                    )

                # LLM 流式阶段已播报过的文本
                streamed_text = ""
                if len(response_message) > 0:
                    streamed_text = "".join(response_message)
                    self.tts.store_tts_text(current_sentence_id, streamed_text)
                    self.dialogue.put(Message(role="assistant", content=streamed_text))
                response_message.clear()

                # 收集所有工具调用的 Future
                futures_with_data = []
                for tool_call_data in tool_calls_list:
                    self.logger.bind(tag=TAG).debug(
                        f"function_name={tool_call_data['name']}, function_id={tool_call_data['id']}, function_arguments={tool_call_data['arguments']}"
                    )

                    # 使用公共方法上报工具调用
                    tool_input = json.loads(tool_call_data.get("arguments") or "{}")
                    enqueue_tool_report(self, tool_call_data['name'], tool_input)

                    future = asyncio.run_coroutine_threadsafe(
                        self.func_handler.handle_llm_function_call(
                            self, tool_call_data
                        ),
                        self.loop,
                    )
                    futures_with_data.append((future, tool_call_data, tool_input))

                # 工具调用超时时间，可配置，默认30秒
                tool_call_timeout = int(self.config.get("tool_call_timeout", 30))
                # 等待协程结束（实际等待时长为最慢的那个）
                tool_results = []

                for future, tool_call_data, tool_input in futures_with_data:
                    try:
                        result = future.result(timeout=tool_call_timeout)
                        tool_results.append((result, tool_call_data))
                        # 使用公共方法上报工具调用结果
                        enqueue_tool_report(self, tool_call_data['name'], tool_input, str(result.result) if result.result else None, report_tool_call=False)

                    except Exception as e:
                        self.logger.bind(tag=TAG).error(
                            f"工具调用超时或异常: {tool_call_data['name']}, 错误: {e}"
                        )
                        # 超时时返回错误响应，避免整个流程卡死
                        tool_results.append((
                            ActionResponse(action=Action.ERROR, result="哎呀，网络遇到点问题，请稍后再试下！"),
                            tool_call_data
                        ))
                        # 上报工具调用错误
                        enqueue_tool_report(self, tool_call_data['name'], tool_input, str(e), report_tool_call=False)

                # 统一处理工具调用结果
                if tool_results:
                    self._handle_function_result(tool_results, depth=depth, streamed_text=streamed_text)

        # 存储对话内容
        if len(response_message) > 0:
            text_buff = "".join(response_message)
            self.tts.store_tts_text(current_sentence_id, text_buff)
            self.dialogue.put(Message(role="assistant", content=text_buff))

            # 更新工具调用统计：如果没有调用工具，增加计数
            if depth == 0 and not tool_call_flag:
                self.tool_call_stats['consecutive_no_call'] += 1

        if depth == 0:
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
            # 使用lambda延迟计算，只有在DEBUG级别时才执行get_llm_dialogue()
            self.logger.bind(tag=TAG).debug(
                lambda: json.dumps(
                    self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False
                )
            )

            # 清理临时插入的工具调用提醒消息（使用标记清理）
            if tool_call_reminder and len(self.dialogue.dialogue) > 0:
                original_length = len(self.dialogue.dialogue)
                self.dialogue.dialogue = [
                    msg for msg in self.dialogue.dialogue
                    if not getattr(msg, 'is_temporary', False)
                ]
                if len(self.dialogue.dialogue) < original_length:
                    self.logger.bind(tag=TAG).debug("已清理临时的工具调用提醒消息")

        return True

    def _maybe_compact_short_term_memory(self):
        if self.memory is None or not hasattr(self.memory, "should_compact_dialogue"):
            return
        if not getattr(self, "loop", None):
            return

        dialogue_messages = [
            msg for msg in self.dialogue.dialogue
            if getattr(msg, "role", None) != "system"
            and not getattr(msg, "is_temporary", False)
        ]
        if not self.memory.should_compact_dialogue(dialogue_messages):
            return

        keep_count = self.memory.get_short_term_recent_message_count()
        keep_messages = self._select_short_term_tail_messages(
            dialogue_messages,
            keep_count,
        )
        summarize_end = len(dialogue_messages) - len(keep_messages)
        messages_to_summarize = dialogue_messages[:summarize_end]
        if len(messages_to_summarize) < 2:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.memory.update_short_term_summary_from_messages(
                    messages_to_summarize,
                    session_id=self.session_id,
                    reason="active_compaction",
                ),
                self.loop,
            )
            result = future.result(
                timeout=int(self.config.get("short_term_compaction_timeout", 25))
            )
            if not result:
                return

            keep_ids = {
                getattr(msg, "uniq_id", None)
                for msg in keep_messages
            }
            original_length = len(self.dialogue.dialogue)
            self.dialogue.dialogue = [
                msg for msg in self.dialogue.dialogue
                if getattr(msg, "role", None) == "system"
                or getattr(msg, "is_temporary", False)
                or getattr(msg, "uniq_id", None) in keep_ids
            ]
            removed = original_length - len(self.dialogue.dialogue)
            if removed > 0:
                self.logger.bind(tag=TAG).info(
                    f"短期记忆已压缩: removed_messages={removed}, keep_messages={keep_count}"
                )
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(f"短期记忆压缩失败，保留原始上下文: {exc}")

    def _maybe_refresh_health_profile(self):
        if self.memory is None or not hasattr(self.memory, "update_health_profile_from_messages"):
            return
        if not getattr(self, "loop", None):
            return

        dialogue_messages = [
            msg for msg in self.dialogue.dialogue
            if getattr(msg, "role", None) != "system"
            and not getattr(msg, "is_temporary", False)
        ]
        if not dialogue_messages:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.memory.update_health_profile_from_messages(
                    dialogue_messages,
                    reason="live_dialogue",
                ),
                self.loop,
            )
            return future.result(
                timeout=float(self.config.get("health_profile_live_update_timeout", 3))
            )
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(f"实时健康档案更新失败: {exc}")
            return None

    def _maybe_resolve_health_profile_voice_review(self, query, sentence_id):
        decision = self._detect_health_profile_review_decision(query)
        if decision is None:
            return False
        pending_reviews = self._get_pending_health_profile_reviews(limit=2)
        if not pending_reviews:
            return False

        review = pending_reviews[0]
        store = self._get_health_profile_store()
        if store is None or not getattr(self, "loop", None):
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                store.resolve_review_item(
                    review.get("review_id"),
                    decision,
                    resolved_by="voice_user",
                ),
                self.loop,
            )
            future.result(
                timeout=float(self.config.get("health_profile_live_update_timeout", 3))
            )
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(f"语音处理健康档案确认失败: {exc}")
            self._reply_health_profile_confirmation(
                "这条档案确认我暂时没有处理成功，请稍后在控制台里确认一下。",
                sentence_id,
            )
            return True

        if decision == "accept":
            text = f"好的，已确认更新档案：{self._format_health_profile_review_applied(review)}。"
        else:
            text = f"好的，这条档案更新我先忽略，{self._format_health_profile_review_current_kept(review)}。"

        remaining_reviews = self._get_pending_health_profile_reviews(limit=1)
        if remaining_reviews:
            text += " 另外还有一条需要你确认：" + self._format_health_profile_review_question(
                remaining_reviews[0],
                remaining_count=0,
            )
        self._reply_health_profile_confirmation(text, sentence_id)
        return True

    def _maybe_reply_health_profile_voice_review(self, stats, sentence_id):
        if not stats or int(stats.get("review_count") or 0) <= 0:
            return False
        pending_reviews = self._get_pending_health_profile_reviews(limit=2)
        if not pending_reviews:
            return False
        text = self._format_health_profile_review_question(
            pending_reviews[0],
            remaining_count=max(0, len(pending_reviews) - 1),
        )
        self._reply_health_profile_confirmation(text, sentence_id)
        return True

    def _get_health_profile_store(self):
        if self.memory is None:
            return None
        return getattr(self.memory, "health_profile_store", None)

    def _get_health_profile_user_id(self):
        if self.memory is not None and getattr(self.memory, "role_id", None):
            return self.memory.role_id
        return self.user_id or normalize_device_user_id(self.device_id)

    def _get_pending_health_profile_reviews(self, limit=1):
        store = self._get_health_profile_store()
        user_id = self._get_health_profile_user_id()
        if store is None or not user_id or not getattr(self, "loop", None):
            return []

        try:
            future = asyncio.run_coroutine_threadsafe(
                store.list_review_items(user_id, status="pending"),
                self.loop,
            )
            reviews = future.result(
                timeout=float(self.config.get("health_profile_live_update_timeout", 3))
            )
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(f"读取待确认健康档案失败: {exc}")
            return []
        return list(reviews or [])[: max(1, int(limit or 1))]

    @staticmethod
    def _detect_health_profile_review_decision(query):
        normalized = re.sub(r"[\s，。！？,.!?:：；;、]+", "", str(query or ""))
        if not normalized or len(normalized) > 16:
            return None
        reject_markers = (
            "忽略",
            "不用",
            "不更新",
            "不要更新",
            "别更新",
            "别改",
            "不是",
            "不对",
            "取消",
            "先别",
            "保持原来",
            "维持原来",
        )
        if any(marker in normalized for marker in reject_markers):
            return "reject"
        accept_markers = (
            "确认更新",
            "确认",
            "更新",
            "改成这个",
            "就按这个",
            "是的",
            "对的",
            "没错",
            "可以",
            "是",
            "对",
        )
        if any(marker == normalized or marker in normalized for marker in accept_markers):
            return "accept"
        return None

    def _format_health_profile_review_question(self, review, remaining_count=0):
        if (review or {}).get("field_type") == "scalar":
            label = self._health_profile_field_label(review.get("name"))
            current_value = self._format_health_profile_value(
                review.get("name"),
                (review.get("current_value") or {}).get("value"),
            )
            proposed_value = self._format_health_profile_value(
                review.get("name"),
                (review.get("proposed_value") or {}).get("value"),
            )
            text = (
                f"我发现你刚说的{label}和档案里已有记录不一致。"
                f"档案里是{current_value}，你刚说的是{proposed_value}。"
                f"要把档案更新为{proposed_value}吗？你可以说“确认更新”或“忽略”。"
            )
        else:
            label = self._health_profile_category_label(review.get("category"))
            proposed = review.get("proposed_value") or {}
            current = review.get("current_value") or {}
            if proposed.get("status") == "negate_category":
                current_names = [
                    str(item.get("name"))
                    for item in current.get("items", [])
                    if item.get("name")
                ]
                current_text = "、".join(current_names[:3]) if current_names else f"已有{label}"
                text = (
                    f"我发现你刚说没有{label}，但档案里记录了{current_text}。"
                    "要把这些记录停用吗？你可以说“确认更新”或“忽略”。"
                )
            else:
                name = proposed.get("name") or review.get("name") or "这条信息"
                text = (
                    f"我发现新的{label}信息“{name}”和档案里的记录有冲突。"
                    "要按这次说法更新档案吗？你可以说“确认更新”或“忽略”。"
                )
        if remaining_count > 0:
            text += f" 还有{remaining_count}条待确认信息，我会一条一条问你。"
        return text

    def _format_health_profile_review_applied(self, review):
        if (review or {}).get("field_type") == "scalar":
            label = self._health_profile_field_label(review.get("name"))
            proposed_value = self._format_health_profile_value(
                review.get("name"),
                (review.get("proposed_value") or {}).get("value"),
            )
            return f"{label}改为{proposed_value}"
        label = self._health_profile_category_label((review or {}).get("category"))
        proposed = (review or {}).get("proposed_value") or {}
        if proposed.get("status") == "negate_category":
            return f"{label}记录已按你确认停用"
        name = proposed.get("name") or (review or {}).get("name") or "这条信息"
        return f"{label}更新为{name}"

    def _format_health_profile_review_current_kept(self, review):
        if (review or {}).get("field_type") == "scalar":
            label = self._health_profile_field_label(review.get("name"))
            current_value = self._format_health_profile_value(
                review.get("name"),
                (review.get("current_value") or {}).get("value"),
            )
            return f"{label}仍保持{current_value}"
        label = self._health_profile_category_label((review or {}).get("category"))
        return f"{label}档案保持原样"

    @staticmethod
    def _health_profile_field_label(field_name):
        labels = {
            "age_years": "年龄",
            "sex": "性别",
            "height_cm": "身高",
            "weight_kg": "体重",
            "bmi": "BMI",
            "activity_level": "活动水平",
            "nutrition_goal": "营养目标",
            "target_energy_kcal": "每日热量目标",
            "target_carbohydrate_g_per_meal": "每餐碳水目标",
            "target_protein_g_per_day": "每日蛋白质目标",
            "target_fat_g_per_day": "每日脂肪目标",
        }
        return labels.get(str(field_name or ""), str(field_name or "档案字段"))

    @staticmethod
    def _health_profile_category_label(category):
        labels = {
            "disease": "疾病",
            "medication": "用药",
            "allergy": "过敏",
            "goal": "健康目标",
            "renal_function": "肾功能",
            "glucose_metric": "血糖指标",
            "exercise": "运动习惯",
            "dietary_restriction": "饮食限制",
        }
        return labels.get(str(category or ""), str(category or "健康档案"))

    @staticmethod
    def _format_health_profile_value(field_name, value):
        if value in (None, ""):
            return "未知"
        field_name = str(field_name or "")
        if field_name == "sex":
            sex_map = {"male": "男", "female": "女", "m": "男", "f": "女"}
            return sex_map.get(str(value).lower(), str(value))
        try:
            numeric = float(value)
            if numeric.is_integer():
                numeric_text = str(int(numeric))
            else:
                numeric_text = f"{numeric:.1f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            numeric_text = str(value)

        units = {
            "age_years": "岁",
            "height_cm": "厘米",
            "weight_kg": "公斤",
            "target_energy_kcal": "千卡",
            "target_carbohydrate_g_per_meal": "克",
            "target_protein_g_per_day": "克",
            "target_fat_g_per_day": "克",
        }
        return f"{numeric_text}{units.get(field_name, '')}"

    def _reply_health_profile_confirmation(self, text, sentence_id):
        if not text:
            text = "我发现档案信息需要确认。你可以说“确认更新”或“忽略”。"
        self.logger.bind(tag=TAG).info(f"健康档案语音确认: {text}")
        self.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=sentence_id,
                sentence_type=SentenceType.MIDDLE,
                content_type=ContentType.TEXT,
                content_detail=text,
            )
        )
        self.tts.store_tts_text(sentence_id, text)
        self.dialogue.put(Message(role="assistant", content=text))
        self.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=sentence_id,
                sentence_type=SentenceType.LAST,
                content_type=ContentType.ACTION,
            )
        )

    @staticmethod
    def _select_short_term_tail_messages(dialogue_messages, keep_count: int):
        keep_count = max(1, int(keep_count or 1))
        start = max(0, len(dialogue_messages) - keep_count)

        # Keep tool result messages attached to their preceding assistant
        # tool_call message after active dialogue compaction.
        while start > 0 and getattr(dialogue_messages[start], "role", None) == "tool":
            start -= 1

        return dialogue_messages[start:]

    def _build_tool_call_reminder(self, tool_summary: str, force_reminder: bool = False) -> str:
        reminder = (
            "<tool_calling>\n"
            "当前会话是连续对话。你必须结合最近几轮上下文理解当前问题，"
            "不能把每一句都当成全新话题。\n"
            "如果用户是在追问、复述、补问、纠正、代词指代或省略主语，"
            "优先延续上一轮语义再决定是否需要调用工具。\n"
            "工具是否要重调可以独立判断，但对话理解不能重置。\n"
            f"当前可用工具: {tool_summary}。\n"
        )
        if force_reminder:
            reminder += (
                "最近多轮没有使用工具。请重新检查这次请求是否需要最新数据、"
                "现场感知或执行操作；如果需要，再调用工具。\n"
            )
        else:
            reminder += (
                "仅当用户请求涉及实时信息查询、现场感知或执行操作时调用工具，"
                "日常连续对话无需为了形式重新开题。\n"
            )
        reminder += "</tool_calling>"
        return reminder

    def _should_auto_trigger_camera(self, query):
        if not query or not getattr(self, "func_handler", None):
            return False
        if not self.func_handler.has_tool("self_camera_take_photo"):
            return False

        normalized = re.sub(r"\s+", "", str(query or ""))
        if not normalized:
            return False

        explicit_visual_markers = (
            "拍照",
            "拍一下",
            "拍一张",
            "帮我看看",
            "给我看看",
            "看一下这个",
            "看下这个",
            "识别一下",
            "扫一扫",
        )
        if any(marker in normalized for marker in explicit_visual_markers):
            return True

        referential_markers = (
            "这个东西",
            "这个",
            "我面前",
            "面前",
            "眼前",
            "手上这个",
            "这瓶",
            "这包",
            "这杯",
            "这盒",
        )
        judgement_markers = (
            "可以喝吗",
            "能喝吗",
            "能不能喝",
            "可以吃吗",
            "能吃吗",
            "能不能吃",
            "这是什么",
            "是什么",
            "是啥",
            "能用吗",
            "能不能用",
        )
        return (
            any(marker in normalized for marker in referential_markers)
            and any(marker in normalized for marker in judgement_markers)
        )

    @staticmethod
    def _build_camera_question(query):
        text = str(query or "").strip()
        if not text:
            return "描述一下看到的物品"
        return f"请结合照片回答用户问题：{text}"

    def _invoke_direct_tool_call(self, function_name, arguments, depth=0):
        if not getattr(self, "func_handler", None):
            return False

        tool_call_data = {
            "id": str(uuid.uuid4().hex),
            "name": function_name,
            "arguments": json.dumps(arguments or {}, ensure_ascii=False),
        }
        tool_input = arguments or {}
        try:
            enqueue_tool_report(self, function_name, tool_input)
            future = asyncio.run_coroutine_threadsafe(
                self.func_handler.handle_llm_function_call(self, tool_call_data),
                self.loop,
            )
            result = future.result(timeout=int(self.config.get("tool_call_timeout", 30)))
            enqueue_tool_report(
                self,
                function_name,
                tool_input,
                str(result.result) if getattr(result, "result", None) else None,
                report_tool_call=False,
            )
            self._handle_function_result([(result, tool_call_data)], depth=depth)
            return True
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(
                f"自动触发工具失败，回退到常规对话: {function_name}, error={exc}"
            )
            return False

    def _reply_clinical_safety_interception(self, text, sentence_id):
        if not text:
            text = "我先帮你拦一下，这里可能存在临床安全风险，建议先咨询医生或药师后再决定。"
        self.logger.bind(tag=TAG).warning(f"临床安全规则已拦截本轮回答: {text}")
        self.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=sentence_id,
                sentence_type=SentenceType.MIDDLE,
                content_type=ContentType.TEXT,
                content_detail=text,
            )
        )
        self.tts.store_tts_text(sentence_id, text)
        self.dialogue.put(Message(role="assistant", content=text))
        self.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=sentence_id,
                sentence_type=SentenceType.LAST,
                content_type=ContentType.ACTION,
            )
        )

    def _get_tool_summary(self, functions: list) -> str:
        """
        从工具定义中提取摘要，用于规则强化注入

        Args:
            functions: 工具列表

        Returns:
            str: 工具名称字符串
        """
        if not functions:
            return ""

        datas = []
        for func in functions:
            func_info = func.get("function", {})
            name = func_info.get("name", "")
            datas.append(name)
        result = "、".join(datas)
        return result

    def _handle_function_result(self, tool_results, depth, streamed_text=""):
        need_llm_tools = []

        for result, tool_call_data in tool_results:
            if result.action in [
                Action.RESPONSE,
                Action.NOTFOUND,
                Action.ERROR,
            ]:
                text = result.response if result.response else result.result
                if streamed_text and text in streamed_text:
                    self.logger.bind(tag=TAG).debug(
                        f"Skipping duplicate TTS for tool {tool_call_data['name']}, already streamed"
                    )
                else:
                    self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=text)
                    self.tts.store_tts_text(self.sentence_id, text)
                self.dialogue.put(Message(role="assistant", content=text))
            elif result.action == Action.REQLLM:
                # 收集需要 LLM 处理的工具
                need_llm_tools.append((result, tool_call_data))
            else:
                pass

        if need_llm_tools:
            all_tool_calls = [
                {
                    "id": tool_call_data["id"],
                    "function": {
                        "arguments": (
                            "{}"
                            if tool_call_data["arguments"] == ""
                            else tool_call_data["arguments"]
                        ),
                        "name": tool_call_data["name"],
                    },
                    "type": "function",
                    "index": idx,
                }
                for idx, (_, tool_call_data) in enumerate(need_llm_tools)
            ]
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tool_call_data in need_llm_tools:
                text = result.result
                if text is not None and len(text) > 0:
                    self.dialogue.put(
                        Message(
                            role="tool",
                            tool_call_id=(
                                str(uuid.uuid4())
                                if tool_call_data["id"] is None
                                else tool_call_data["id"]
                            ),
                            content=text,
                        )
                    )

            self.chat(None, depth=depth + 1)

    def _report_worker(self):
        """聊天记录上报工作线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取数据，设置超时以便定期检查停止事件
                item = self.report_queue.get(timeout=1)
                if item is None:  # 检测毒丸对象
                    break
                try:
                    # 检查线程池状态
                    if self.executor is None:
                        continue
                    # 提交任务到线程池
                    self.executor.submit(self._process_report, *item)
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"聊天记录上报线程异常: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"聊天记录上报工作线程异常: {e}")

        self.logger.bind(tag=TAG).info("聊天记录上报线程已退出")

    def _process_report(self, type, text, audio_data, report_time):
        """处理上报任务"""
        try:
            # 执行异步上报（在事件循环中运行）
            asyncio.run(report(self, type, text, audio_data, report_time))
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"上报处理异常: {e}")
        finally:
            # 标记任务完成
            self.report_queue.task_done()

    def clearSpeakStatus(self):
        self.client_is_speaking = False
        self.logger.bind(tag=TAG).debug(f"清除服务端讲话状态")

    async def close(self, ws=None):
        """资源清理方法"""
        try:
            # 清理 VAD 连接资源
            if (
                    hasattr(self, "vad")
                    and self.vad
                    and hasattr(self.vad, "release_conn_resources")
            ):
                self.vad.release_conn_resources(self)

            # 清理音频缓冲区
            if hasattr(self, "audio_buffer"):
                self.audio_buffer.clear()

            # 取消超时任务
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                try:
                    await self.timeout_task
                except asyncio.CancelledError:
                    pass
                self.timeout_task = None

            # 清理工具处理器资源
            if hasattr(self, "func_handler") and self.func_handler:
                try:
                    await self.func_handler.cleanup()
                except Exception as cleanup_error:
                    self.logger.bind(tag=TAG).error(
                        f"清理工具处理器时出错: {cleanup_error}"
                    )

            # 触发停止事件
            if self.stop_event:
                self.stop_event.set()

            # 清空任务队列
            self.clear_queues()

            # 关闭WebSocket连接
            try:
                if ws:
                    # 安全地检查WebSocket状态并关闭
                    try:
                        if hasattr(ws, "closed") and not ws.closed:
                            await ws.close()
                        elif hasattr(ws, "state") and ws.state.name != "CLOSED":
                            await ws.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await ws.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
                elif self.websocket:
                    try:
                        if (
                                hasattr(self.websocket, "closed")
                                and not self.websocket.closed
                        ):
                            await self.websocket.close()
                        elif (
                                hasattr(self.websocket, "state")
                                and self.websocket.state.name != "CLOSED"
                        ):
                            await self.websocket.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await self.websocket.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
            except Exception as ws_error:
                self.logger.bind(tag=TAG).error(f"关闭WebSocket连接时出错: {ws_error}")

            if self.tts:
                await self.tts.close()
            if self.asr:
                await self.asr.close()

            # 最后关闭线程池（避免阻塞）
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(
                        f"关闭线程池时出错: {executor_error}"
                    )
                self.executor = None
            self.logger.bind(tag=TAG).info("连接资源已释放")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"关闭连接时出错: {e}")
        finally:
            # 确保停止事件被设置
            if self.stop_event:
                self.stop_event.set()

    def clear_queues(self):
        """清空所有任务队列"""
        if self.tts:
            self.logger.bind(tag=TAG).debug(
                f"开始清理: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

            # 使用非阻塞方式清空队列
            for q in [
                self.tts.tts_text_queue,
                self.tts.tts_audio_queue,
                self.report_queue,
            ]:
                if not q:
                    continue
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # 重置音频流控器（取消后台任务并清空队列）
            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                self.audio_rate_controller.reset()
                self.logger.bind(tag=TAG).debug("已重置音频流控器")

            self.logger.bind(tag=TAG).debug(
                f"清理结束: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

    def reset_audio_states(self):
        """
        重置所有音频相关状态(VAD + ASR)
        """
        # Reset VAD states
        self.client_audio_buffer.clear()
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_voice_window.clear()
        self.last_is_voice = False
        self.vad_last_voice_time = 0.0

        # Clear ASR buffers
        self.asr_audio.clear()
        while True:
            try:
                self.asr_audio_queue.get_nowait()
            except queue.Empty:
                break
        if hasattr(self, "audio_timestamp_buffer"):
            self.audio_timestamp_buffer.clear()
        self.last_processed_timestamp = 0

        self.logger.bind(tag=TAG).debug("All audio states reset.")

    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")

    async def _check_timeout(self):
        """检查连接超时"""
        try:
            while not self.stop_event.is_set():
                last_activity_time = self.last_activity_time
                if self.need_bind:
                    last_activity_time = self.first_activity_time

                # 检查是否超时（只有在时间戳已初始化的情况下）
                if last_activity_time > 0.0:
                    current_time = time.time() * 1000
                    if current_time - last_activity_time > self.timeout_seconds * 1000:
                        if not self.stop_event.is_set():
                            self.logger.bind(tag=TAG).info("连接超时，准备关闭")
                            # 设置停止事件，防止重复处理
                            self.stop_event.set()
                            # 使用 try-except 包装关闭操作，确保不会因为异常而阻塞
                            try:
                                await self.close(self.websocket)
                            except Exception as close_error:
                                self.logger.bind(tag=TAG).error(
                                    f"超时关闭连接时出错: {close_error}"
                                )
                        break
                # 每10秒检查一次，避免过于频繁
                await asyncio.sleep(10)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"超时检查任务出错: {e}")
        finally:
            self.logger.bind(tag=TAG).info("超时检查任务已退出")

    def _merge_tool_calls(self, tool_calls_list, tools_call):
        """合并工具调用列表

        Args:
            tool_calls_list: 已收集的工具调用列表
            tools_call: 新的工具调用
        """
        for tool_call in tools_call:
            tool_index = getattr(tool_call, "index", None)
            if tool_index is None:
                if tool_call.function.name:
                    # 有 function_name，说明是新的工具调用
                    tool_index = len(tool_calls_list)
                else:
                    tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0

            # 确保列表有足够的位置
            if tool_index >= len(tool_calls_list):
                tool_calls_list.append({"id": "", "name": "", "arguments": ""})

            # 更新工具调用信息
            if tool_call.id:
                tool_calls_list[tool_index]["id"] = tool_call.id
            if tool_call.function.name:
                tool_calls_list[tool_index]["name"] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls_list[tool_index]["arguments"] += tool_call.function.arguments
