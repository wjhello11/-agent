import json
import copy
from aiohttp import web
from config.logger import setup_logging
from core.api.base_handler import BaseHandler
from core.utils.util import get_vision_url, is_valid_image_file
from core.utils.vllm import create_instance
from config.config_loader import get_private_config_from_api
from core.clinical_nutrition.vision_nutrition import VisionNutritionAnalyzer
from core.providers.memory.clinical_ltm.health_profile import HealthProfileStore
from core.utils.device_identity import normalize_device_user_id
from core.utils.auth import AuthToken
import base64
from pathlib import Path
from typing import Tuple, Optional
from plugins_func.register import Action

TAG = __name__

# 设置最大文件大小为5MB
MAX_FILE_SIZE = 5 * 1024 * 1024


class VisionHandler(BaseHandler):
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化认证工具
        self.auth = AuthToken(config.get("server", {}).get("auth_key", ""))

    def _create_error_response(self, message: str) -> dict:
        """创建统一的错误响应格式"""
        return {"success": False, "message": message}

    def _verify_auth_token(self, request) -> Tuple[bool, Optional[str]]:
        """验证认证token"""
        # 测试模式：允许特定测试令牌或跳过验证
        auth_header = request.headers.get("Authorization", "")
        client_id = request.headers.get("Client-Id", "")

        # 允许测试客户端跳过认证
        if client_id == "web_test_client":
            device_id = request.headers.get("Device-Id", "test_device")
            return True, device_id

        if not auth_header.startswith("Bearer "):
            return False, None

        token = auth_header[7:]  # 移除"Bearer "前缀
        return self.auth.verify_token(token)

    async def handle_post(self, request):
        """处理 MCP Vision POST 请求"""
        response = None  # 初始化response变量
        try:
            # 验证token
            is_valid, token_device_id = self._verify_auth_token(request)
            if not is_valid:
                response = web.Response(
                    text=json.dumps(
                        self._create_error_response("无效的认证token或token已过期")
                    ),
                    content_type="application/json",
                    status=401,
                )
                return response

            # 获取请求头信息
            device_id = request.headers.get("Device-Id", "")
            client_id = request.headers.get("Client-Id", "")
            if device_id != token_device_id:
                raise ValueError("设备ID与token不匹配")
            # 解析multipart/form-data请求
            reader = await request.multipart()

            # 读取question字段
            question_field = await reader.next()
            if question_field is None:
                raise ValueError("缺少问题字段")
            question = await question_field.text()
            self.logger.bind(tag=TAG).debug(f"Question: {question}")

            # 读取图片文件
            image_field = await reader.next()
            if image_field is None:
                raise ValueError("缺少图片文件")

            # 读取图片数据
            image_data = await image_field.read()
            if not image_data:
                raise ValueError("图片数据为空")

            # 检查文件大小
            if len(image_data) > MAX_FILE_SIZE:
                raise ValueError(
                    f"图片大小超过限制，最大允许{MAX_FILE_SIZE/1024/1024}MB"
                )

            # 检查文件格式
            if not is_valid_image_file(image_data):
                raise ValueError(
                    "不支持的文件格式，请上传有效的图片文件（支持JPEG、PNG、GIF、BMP、TIFF、WEBP格式）"
                )

            # 将图片转换为base64编码
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 如果开启了智控台，则从智控台获取模型配置
            current_config = copy.deepcopy(self.config)
            read_config_from_api = current_config.get("read_config_from_api", False)
            if read_config_from_api:
                current_config = await get_private_config_from_api(
                    current_config,
                    device_id,
                    client_id,
                )

            select_vllm_module = current_config["selected_module"].get("VLLM")
            if not select_vllm_module:
                raise ValueError("您还未设置默认的视觉分析模块")

            vllm_type = (
                select_vllm_module
                if "type" not in current_config["VLLM"][select_vllm_module]
                else current_config["VLLM"][select_vllm_module]["type"]
            )

            if not vllm_type:
                raise ValueError(f"无法找到VLLM模块对应的供应器{vllm_type}")

            vllm = create_instance(
                vllm_type, current_config["VLLM"][select_vllm_module]
            )

            health_profile_context, health_profile = await self._load_health_profile_bundle(
                current_config,
                normalize_device_user_id(device_id),
            )
            analyzer = VisionNutritionAnalyzer(
                project_root=Path(__file__).resolve().parents[2],
                config=current_config,
                logger=self.logger,
            )
            structured_question = analyzer.build_structured_prompt(
                user_question=question,
                health_profile_context=health_profile_context,
            )
            raw_result = vllm.response(structured_question, image_base64)
            analysis = analyzer.build_response(
                vlm_raw_text=raw_result,
                user_question=question,
                health_profile=health_profile,
                food_db_path=self._food_db_path(current_config),
                rules_path=self._rules_path(current_config),
            )

            return_json = {
                "success": True,
                "action": Action.RESPONSE.name,
                "response": analysis.response_text,
                "structured_vision": analysis.structured,
                "nutrition": analysis.nutrition,
                "safety_findings": analysis.safety_findings,
            }

            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except ValueError as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision POST请求异常: {e}")
            return_json = self._create_error_response(str(e))
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision POST请求异常: {e}")
            return_json = self._create_error_response("处理请求时发生错误")
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_get(self, request):
        """处理 MCP Vision GET 请求"""
        try:
            vision_explain = get_vision_url(self.config)
            if vision_explain and len(vision_explain) > 0 and "null" != vision_explain:
                message = (
                    f"MCP Vision 接口运行正常，视觉解释接口地址是：{vision_explain}"
                )
            else:
                message = "MCP Vision 接口运行不正常，请打开data目录下的.config.yaml文件，找到【server.vision_explain】，设置好地址"

            response = web.Response(text=message, content_type="text/plain")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision GET请求异常: {e}")
            return_json = self._create_error_response("服务器内部错误")
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        finally:
            self._add_cors_headers(response)
            return response

    async def _build_clinical_vision_question(
        self,
        current_config: dict,
        device_id: str,
        question: str,
    ) -> str:
        user_id = normalize_device_user_id(device_id)
        health_profile_context = await self._load_health_profile_context(
            current_config,
            user_id,
        )

        parts = [
            "你是个性化临床营养师 AI Agent 的视觉分析模块。",
            "请先识别图片中的食物或饮品，再结合用户问题判断是否适合当前用户。",
            "如果用户问“能不能喝/吃、午餐能不能喝/吃、适不适合我”，必须优先结合健康档案中的疾病、血糖、用药、过敏、肾功能和营养目标。",
            "对于奶茶、含糖饮料、甜品、果汁等高糖饮品，如健康档案提示糖尿病或血糖异常，应优先说明血糖风险、建议不喝或只在医生/营养师允许且明确控量时少量饮用；不要把回答重点放在是否冰、新鲜、是不是别人的，除非用户明确问食品卫生。",
            "回答要简短、中文口语化，先给结论，再给理由和可执行替代建议。",
        ]
        if health_profile_context:
            parts.append(health_profile_context)
        parts.append(f"用户问题：{question}")
        return "\n".join(parts)

    def _food_db_path(self, current_config: dict) -> Path:
        plugin_config = current_config.get("plugins", {}).get("search_food_nutrition", {})
        configured = plugin_config.get("db_path") or "data/clinical_foods.db"
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path

    def _rules_path(self, current_config: dict) -> Path:
        configured = current_config.get("clinical_safety_rules_path")
        if not configured:
            configured = "knowledge_base/rules/clinical_safety_rules.json"
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path

    async def _load_health_profile_bundle(
        self,
        current_config: dict,
        user_id: str,
    ) -> tuple[str, dict]:
        if not user_id:
            return "", {}

        memory_config = current_config.get("Memory", {})
        selected_memory = current_config.get("selected_module", {}).get("Memory", "")
        selected_memory_config = memory_config.get(selected_memory, {})
        if not selected_memory_config.get("health_profile_enabled", True):
            return "", {}

        db_path = selected_memory_config.get("health_profile_sqlite_path")
        if not db_path:
            sqlite_path = selected_memory_config.get("sqlite_path", "data/clinical_ltm.db")
            db_path = str(Path(sqlite_path).with_name("clinical_health_profile.db"))

        resolved_path = Path(db_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).resolve().parents[2] / resolved_path

        try:
            store = HealthProfileStore(resolved_path)
            profile = await store.get_profile(user_id)
            context = await store.build_prompt_context(user_id)
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(
                f"视觉分析加载健康档案失败: user_id={user_id}, error={exc}"
            )
            return "", {}

        if not context or "暂无结构化健康档案" in context:
            context = ""
        return context, profile

    async def _load_health_profile_context(
        self,
        current_config: dict,
        user_id: str,
    ) -> str:
        context, _ = await self._load_health_profile_bundle(current_config, user_id)
        return context
