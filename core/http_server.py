import asyncio
from aiohttp import web
from config.logger import setup_logging
from core.api.clinical_console_handler import ClinicalConsoleHandler
from core.api.ota_handler import OTAHandler
from core.api.vision_handler import VisionHandler

TAG = __name__


class SimpleHttpServer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.clinical_console_handler = ClinicalConsoleHandler(config)
        self.ota_handler = OTAHandler(config)
        self.vision_handler = VisionHandler(config)

    def _get_websocket_url(self, local_ip: str, port: int) -> str:
        """获取websocket地址

        Args:
            local_ip: 本地IP地址
            port: 端口号

        Returns:
            str: websocket地址
        """
        server_config = self.config["server"]
        websocket_config = server_config.get("websocket")

        if websocket_config and "你" not in websocket_config:
            return websocket_config
        else:
            return f"ws://{local_ip}:{port}/xiaozhi/v1/"

    async def start(self):
        try:
            server_config = self.config["server"]
            read_config_from_api = self.config.get("read_config_from_api", False)
            host = server_config.get("ip", "0.0.0.0")
            port = int(server_config.get("http_port", 8003))

            if port:
                app = web.Application()

                if not read_config_from_api:
                    # 如果没有开启智控台，只是单模块运行，就需要再添加简单OTA接口，用于下发websocket接口
                    app.add_routes(
                        [
                            web.get("/xiaozhi/ota/", self.ota_handler.handle_get),
                            web.post("/xiaozhi/ota/", self.ota_handler.handle_post),
                            web.options(
                                "/xiaozhi/ota/", self.ota_handler.handle_options
                            ),
                            # 下载接口，仅提供 data/bin/*.bin 下载
                            web.get(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_download,
                            ),
                            web.options(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_options,
                            ),
                        ]
                    )
                # 添加路由
                app.add_routes(
                    [
                        web.get("/console", self.clinical_console_handler.handle_console),
                        web.get("/console/", self.clinical_console_handler.handle_console),
                        web.get(
                            "/console/api/summary",
                            self.clinical_console_handler.handle_summary,
                        ),
                        web.get(
                            "/console/api/users",
                            self.clinical_console_handler.handle_users,
                        ),
                        web.get(
                            "/console/api/profile",
                            self.clinical_console_handler.handle_profile,
                        ),
                        web.get(
                            "/console/api/profile/{user_id}",
                            self.clinical_console_handler.handle_profile,
                        ),
                        web.get(
                            "/console/api/profile-review",
                            self.clinical_console_handler.handle_profile_review,
                        ),
                        web.get(
                            "/console/api/profile-review/{user_id}",
                            self.clinical_console_handler.handle_profile_review,
                        ),
                        web.post(
                            "/console/api/profile-review/{review_id}/resolve",
                            self.clinical_console_handler.handle_profile_review_resolve,
                        ),
                        web.get(
                            "/console/api/memory",
                            self.clinical_console_handler.handle_memory,
                        ),
                        web.get(
                            "/console/api/memory/{user_id}",
                            self.clinical_console_handler.handle_memory,
                        ),
                        web.get(
                            "/console/api/knowledge/files",
                            self.clinical_console_handler.handle_knowledge_files,
                        ),
                        web.post(
                            "/console/api/knowledge/upload",
                            self.clinical_console_handler.handle_knowledge_upload,
                        ),
                        web.get(
                            "/console/api/rag/documents",
                            self.clinical_console_handler.handle_rag_documents,
                        ),
                        web.post(
                            "/console/api/rag/upload",
                            self.clinical_console_handler.handle_rag_upload,
                        ),
                        web.post(
                            "/console/api/rag/documents/{document_id}/index",
                            self.clinical_console_handler.handle_rag_index,
                        ),
                        web.get(
                            "/console/api/rag/jobs/{job_id}",
                            self.clinical_console_handler.handle_rag_job,
                        ),
                        web.get(
                            "/console/api/rag/documents/{document_id}/chunks",
                            self.clinical_console_handler.handle_rag_chunks,
                        ),
                        web.post(
                            "/console/api/rag/search",
                            self.clinical_console_handler.handle_rag_search,
                        ),
                        web.delete(
                            "/console/api/rag/documents/{document_id}",
                            self.clinical_console_handler.handle_rag_delete,
                        ),
                        web.post(
                            "/console/api/clinical-knowledge/documents/{document_id}/llm-review",
                            self.clinical_console_handler.handle_structured_knowledge_review,
                        ),
                        web.post(
                            "/console/api/clinical-knowledge/documents/{document_id}/approve",
                            self.clinical_console_handler.handle_structured_knowledge_approve,
                        ),
                        web.post(
                            "/console/api/clinical-knowledge/needs-review/{review_id}/resolve",
                            self.clinical_console_handler.handle_structured_needs_review_resolve,
                        ),
                        web.post(
                            "/console/api/knowledge/ingest",
                            self.clinical_console_handler.handle_knowledge_ingest,
                        ),
                        web.get(
                            "/console/api/knowledge/ingestion/drafts",
                            self.clinical_console_handler.handle_knowledge_ingestion_drafts,
                        ),
                        web.get(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}",
                            self.clinical_console_handler.handle_knowledge_ingestion_detail,
                        ),
                        web.post(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}/approve",
                            self.clinical_console_handler.handle_knowledge_ingestion_approve,
                        ),
                        web.post(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}/review",
                            self.clinical_console_handler.handle_knowledge_ingestion_review,
                        ),
                        web.post(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}/regenerate-plan",
                            self.clinical_console_handler.handle_knowledge_ingestion_regenerate_plan,
                        ),
                        web.post(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}/extract-structured",
                            self.clinical_console_handler.handle_knowledge_ingestion_extract_structured,
                        ),
                        web.get(
                            "/console/api/knowledge/ingestion/drafts/{draft_id}/needs-review",
                            self.clinical_console_handler.handle_knowledge_ingestion_needs_review,
                        ),
                        web.get(
                            "/console/api/rules",
                            self.clinical_console_handler.handle_rules,
                        ),
                        web.get(
                            "/console/api/food",
                            self.clinical_console_handler.handle_food_search,
                        ),
                        web.post(
                            "/console/api/meal/analyze",
                            self.clinical_console_handler.handle_meal_analyze,
                        ),
                        web.get(
                            "/console/api/model-config",
                            self.clinical_console_handler.handle_model_config_get,
                        ),
                        web.post(
                            "/console/api/model-config",
                            self.clinical_console_handler.handle_model_config_save,
                        ),
                        web.get(
                            "/console/api/agent-settings",
                            self.clinical_console_handler.handle_agent_settings_get,
                        ),
                        web.post(
                            "/console/api/agent-settings",
                            self.clinical_console_handler.handle_agent_settings_save,
                        ),
                        web.get(
                            "/console/api/history",
                            self.clinical_console_handler.handle_history_list,
                        ),
                        web.get(
                            "/console/api/history/{session_id}",
                            self.clinical_console_handler.handle_history_detail,
                        ),
                        web.options(
                            "/console/api/{tail:.*}",
                            self.clinical_console_handler.handle_options,
                        ),
                        web.get(
                            "/console/{path:.*}",
                            self.clinical_console_handler.handle_console_asset,
                        ),
                        web.get("/mcp/vision/explain", self.vision_handler.handle_get),
                        web.post(
                            "/mcp/vision/explain", self.vision_handler.handle_post
                        ),
                        web.options(
                            "/mcp/vision/explain", self.vision_handler.handle_options
                        ),
                    ]
                )

                # 运行服务
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, host, port)
                await site.start()

                # 保持服务运行
                while True:
                    await asyncio.sleep(3600)  # 每隔 1 小时检查一次
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"HTTP服务器启动失败: {e}")
            import traceback

            self.logger.bind(tag=TAG).error(f"错误堆栈: {traceback.format_exc()}")
            raise
