const NAV_ITEMS = [
  { key: "settings", label: "设置", icon: "⌘" },
  { key: "llm-settings", label: "大模型设置", icon: "◎" },
  { key: "profile", label: "健康档案", icon: "◫" },
  { key: "knowledge", label: "知识库", icon: "▤" },
  { key: "memory", label: "记忆库", icon: "◌" },
  { key: "chat", label: "历史对话", icon: "☰" },
];

const PROFILE_CATEGORY_LABELS = {
  allergy: "过敏",
  disease: "疾病",
  medication: "用药",
  medicine: "用药",
  diet_restriction: "饮食限制",
  dietary_restriction: "饮食限制",
  goal: "目标",
  renal_function: "肾功能",
  glycemic_metric: "血糖指标",
  blood_pressure: "血压",
  lipid: "血脂",
  exercise: "运动",
  activity: "运动量",
  habit: "生活习惯",
  preference: "饮食偏好",
  note: "备注",
  other: "其他",
};

const PROFILE_SOURCE_LABELS = {
  dialogue_user: "对话抽取",
  dialogue_assistant: "对话整理",
  manual: "手动录入",
  console_admin: "控制台录入",
  system: "系统写入",
};

const state = {
  page: "settings",
  users: [],
  currentUserId: "",
  summary: null,
  agentSettings: null,
  modelConfig: null,
  profile: null,
  memory: null,
  knowledgeFiles: [],
  wikiDrafts: [],
  selectedWikiDraftId: "",
  selectedWikiDraft: null,
  selectedWikiNeedsReview: null,
  ragDocuments: [],
  selectedRagDocumentId: "",
  selectedRagChunks: [],
  ragSearchQuery: "",
  ragSearchResults: [],
  structuredReviewResult: null,
  rules: [],
  historyRecords: [],
  historySessions: [],
  selectedSessionId: "",
  selectedSession: null,
  historyView: "chat",
  foodQuery: "",
  foodResults: [],
  mealText: "两个鸡蛋、一杯牛奶、两片白面包",
  mealRecord: true,
  mealAnalysis: null,
};

const pageTitleEl = document.getElementById("page-title");
const navEl = document.getElementById("nav");
const userSelectEl = document.getElementById("user-select");
const summaryStripEl = document.getElementById("summary-strip");
const pageContentEl = document.getElementById("page-content");
const refreshButtonEl = document.getElementById("refresh-button");
const toastRootEl = document.getElementById("toast-root");

bootstrap();

async function bootstrap() {
  renderNav();
  bindShellEvents();
  await refreshAll();
}

function bindShellEvents() {
  refreshButtonEl.addEventListener("click", async () => {
    await refreshAll();
  });

  userSelectEl.addEventListener("change", async (event) => {
    state.currentUserId = event.target.value;
    await loadUserScopedData();
    renderAll();
  });
}

async function refreshAll() {
  try {
    renderLoading("正在加载控制台数据...");
    await Promise.all([
      loadSummary(),
      loadUsers(),
      loadAgentSettings(),
      loadModelConfig(),
      loadKnowledgeData(),
    ]);
    await loadUserScopedData();
    renderAll();
  } catch (error) {
    renderError(error);
    showToast(error.message || "加载失败", "error");
  }
}

async function loadSummary() {
  state.summary = await apiGet("/console/api/summary");
}

async function loadUsers() {
  const payload = await apiGet("/console/api/users?limit=80");
  state.users = payload.users || [];
  if (!state.currentUserId && state.users.length) {
    state.currentUserId = state.users[0].user_id;
  }
  renderUserSelect();
}

async function loadAgentSettings() {
  const payload = await apiGet("/console/api/agent-settings");
  state.agentSettings = payload.agent_settings || null;
}

async function loadModelConfig() {
  const payload = await apiGet("/console/api/model-config");
  state.modelConfig = payload.model_config || null;
}

async function loadKnowledgeData() {
  const [filesPayload, rulesPayload, ragPayload, draftsPayload] = await Promise.all([
    apiGet("/console/api/knowledge/files"),
    apiGet("/console/api/rules"),
    apiGet("/console/api/rag/documents"),
    apiGet("/console/api/knowledge/ingestion/drafts"),
  ]);
  state.knowledgeFiles = filesPayload.files || [];
  state.rules = rulesPayload.rules || [];
  state.ragDocuments = ragPayload.documents || [];
  state.wikiDrafts = draftsPayload.drafts || [];
  if (
    state.selectedRagDocumentId &&
    !state.ragDocuments.some((document) => document.document_id === state.selectedRagDocumentId)
  ) {
    state.selectedRagDocumentId = "";
    state.selectedRagChunks = [];
  }
  if (
    state.selectedWikiDraftId &&
    !state.wikiDrafts.some((draft) => draft.draft_id === state.selectedWikiDraftId)
  ) {
    state.selectedWikiDraftId = "";
    state.selectedWikiDraft = null;
  }
}

async function loadRagChunks(documentId) {
  if (!documentId) {
    state.selectedRagDocumentId = "";
    state.selectedRagChunks = [];
    return;
  }
  const payload = await apiGet(`/console/api/rag/documents/${encodeURIComponent(documentId)}/chunks?limit=200`);
  state.selectedRagDocumentId = documentId;
  state.selectedRagChunks = payload.chunks || [];
  if (state.structuredReviewResult?.document?.document_id !== documentId) {
    state.structuredReviewResult = null;
  }
}

async function loadWikiDraft(draftId) {
  if (!draftId) {
    state.selectedWikiDraftId = "";
    state.selectedWikiDraft = null;
    state.selectedWikiNeedsReview = null;
    return;
  }
  const payload = await apiGet(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}`);
  state.selectedWikiDraftId = draftId;
  state.selectedWikiDraft = payload.draft || null;
  state.selectedWikiNeedsReview = null;
}

async function loadWikiDraftNeedsReview(draftId) {
  if (!draftId) {
    state.selectedWikiNeedsReview = null;
    return;
  }
  const payload = await apiGet(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}/needs-review`);
  state.selectedWikiNeedsReview = payload || null;
}

async function loadUserScopedData() {
  if (!state.currentUserId) {
    state.profile = null;
    state.memory = null;
    state.historyRecords = [];
    state.historySessions = [];
    state.selectedSessionId = "";
    state.selectedSession = null;
    return;
  }

  const [profilePayload, memoryPayload, historyPayload] = await Promise.all([
    apiGet(`/console/api/profile/${encodeURIComponent(state.currentUserId)}`),
    apiGet(`/console/api/memory/${encodeURIComponent(state.currentUserId)}?limit=80`),
    apiGet(`/console/api/history?user_id=${encodeURIComponent(state.currentUserId)}&limit=80`),
  ]);

  state.profile = profilePayload.profile || null;
  state.memory = memoryPayload.memory || null;
  state.historyRecords = historyPayload.sessions || [];
  state.historySessions = buildHistoryBuckets(state.historyRecords);

  const stillExists = state.historySessions.some(
    (item) => item.session_id === state.selectedSessionId,
  );
  if (!stillExists) {
    state.selectedSessionId = state.historySessions[0]?.session_id || "";
  }
  if (state.selectedSessionId) {
    await loadSelectedSession(state.selectedSessionId);
  } else {
    state.selectedSession = null;
  }
}

async function loadSelectedSession(sessionId) {
  state.selectedSessionId = sessionId;
  if (!sessionId) {
    state.selectedSession = null;
    return;
  }
  const bucket = state.historySessions.find((item) => item.session_id === sessionId);
  if (!bucket) {
    state.selectedSession = null;
    return;
  }

  const detailResults = await Promise.allSettled(
    (bucket.session_ids || []).map((sourceSessionId) =>
      apiGet(`/console/api/history/${encodeURIComponent(sourceSessionId)}`),
    ),
  );

  const detailSessions = detailResults
    .filter((result) => result.status === "fulfilled" && result.value?.session)
    .map((result) => result.value.session);

  if (!detailSessions.length) {
    throw new Error("当天对话读取失败");
  }

  const mergedMessages = detailSessions
    .flatMap((detail) =>
      (Array.isArray(detail.messages) ? detail.messages : []).map((message) => ({
        ...message,
        source_session_id: detail.session_id,
      })),
    )
    .sort((left, right) => compareDateValue(left.created_at, right.created_at));

  state.selectedSession = {
    session_id: bucket.session_id,
    title: bucket.title,
    preview: bucket.preview,
    user_id: bucket.user_id,
    device_id: bucket.device_id || "",
    created_at: bucket.created_at,
    updated_at: bucket.updated_at,
    message_count: mergedMessages.length,
    has_tool_calls: bucket.has_tool_calls,
    has_vision: bucket.has_vision,
    session_count: bucket.session_ids.length,
    source_session_ids: bucket.session_ids,
    messages: mergedMessages,
  };
}

function renderAll() {
  renderNav();
  renderUserSelect();
  renderSummaryStrip();
  renderCurrentPage();
}

function renderNav() {
  navEl.innerHTML = NAV_ITEMS.map(
    (item) => `
      <button class="nav-button ${state.page === item.key ? "active" : ""}" data-page="${item.key}" type="button">
        <span class="nav-icon">${item.icon}</span>
        <span>${item.label}</span>
      </button>
    `,
  ).join("");
  navEl.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => {
      state.page = button.dataset.page;
      renderAll();
    });
  });
}

function renderUserSelect() {
  if (!state.users.length) {
    userSelectEl.innerHTML = `<option value="">暂无用户</option>`;
    userSelectEl.disabled = true;
    return;
  }
  userSelectEl.disabled = false;
  userSelectEl.innerHTML = state.users
    .map(
      (user) => `
        <option value="${escapeAttr(user.user_id)}" ${user.user_id === state.currentUserId ? "selected" : ""}>
          ${escapeHtml(user.user_id)}
        </option>
      `,
    )
    .join("");
}

function renderSummaryStrip() {
  const summary = state.summary;
  if (!summary || state.page !== "settings") {
    summaryStripEl.innerHTML = "";
    return;
  }
  const usersCount = Array.isArray(summary.users) ? summary.users.length : 0;
  const ragChunks = summary.knowledge?.rag_chunks || 0;
  const rulesCount = summary.rules?.count || 0;
  const foodsCount =
    summary.databases?.food_nutrition?.tables?.food_items ?? 0;

  summaryStripEl.innerHTML = [
    metricCard("设备 / 用户", String(usersCount), "按设备 ID 绑定"),
    metricCard("RAG 片段", String(ragChunks), "指南证据索引"),
    metricCard("安全规则", String(rulesCount), "临床红线"),
    metricCard("食物条目", String(foodsCount), "结构化营养库"),
  ].join("");
}

function renderCurrentPage() {
  const navItem = NAV_ITEMS.find((item) => item.key === state.page);
  pageTitleEl.textContent = navItem?.label || "控制台";

  if (state.page === "settings") {
    pageContentEl.innerHTML = renderSettingsPage();
    bindSettingsEvents();
    return;
  }
  if (state.page === "llm-settings") {
    pageContentEl.innerHTML = renderModelSettingsPage();
    bindModelSettingsEvents();
    return;
  }
  if (state.page === "profile") {
    pageContentEl.innerHTML = renderProfilePage();
    bindProfileEvents();
    return;
  }
  if (state.page === "knowledge") {
    pageContentEl.innerHTML = renderKnowledgePage();
    bindKnowledgeEvents();
    return;
  }
  if (state.page === "memory") {
    pageContentEl.innerHTML = renderMemoryPage();
    return;
  }
  if (state.page === "chat") {
    pageContentEl.innerHTML = renderHistoryPage();
    bindHistoryEvents();
    return;
  }

  pageContentEl.innerHTML = `<div class="empty">未找到页面。</div>`;
}

function renderSettingsPage() {
  const settings = state.agentSettings;
  const summary = state.memory?.short_term_summary;
  if (!settings) {
    return `<div class="empty">Agent 设置尚未加载。</div>`;
  }
  const controls = settings.voice?.controls || {};
  const rateMeta = sliderMeta(controls.rate_field, "rate");
  const pitchMeta = sliderMeta(controls.pitch_field, "pitch");
  const volumeMeta = sliderMeta(controls.volume_field, "volume");
  const memoryPrompts = settings.memory_prompts || {};

  return `
    <form id="agent-settings-form" class="page-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">角色与 Prompt</div>
            <div class="section-help">沿用你的草图结构，把昵称、语言、角色介绍和运行 Prompt 放在同一页里。</div>
          </div>
          <span class="status-pill info">保存后需重启服务</span>
        </div>
        <div class="panel-body band">
          <div class="two-col">
            <label class="field">
              <span>助手昵称</span>
              <input name="assistant_name" value="${escapeAttr(settings.assistant_name || "")}" />
            </label>
            <label class="field">
              <span>对话语言</span>
              <select name="language">
                ${optionList([settings.language || "普通话", "普通话", "中文", "English"])}
              </select>
            </label>
          </div>
          <label class="field">
            <span>角色身份 / 主 Prompt</span>
            <textarea name="prompt">${escapeHtml(settings.prompt || "")}</textarea>
            <small>这里只写可编辑的身份和语气边界；具体任务、安全规则、工具调用和上下文占位符放在下面的运行基础模板里。</small>
          </label>
          <div class="notice">
            <div class="label">当前 Prompt 模板</div>
            <div class="mono">${escapeHtml(settings.prompt_template || "")}</div>
          </div>
          <label class="field prompt-field">
            <span>运行基础 Prompt 模板</span>
            <textarea name="prompt_template_content">${escapeHtml(settings.prompt_template_content || "")}</textarea>
            <small>这是最终 system prompt 的固定骨架，会把上面的主 Prompt 填入 {{base_prompt}}；请保留 {{base_prompt}}、{{current_time}}、{{today_date}}、{{local_address}}、{{weather_info}} 等占位符。</small>
          </label>
          <div class="two-col">
            <label class="field prompt-field">
              <span>长期记忆抽取 Prompt</span>
              <textarea name="memory_prompts.long_term_extraction_system_prompt">${escapeHtml(memoryPrompts.long_term_extraction_system_prompt || "")}</textarea>
              <small>用于判断哪些对话要写入长期记忆，并抽取事实、事件和长期规律。</small>
            </label>
            <label class="field prompt-field">
              <span>短期记忆压缩 Prompt</span>
              <textarea name="memory_prompts.short_term_summary_system_prompt">${escapeHtml(memoryPrompts.short_term_summary_system_prompt || "")}</textarea>
              <small>用于把多轮对话压缩成“当前记忆”，请保留 {max_chars} 占位符。</small>
            </label>
          </div>
          <div class="notice">
            <div class="label">Prompt 结构</div>
            <div class="tag-row" style="margin-top:10px;">
              <span class="tag">主 Prompt：回答用户</span>
              <span class="tag">长期记忆抽取：写入 LTM</span>
              <span class="tag">短期记忆压缩：连续对话上下文</span>
            </div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">语音与高级设置</div>
            <div class="section-help">把音色、语速、音调、音量，以及 ASR 响应速度都放进这个“调音台”。</div>
          </div>
        </div>
        <div class="panel-body band">
          <div class="two-col">
            <label class="field">
              <span>角色音色模块</span>
              <select name="voice.module">${optionList(settings.voice?.available_modules || [], settings.voice?.module)}</select>
            </label>
            <label class="field">
              <span>语音识别模块</span>
              <select name="asr.module">${optionList(settings.asr?.available_modules || [], settings.asr?.module)}</select>
            </label>
            <label class="field">
              <span>voice</span>
              <input name="voice.voice" value="${escapeAttr(settings.voice?.voice || "")}" />
            </label>
            <label class="field">
              <span>speaker</span>
              <input name="voice.speaker" value="${escapeAttr(settings.voice?.speaker || "")}" />
            </label>
            <label class="field">
              <span>cluster</span>
              <input name="voice.cluster" value="${escapeAttr(settings.voice?.cluster || "")}" />
            </label>
            <label class="field">
              <span>resource_id</span>
              <input name="voice.resource_id" value="${escapeAttr(settings.voice?.resource_id || "")}" />
            </label>
          </div>
          <div class="three-col">
            ${sliderBox("角色语速", "voice.controls.rate", controls.rate ?? 1, rateMeta)}
            ${sliderBox("角色音调", "voice.controls.pitch", controls.pitch ?? 1, pitchMeta)}
            ${sliderBox("角色音量", "voice.controls.volume", controls.volume ?? 1, volumeMeta)}
          </div>
          <div class="two-col">
            <label class="field">
              <span>语音识别速度</span>
              <select name="asr.recognition_speed_preset">
                ${optionList(["fast", "normal", "stable"], settings.asr?.recognition_speed_preset || "normal", {
                  fast: "快速",
                  normal: "正常",
                  stable: "稳定",
                })}
              </select>
            </label>
            <label class="field">
              <span>MCP 接入点</span>
              <input name="mcp.endpoint" value="${escapeAttr(settings.mcp?.endpoint || "")}" />
            </label>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">短期记忆</div>
            <div class="section-help">这里是你要求的“压缩后短期记忆”界面，和官方控制台类似，但直接读取你现在的临床记忆引擎。</div>
          </div>
          <span class="status-pill ${summary ? "ok" : "warn"}">${summary ? "已生成摘要" : "暂无摘要"}</span>
        </div>
        <div class="panel-body band">
          <div class="three-col">
            <label class="field field-check">
              <input type="checkbox" name="short_term_memory.enabled" ${settings.short_term_memory?.enabled ? "checked" : ""} />
              <span>启用短期记忆压缩</span>
            </label>
            <label class="field">
              <span>字数上限</span>
              <input type="number" name="short_term_memory.max_chars" value="${escapeAttr(settings.short_term_memory?.max_chars ?? 2000)}" />
            </label>
            <label class="field">
              <span>压缩 token 上限</span>
              <input type="number" name="short_term_memory.max_tokens" value="${escapeAttr(settings.short_term_memory?.max_tokens ?? 1200)}" />
            </label>
            <label class="field">
              <span>压缩温度</span>
              <input type="number" step="0.1" name="short_term_memory.temperature" value="${escapeAttr(settings.short_term_memory?.temperature ?? 0.2)}" />
            </label>
            <label class="field">
              <span>保留最近消息数</span>
              <input type="number" name="short_term_memory.recent_messages" value="${escapeAttr(settings.short_term_memory?.recent_messages ?? 8)}" />
            </label>
            <label class="field">
              <span>触发压缩阈值</span>
              <input type="number" name="short_term_memory.compact_trigger_messages" value="${escapeAttr(settings.short_term_memory?.compact_trigger_messages ?? 18)}" />
            </label>
          </div>
          <div class="notice">
            <div class="label">当前短期记忆摘要</div>
            ${summary ? `
              <div class="tag-row" style="margin: 10px 0 12px;">
                <span class="tag">会话 ${escapeHtml(summary.source_session_id || "-")}</span>
                <span class="tag">轮数 ${escapeHtml(summary.source_turn_count ?? "-")}</span>
                <span class="tag">更新 ${escapeHtml(formatDate(summary.updated_at))}</span>
              </div>
              <div>${escapeHtml(summary.summary || "")}</div>
            ` : `<div class="muted">当前用户还没有可展示的短期记忆摘要。</div>`}
          </div>
        </div>
      </section>

      <div class="action-row">
        <button class="primary-button" type="submit">保存设置</button>
      </div>
    </form>
  `;
}

function renderModelSettingsPage() {
  const config = state.modelConfig;
  if (!config) {
    return `<div class="empty">模型配置尚未加载。</div>`;
  }

  return `
    <form id="model-config-form" class="page-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">模型编排面板</div>
            <div class="section-help">这里不是泛泛的 API 区，而是按“每个模型的作用”分组：主对话、意图、记忆抽取、Embedding、Vision、文档入库。</div>
          </div>
        </div>
        <div class="panel-body band">
          ${renderModelEditorSection("主对话 LLM", "负责最终回答用户", "main_llm", config.main_llm)}
          ${renderIntentSection(config.intent)}
          ${renderSimpleModelSection("PowerMem LLM", "PowerMem 文本理解与记忆整理", "powermem_llm", config.powermem_llm)}
          ${renderEmbeddingSection(config.embedding)}
          ${renderSimpleMem0Section(config.mem0)}
          ${renderVectorStoreSection(config.vector_store)}
          ${renderRagSettingsSection(config.clinical_rag)}
          ${renderRuntimeModelSection("Vision", "负责图片理解与视觉判断", "vision", config.vision)}
          ${renderRuntimeModelSection("ASR", "负责把语音转成文本", "asr", config.asr)}
          ${renderRuntimeModelSection("TTS", "负责把回答转成语音", "tts", config.tts)}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">模型使用地图</div>
            <div class="section-help">直接对应你这个 Agent 每一段链路到底在用哪个模型。</div>
          </div>
        </div>
        <div class="panel-body scroll-panel">
          <table>
            <thead>
              <tr>
                <th>模块</th>
                <th>状态</th>
                <th>路径</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              ${(config.usage_inventory || []).map((item) => `
                <tr>
                  <td><strong>${escapeHtml(item.name || item.key || "-")}</strong><div class="muted">${escapeHtml(item.model || "")}</div></td>
                  <td>${statusBadgeFromText(item.status || "")}</td>
                  <td><span class="mono">${escapeHtml(item.config_path || "")}</span></td>
                  <td>${escapeHtml(item.note || "")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </section>

      <div class="action-row">
        <button class="primary-button" type="submit">保存模型配置</button>
      </div>
    </form>
  `;
}

function renderProfilePage() {
  const profile = state.profile;
  if (!state.currentUserId) {
    return `<div class="empty">还没有可选用户。先让设备连上后端并说几句话，我们就能在这里看档案。</div>`;
  }
  if (!profile || !profile.exists) {
    return `
      <div class="page-stack">
        <section class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">用户画像与健康档案</div>
              <div class="section-help">当前设备 ${escapeHtml(state.currentUserId)} 还没有结构化档案。</div>
            </div>
          </div>
          <div class="panel-body">
            <div class="empty">暂无档案数据。先通过对话录入年龄、身高体重、疾病、用药、过敏等信息。</div>
          </div>
        </section>
      </div>
    `;
  }

  const scalarLabels = {
    age_years: "年龄",
    sex: "性别",
    height_cm: "身高",
    weight_kg: "体重",
    bmi: "BMI",
    activity_level: "运动量",
    nutrition_goal: "营养目标",
    target_energy_kcal: "目标热量",
    target_carbohydrate_g_per_meal: "每餐碳水目标",
    target_protein_g_per_day: "每日蛋白目标",
    target_fat_g_per_day: "每日脂肪目标",
    notes: "备注",
  };

  const grouped = groupBy(profile.items || [], (item) => item.category || "其他");
  const glucoseReadings = profile.glucose_readings || [];
  const glucoseAnalysis = profile.glucose_analysis || {};
  const glucoseAlerts = glucoseAnalysis.alerts || [];
  const reviewItems = profile.review_items || [];
  const nutritionTargets = profile.nutrition_targets || {};
  const nutritionSeries = profile.nutrition_intake_series || [];

  return `
    <div class="page-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">用户画像与健康档案</div>
            <div class="section-help">当前按设备 ID 绑定用户。这里读取的是你已经落地到结构化健康档案库里的数据。</div>
          </div>
          <span class="status-pill ok">${escapeHtml(state.currentUserId)}</span>
        </div>
        <div class="panel-body band">
          <div class="three-col">
            ${Object.entries(scalarLabels).map(([key, label]) => `
              <div class="notice">
                <div class="label">${escapeHtml(label)}</div>
                <div style="margin-top:8px;font-weight:700;">${escapeHtml(formatProfileScalarCardValue(key, profile, nutritionTargets))}</div>
                ${renderProfileScalarCardHint(key, profile, nutritionTargets)}
              </div>
            `).join("")}
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">每日营养目标估算</div>
            <div class="section-help">用户明确说过的目标优先；没有明确目标时，按档案里的性别、体重、年龄、身高和活动量做粗略估算。</div>
          </div>
          <span class="status-pill ${nutritionTargets.available ? "info" : "warn"}">${nutritionTargets.available ? "估算可用" : "缺少体重"}</span>
        </div>
        <div class="panel-body band">
          ${renderNutritionTargets(nutritionTargets)}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">每日摄入曲线</div>
            <div class="section-help">曲线来自整餐营养分析工具的结构化记录；每次分析一餐后，会自动汇总到当天。</div>
          </div>
          <span class="status-pill ${nutritionSeries.some((item) => item.intake_count) ? "ok" : "warn"}">${nutritionSeries.reduce((sum, item) => sum + Number(item.intake_count || 0), 0)} 餐</span>
        </div>
        <div class="panel-body band">
          ${renderNutritionIntakeChart(nutritionSeries, nutritionTargets)}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">档案待确认</div>
            <div class="section-help">与现有健康档案冲突的信息会先进入这里，确认后才会覆盖关键字段。</div>
          </div>
          <span class="status-pill ${reviewItems.length ? "warn" : "ok"}">${reviewItems.length} 项</span>
        </div>
        <div class="panel-body band">
          ${reviewItems.length ? `
            <table>
              <thead>
                <tr>
                  <th>字段</th>
                  <th>当前值</th>
                  <th>新上报</th>
                  <th>原因</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                ${reviewItems.map((item) => `
                  <tr>
                    <td><strong>${escapeHtml(formatReviewName(item))}</strong><div class="muted">${escapeHtml(formatProfileSource(item.source))}</div></td>
                    <td>${escapeHtml(formatValue(item.current_value))}</td>
                    <td>${escapeHtml(formatValue(item.proposed_value))}<div class="muted">${escapeHtml(item.evidence || "")}</div></td>
                    <td>${escapeHtml(item.reason || "-")}</td>
                    <td>
                      <button class="secondary-button compact" type="button" data-resolve-review="${escapeAttr(item.review_id)}" data-decision="accept">确认</button>
                      <button class="danger-button compact" type="button" data-resolve-review="${escapeAttr(item.review_id)}" data-decision="reject">忽略</button>
                    </td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          ` : `<div class="empty">暂无需要确认的档案冲突。</div>`}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">血糖时间序列</div>
            <div class="section-help">用户通过语音上报的血糖会按测量时间保存；这里展示趋势摘要和需要及时关注的提醒。</div>
          </div>
          <span class="status-pill ${glucoseAlerts.length ? "warn" : "ok"}">${glucoseReadings.length} 条</span>
        </div>
        <div class="panel-body band">
          ${glucoseAlerts.length ? `
            <div class="notice danger">
              ${glucoseAlerts.map((alert) => `
                <div style="margin-bottom:8px;">
                  <strong>${escapeHtml(alert.message || "-")}</strong>
                  <div class="muted">${escapeHtml(alert.recommendation || "")}</div>
                </div>
              `).join("")}
            </div>
          ` : `<div class="notice"><strong>暂无血糖告警</strong><div class="muted">${escapeHtml(glucoseAnalysis.summary || "还没有足够的血糖记录形成趋势。")}</div></div>`}
          ${glucoseReadings.length ? `
            <table>
              <thead>
                <tr>
                  <th>测量时间</th>
                  <th>类型</th>
                  <th>血糖</th>
                  <th>来源</th>
                  <th>原话</th>
                </tr>
              </thead>
              <tbody>
                ${glucoseReadings.slice(0, 20).map((item) => `
                  <tr>
                    <td>${escapeHtml(formatDate(item.measured_at))}</td>
                    <td>${escapeHtml(formatGlucoseType(item.measurement_type))}</td>
                    <td><strong>${escapeHtml(formatGlucoseValue(item.value_mmol_l))}</strong></td>
                    <td>${escapeHtml(formatProfileSource(item.source))}</td>
                    <td>${escapeHtml(item.evidence || "-")}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          ` : `<div class="empty">还没有血糖记录。你可以直接对小智说“今天早上8点空腹血糖7.2”或“午餐后两小时血糖10.5”。</div>`}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">结构化条目</div>
            <div class="section-help">疾病、用药、过敏、目标、肾功能、血糖指标等都在这张结构化表里。</div>
          </div>
        </div>
        <div class="panel-body band">
          ${Object.keys(grouped).length ? Object.entries(grouped).map(([category, items]) => `
            <div class="notice">
              <div class="label">${escapeHtml(formatProfileCategory(category))}</div>
              <table>
                <thead>
                  <tr>
                    <th>名称</th>
                    <th>值</th>
                    <th>来源</th>
                    <th>更新时间</th>
                  </tr>
                </thead>
                <tbody>
                  ${items.map((item) => `
                    <tr>
                      <td>${escapeHtml(item.name || "-")}</td>
                      <td>${escapeHtml(formatProfileValue(item.value))}</td>
                      <td>${escapeHtml(formatProfileSource(item.source))}</td>
                      <td>${escapeHtml(formatDate(item.updated_at))}</td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          `).join("") : `<div class="empty">结构化条目为空。</div>`}
        </div>
      </section>
    </div>
  `;
}

function renderNutritionTargets(targets) {
  if (!targets || !targets.available) {
    return `
      <div class="empty">${escapeHtml(targets?.reason || "缺少体重，暂时无法估算每日营养目标。")}</div>
    `;
  }
  const effective = targets.effective || {};
  const cards = [
    ["energy_kcal", "每日热量", "energy"],
    ["carbohydrate_g_per_day", "每日碳水", "carb"],
    ["protein_g_per_day", "每日蛋白质", "protein"],
    ["fat_g_per_day", "每日脂肪", "fat"],
    ["carbohydrate_g_per_meal", "每餐碳水", "carb"],
  ];
  return `
    <div class="nutrition-target-grid">
      ${cards.map(([key, label, tone]) => renderNutritionTargetCard(label, effective[key], tone)).join("")}
    </div>
    <div class="notice">
      <div class="label">估算依据</div>
      <div class="tag-row" style="margin-top:10px;">
        <span class="tag">方法 ${escapeHtml(formatNutritionMethod(targets.method))}</span>
        <span class="tag">活动量 ${escapeHtml(formatActivityBucket(targets.activity))}</span>
        ${targets.flags?.diabetes_adjusted_carbohydrate_ratio ? `<span class="tag">糖尿病碳水保守估算</span>` : ""}
        ${targets.flags?.renal_protein_conservative ? `<span class="tag">肾功能风险蛋白保守估算</span>` : ""}
      </div>
      <div class="source-list">
        ${(targets.notes || []).map((note) => `<div>${escapeHtml(note)}</div>`).join("")}
        ${(targets.sources || []).map((source) => `<a href="${escapeAttr(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title)}</a>`).join("")}
      </div>
    </div>
  `;
}

function formatProfileScalarCardValue(key, profile, nutritionTargets) {
  const explicit = profile?.scalars?.[key];
  if (explicit !== undefined && explicit !== null && explicit !== "") {
    return formatValue(explicit);
  }
  const estimated = estimatedScalarFallback(key, nutritionTargets);
  if (estimated) {
    return formatNutritionTarget(estimated);
  }
  return "-";
}

function renderProfileScalarCardHint(key, profile, nutritionTargets) {
  const explicit = profile?.scalars?.[key];
  if (explicit !== undefined && explicit !== null && explicit !== "") {
    return "";
  }
  const estimated = estimatedScalarFallback(key, nutritionTargets);
  if (!estimated) {
    return "";
  }
  return `<div class="muted" style="margin-top:6px;">系统估算，未写入档案字段</div>`;
}

function estimatedScalarFallback(key, nutritionTargets) {
  if (!nutritionTargets?.available) return null;
  const effective = nutritionTargets.effective || {};
  const mapping = {
    target_energy_kcal: "energy_kcal",
    target_carbohydrate_g_per_meal: "carbohydrate_g_per_meal",
    target_protein_g_per_day: "protein_g_per_day",
    target_fat_g_per_day: "fat_g_per_day",
  };
  return effective[mapping[key]] || null;
}

function renderNutritionTargetCard(label, target, tone) {
  return `
    <div class="notice nutrition-target-card ${escapeAttr(tone || "")}">
      <div class="label">${escapeHtml(label)}</div>
      <div class="nutrition-target-value">${escapeHtml(formatNutritionTarget(target))}</div>
      <div class="muted">${target?.source === "profile" ? "用户/档案明确目标" : "系统估算目标"}</div>
    </div>
  `;
}

function renderNutritionIntakeChart(series, targets) {
  const rows = buildNutritionChartRows(series, targets);
  const hasRecords = rows.some((row) => row.intake_count > 0);
  if (!hasRecords) {
    return `<div class="empty">还没有每日摄入记录。对小智说“帮我分析这餐：两个鸡蛋、一杯牛奶、两片面包”，曲线就会开始累积。</div>`;
  }
  const chart = renderNutritionPercentSvg(rows);
  const recentRows = rows.slice(-14).reverse();
  return `
    <div class="nutrition-chart-wrap">
      ${chart}
      <div class="nutrition-legend">
        <span><i class="line-swatch energy"></i>热量</span>
        <span><i class="line-swatch carb"></i>碳水</span>
        <span><i class="line-swatch protein"></i>蛋白质</span>
        <span><i class="line-swatch fat"></i>脂肪</span>
      </div>
    </div>
    <div class="scroll-panel">
      <table>
        <thead>
          <tr>
            <th>日期</th>
            <th>记录餐次</th>
            <th>热量</th>
            <th>碳水</th>
            <th>蛋白质</th>
            <th>脂肪</th>
          </tr>
        </thead>
        <tbody>
          ${recentRows.map((row) => `
            <tr>
              <td>${escapeHtml(row.date)}</td>
              <td>${escapeHtml(row.intake_count)}</td>
              <td>${escapeHtml(formatNutrientWithPercent(row.energy_kcal, "kcal", row.energy_pct))}</td>
              <td>${escapeHtml(formatNutrientWithPercent(row.carbohydrate_g, "g", row.carbohydrate_pct))}</td>
              <td>${escapeHtml(formatNutrientWithPercent(row.protein_g, "g", row.protein_pct))}</td>
              <td>${escapeHtml(formatNutrientWithPercent(row.fat_g, "g", row.fat_pct))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function buildNutritionChartRows(series, targets) {
  const effective = targets?.effective || {};
  const targetEnergy = numberValue(effective.energy_kcal?.value);
  const targetCarb = numberValue(effective.carbohydrate_g_per_day?.value);
  const targetProtein = numberValue(effective.protein_g_per_day?.value);
  const targetFat = numberValue(effective.fat_g_per_day?.value);
  return (series || []).map((item) => {
    const energy = numberValue(item.energy_kcal);
    const carb = numberValue(item.carbohydrate_g);
    const protein = numberValue(item.protein_g);
    const fat = numberValue(item.fat_g);
    return {
      date: item.date || "",
      intake_count: Number(item.intake_count || 0),
      energy_kcal: energy,
      carbohydrate_g: carb,
      protein_g: protein,
      fat_g: fat,
      energy_pct: percentOf(energy, targetEnergy),
      carbohydrate_pct: percentOf(carb, targetCarb),
      protein_pct: percentOf(protein, targetProtein),
      fat_pct: percentOf(fat, targetFat),
    };
  });
}

function renderNutritionPercentSvg(rows) {
  const width = 760;
  const height = 260;
  const padding = { left: 48, right: 18, top: 20, bottom: 34 };
  const maxPct = Math.max(120, ...rows.flatMap((row) => [
    row.energy_pct,
    row.carbohydrate_pct,
    row.protein_pct,
    row.fat_pct,
  ]));
  const yMax = Math.min(200, Math.ceil(maxPct / 20) * 20);
  const series = [
    ["energy_pct", "energy"],
    ["carbohydrate_pct", "carb"],
    ["protein_pct", "protein"],
    ["fat_pct", "fat"],
  ];
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (index) => padding.left + (rows.length <= 1 ? 0 : (plotWidth * index) / (rows.length - 1));
  const yFor = (value) => padding.top + plotHeight - (Math.min(value, yMax) / yMax) * plotHeight;
  const gridValues = [0, 50, 100, yMax].filter((value, index, all) => all.indexOf(value) === index && value <= yMax);
  return `
    <svg class="nutrition-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="每日营养摄入占目标百分比曲线">
      ${gridValues.map((value) => `
        <line x1="${padding.left}" y1="${yFor(value)}" x2="${width - padding.right}" y2="${yFor(value)}" class="chart-grid" />
        <text x="8" y="${yFor(value) + 4}" class="chart-axis">${value}%</text>
      `).join("")}
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" class="chart-axis-line" />
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" class="chart-axis-line" />
      ${series.map(([key, tone]) => {
        const points = rows.map((row, index) => `${xFor(index).toFixed(1)},${yFor(row[key]).toFixed(1)}`).join(" ");
        return `<polyline class="chart-line ${tone}" points="${points}" />`;
      }).join("")}
      ${rows.map((row, index) => {
        if (index % Math.ceil(rows.length / 6) !== 0 && index !== rows.length - 1) return "";
        return `<text x="${xFor(index)}" y="${height - 10}" class="chart-axis date" text-anchor="middle">${escapeHtml(row.date.slice(5))}</text>`;
      }).join("")}
    </svg>
  `;
}

function renderKnowledgePage() {
  const summary = state.summary;
  const files = state.knowledgeFiles || [];
  const wikiDrafts = state.wikiDrafts || [];
  const selectedWikiDraft = state.selectedWikiDraft;
  const ragDocuments = state.ragDocuments || [];
  const selectedRagDocument = ragDocuments.find((item) => item.document_id === state.selectedRagDocumentId);
  const rules = state.rules || [];
  const foodResults = state.foodResults || [];
  const foodTableRows = foodResults.length
    ? foodResults.map((food) => `
        <tr>
          <td><strong>${escapeHtml(food.canonical_name || "-")}</strong><div class="muted">${escapeHtml(food.chinese_name || food.english_name || "")}</div></td>
          <td>${escapeHtml(food.food_category || "-")}</td>
          <td>${escapeHtml(food.nutrients_per_100g?.energy_kcal ?? "-")}</td>
          <td>${escapeHtml(food.nutrients_per_100g?.carbohydrate_g ?? "-")}</td>
          <td>${escapeHtml(food.nutrients_per_100g?.protein_g ?? "-")}</td>
          <td>${escapeHtml(food.nutrients_per_100g?.fat_g ?? "-")}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="6" class="muted">还没有搜索结果。</td></tr>`;

  return `
    <div class="page-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">知识库总览</div>
            <div class="section-help">结构化营养库、LLMWiki、本地 RAG、上传素材和临床红线都在这里汇总。</div>
          </div>
        </div>
        <div class="panel-body three-col">
          <div class="notice">
            <div class="label">LLMWiki 页面</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(summary?.knowledge?.wiki_pages ?? 0)}</div>
          </div>
          <div class="notice">
            <div class="label">Wiki 草案</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(summary?.knowledge?.wiki_drafts ?? wikiDrafts.length)}</div>
          </div>
          <div class="notice">
            <div class="label">RAG 片段</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(summary?.knowledge?.rag_chunks ?? 0)}</div>
          </div>
          <div class="notice">
            <div class="label">结构化表格</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(summary?.knowledge?.structured_tables ?? 0)}</div>
          </div>
          <div class="notice">
            <div class="label">结构化食谱</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(summary?.knowledge?.structured_recipe_plans ?? 0)}</div>
          </div>
          <div class="notice">
            <div class="label">食养方 / MET</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(`${summary?.knowledge?.structured_therapeutic_recipes ?? 0} / ${summary?.knowledge?.structured_activity_mets ?? 0}`)}</div>
          </div>
          <div class="notice">
            <div class="label">已上传原始文档</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(files.length)}</div>
          </div>
          <div class="notice">
            <div class="label">RAG 文档</div>
            <div style="margin-top:8px;font-size:22px;font-weight:700;">${escapeHtml(ragDocuments.length)}</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">上传知识文档</div>
            <div class="section-help">支持把 PDF、Word、Markdown、表格等资料先上传进素材层。</div>
          </div>
        </div>
        <div class="panel-body band">
          <form id="upload-form" class="toolbar upload-toolbar">
            <label class="upload-button file-picker-button" for="upload-input">选择文件</label>
            <input type="file" id="upload-input" class="file-input-hidden" multiple />
            <span id="upload-file-name" class="file-name">未选择文件</span>
            <button class="upload-button" type="submit">上传文件</button>
          </form>
          <div class="scroll-panel">
            <table>
              <thead>
                <tr>
                  <th>文件</th>
                  <th>大小</th>
                  <th>状态</th>
                  <th>更新时间</th>
                    <th>入库</th>
                </tr>
              </thead>
              <tbody>
                ${files.length ? files.map((file) => `
                  <tr>
                    <td>${escapeHtml(file.original_name || file.stored_name || "-")}</td>
                    <td>${escapeHtml(formatBytes(file.size_bytes || 0))}</td>
                    <td>${escapeHtml(file.status || "raw_uploaded")}</td>
                    <td>${escapeHtml(formatDate(file.updated_at || file.uploaded_at))}</td>
                    <td><button class="secondary-button compact" type="button" data-ingest-file="${escapeAttr(file.relative_path || "")}">Wiki + RAG</button></td>
                  </tr>
                `).join("") : `<tr><td colspan="5" class="muted">还没有上传文档。</td></tr>`}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">LLMWiki 草案</div>
            <div class="section-help">这里审查大模型整理后的知识页和安全规则草案，确认后才写入正式 Wiki。</div>
          </div>
        </div>
        <div class="panel-body band">
          <div class="scroll-panel">
            <table>
              <thead>
                <tr>
                  <th>草案</th>
                  <th>来源文件</th>
                  <th>状态</th>
                  <th>分块</th>
                  <th>更新时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                ${wikiDrafts.length ? wikiDrafts.map((draft) => `
                  <tr>
                    <td><strong>${escapeHtml(draft.title || "-")}</strong><div class="muted">${escapeHtml(draft.draft_id || "")}</div></td>
                    <td>${escapeHtml(draft.source_name || "-")}</td>
                    <td>${statusBadgeFromText(draft.status || "draft")}</td>
                    <td>${escapeHtml(`${draft.chunk_success_count ?? 0} / ${draft.chunk_count ?? 0}`)}</td>
                    <td>${escapeHtml(formatDate(draft.updated_at || draft.created_at))}</td>
                    <td>
                      <button class="secondary-button compact" type="button" data-view-wiki-draft="${escapeAttr(draft.draft_id || "")}">查看草案</button>
                      <button class="secondary-button compact" type="button" data-approve-wiki-draft="${escapeAttr(draft.draft_id || "")}">确认入库</button>
                    </td>
                  </tr>
                `).join("") : `<tr><td colspan="6" class="muted">还没有 LLMWiki 草案。</td></tr>`}
              </tbody>
            </table>
          </div>
          ${selectedWikiDraft ? renderWikiDraftReview(selectedWikiDraft) : `
            <div class="empty">上传资料后点击“Wiki + RAG”，这里会展示可审查的 Wiki 页面、规则草案和分块摘要。</div>
          `}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">RAG 文档索引</div>
            <div class="section-help">上传文件会建立本地 RAG 索引；回答时返回带来源页码的证据片段。</div>
          </div>
        </div>
        <div class="panel-body band">
          <div class="scroll-panel">
            <table>
            <thead>
              <tr>
                <th>文档</th>
                <th>来源文件</th>
                <th>状态</th>
                <th>片段 / 向量</th>
                <th>更新时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              ${ragDocuments.length ? ragDocuments.map((document) => `
                <tr>
                  <td><strong>${escapeHtml(document.title || document.original_name || "-")}</strong><div class="muted">${escapeHtml(document.document_id || "")}</div></td>
                  <td>${escapeHtml(document.original_name || "-")}</td>
                  <td>${statusBadgeFromText(document.status || "uploaded")}</td>
                  <td>${escapeHtml(`${document.chunk_count || 0} / ${document.embedded_count || 0}`)}</td>
                  <td>${escapeHtml(formatDate(document.updated_at || document.created_at))}</td>
                  <td>
                    <button class="secondary-button compact" type="button" data-view-rag="${escapeAttr(document.document_id || "")}">查看片段</button>
                    <button class="secondary-button compact" type="button" data-index-rag="${escapeAttr(document.document_id || "")}">重建索引</button>
                    <button class="secondary-button compact" type="button" data-delete-rag="${escapeAttr(document.document_id || "")}">删除</button>
                  </td>
                </tr>
              `).join("") : `<tr><td colspan="6" class="muted">还没有 RAG 文档。</td></tr>`}
            </tbody>
          </table>
          </div>
          ${selectedRagDocument ? renderRagDocumentReview(selectedRagDocument) : `
            <div class="empty">点击某条文档右侧的“查看片段”，在这里审查分块、页码、索引状态和检索效果。</div>
          `}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">规则库预览</div>
            <div class="section-help">这里展示当前启用的临床安全红线，前几条足够你快速巡检。</div>
          </div>
        </div>
        <div class="panel-body scroll-panel">
          <table>
            <thead>
              <tr>
                <th>规则 ID</th>
                <th>级别</th>
                <th>分类</th>
                <th>提示</th>
              </tr>
            </thead>
            <tbody>
              ${rules.slice(0, 20).map((rule) => `
                <tr>
                  <td>${escapeHtml(rule.rule_id || "-")}</td>
                  <td>${statusBadgeFromText(rule.severity || "-")}</td>
                  <td>${escapeHtml(rule.category || "-")}</td>
                  <td>${escapeHtml(rule.message || rule.description || "-")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">结构化营养搜索</div>
            <div class="section-help">查询单个食物的每 100g 数据，用来排查别名匹配和营养库质量。</div>
          </div>
        </div>
        <div class="panel-body band">
          <form id="food-search-form" class="search-row">
            <input name="food_query" value="${escapeAttr(state.foodQuery || "")}" placeholder="例如：鸡蛋、牛奶、白面包" />
            <button class="secondary-button" type="submit">搜索食物</button>
          </form>
          <table>
            <thead>
              <tr>
                <th>食物</th>
                <th>分类</th>
                <th>热量</th>
                <th>碳水</th>
                <th>蛋白质</th>
                <th>脂肪</th>
              </tr>
            </thead>
            <tbody>${foodTableRows}</tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">整餐营养计算</div>
            <div class="section-help">输入自然语言餐食，系统会解析份量、查询结构化营养库、汇总整餐，并可写入当前用户的每日摄入曲线。</div>
          </div>
          <span class="status-pill info">闭环测试</span>
        </div>
        <div class="panel-body band">
          <form id="meal-analysis-form" class="meal-analysis-form">
            <div class="search-row">
              <input name="meal_text" value="${escapeAttr(state.mealText || "")}" placeholder="例如：两个鸡蛋、一杯牛奶、两片白面包" />
              <button class="primary-button" type="submit">分析整餐</button>
            </div>
            <label class="field field-check compact-check">
              <input type="checkbox" name="record" ${state.mealRecord ? "checked" : ""} />
              <span>写入当前设备用户的每日摄入曲线</span>
            </label>
          </form>
          ${renderMealAnalysisResult(state.mealAnalysis)}
        </div>
      </section>
    </div>
  `;
}

function renderMealAnalysisResult(payload) {
  if (!payload) {
    return `<div class="empty">还没有整餐分析结果。可以试试“两个鸡蛋一杯牛奶两片白面包”，不加标点也可以。</div>`;
  }
  const analysis = payload.analysis || {};
  const totals = analysis.totals || {};
  const resolved = analysis.resolved_items || [];
  const unresolved = analysis.unresolved_items || [];
  const targets = payload.nutrition_targets || {};
  const effective = targets.effective || {};
  const targetCards = [
    ["energy_kcal", "热量", "kcal", effective.energy_kcal?.value],
    ["carbohydrate_g", "碳水", "g", effective.carbohydrate_g_per_meal?.value || (effective.carbohydrate_g_per_day?.value ? effective.carbohydrate_g_per_day.value / 3 : null)],
    ["protein_g", "蛋白质", "g", effective.protein_g_per_day?.value ? effective.protein_g_per_day.value / 3 : null],
    ["fat_g", "脂肪", "g", effective.fat_g_per_day?.value ? effective.fat_g_per_day.value / 3 : null],
  ];
  return `
    <div class="meal-result">
      <div class="nutrition-target-grid">
        ${targetCards.map(([key, label, unit, target]) => `
          <div class="notice nutrition-target-card">
            <div class="label">${escapeHtml(label)}</div>
            <div class="nutrition-target-value">${escapeHtml(formatCompactNumber(totals[key]))} ${escapeHtml(unit)}</div>
            <div class="muted">${target ? `本餐参考 ${formatCompactNumber(target)} ${unit}，约 ${Math.round((Number(totals[key] || 0) / Number(target || 1)) * 100)}%` : "暂无目标"}</div>
          </div>
        `).join("")}
      </div>
      <div class="notice">
        <div class="label">分析状态</div>
        <div class="tag-row" style="margin-top:10px;">
          <span class="tag">已匹配 ${resolved.length} 项</span>
          <span class="tag">未匹配 ${unresolved.length} 项</span>
          <span class="tag">${payload.recorded ? "已写入每日曲线" : "未写入每日曲线"}</span>
        </div>
      </div>
      <div class="scroll-panel">
        <table>
          <thead>
            <tr>
              <th>输入</th>
              <th>匹配食物</th>
              <th>克重</th>
              <th>热量</th>
              <th>碳水</th>
              <th>蛋白质</th>
              <th>脂肪</th>
            </tr>
          </thead>
          <tbody>
            ${resolved.length ? resolved.map((item) => `
              <tr>
                <td>${escapeHtml(item.raw_text || "-")}</td>
                <td><strong>${escapeHtml(item.matched_food || "-")}</strong><div class="muted">${escapeHtml(item.portion_source || "")}</div></td>
                <td>${escapeHtml(formatCompactNumber(item.grams))} g</td>
                <td>${escapeHtml(formatCompactNumber(item.nutrients?.energy_kcal))}</td>
                <td>${escapeHtml(formatCompactNumber(item.nutrients?.carbohydrate_g))}</td>
                <td>${escapeHtml(formatCompactNumber(item.nutrients?.protein_g))}</td>
                <td>${escapeHtml(formatCompactNumber(item.nutrients?.fat_g))}</td>
              </tr>
            `).join("") : `<tr><td colspan="7" class="muted">没有匹配到可计算食物。</td></tr>`}
          </tbody>
        </table>
      </div>
      ${unresolved.length ? `
        <div class="notice warn">
          <div class="label">未匹配食物</div>
          <div style="margin-top:8px;">${unresolved.map((item) => escapeHtml(item.raw_text || item.food_name || "-")).join("、")}</div>
        </div>
      ` : ""}
    </div>
  `;
}

function renderMemoryPage() {
  const memory = state.memory;
  if (!state.currentUserId) {
    return `<div class="empty">先选择一个设备绑定用户。</div>`;
  }
  if (!memory) {
    return `<div class="empty">记忆数据尚未加载。</div>`;
  }
  const summary = memory.short_term_summary;
  return `
    <div class="page-stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">短期记忆摘要</div>
            <div class="section-help">这是你刚刚让我们加进去的 2000 字以内压缩短期记忆。</div>
          </div>
          <span class="status-pill ${summary ? "ok" : "warn"}">${summary ? "有摘要" : "暂无摘要"}</span>
        </div>
        <div class="panel-body band">
          ${summary ? `
            <div class="tag-row">
              <span class="tag">会话 ${escapeHtml(summary.source_session_id || "-")}</span>
              <span class="tag">来源轮数 ${escapeHtml(summary.source_turn_count ?? "-")}</span>
              <span class="tag">上限 ${escapeHtml(summary.max_chars ?? "-")} 字</span>
              <span class="tag">更新时间 ${escapeHtml(formatDate(summary.updated_at))}</span>
            </div>
            <div class="notice">${escapeHtml(summary.summary || "")}</div>
          ` : `<div class="empty">当前用户还没有短期记忆摘要。</div>`}
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">Working Memory</div>
            <div class="section-help">最近工作记忆里的原始轮次。用来看系统有没有抓住“刚才那句话”的上下文。</div>
          </div>
        </div>
        <div class="panel-body scroll-panel">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>会话</th>
                <th>角色</th>
                <th>内容</th>
              </tr>
            </thead>
            <tbody>
              ${(memory.working || []).length ? (memory.working || []).map((item) => `
                <tr>
                  <td>${escapeHtml(formatDate(item.created_at))}</td>
                  <td><span class="mono">${escapeHtml(item.session_id || "-")}</span></td>
                  <td>${escapeHtml(item.role || "-")}</td>
                  <td>${escapeHtml(item.content || "")}</td>
                </tr>
              `).join("") : `<tr><td colspan="4" class="muted">暂无 working memory。</td></tr>`}
            </tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">长期记忆条目</div>
            <div class="section-help">事实、事件、语义记忆都在这张表里。</div>
          </div>
        </div>
        <div class="panel-body scroll-panel">
          <table>
            <thead>
              <tr>
                <th>层级</th>
                <th>属性</th>
                <th>值</th>
                <th>内容</th>
                <th>重要度</th>
                <th>更新时间</th>
              </tr>
            </thead>
            <tbody>
              ${(memory.structured || []).length ? (memory.structured || []).map((item) => `
                <tr>
                  <td>${escapeHtml(item.layer || "-")}</td>
                  <td>${escapeHtml(item.attribute || "-")}</td>
                  <td>${escapeHtml(item.value || "-")}</td>
                  <td>${escapeHtml(item.content || "-")}</td>
                  <td>${escapeHtml(item.importance ?? "-")}</td>
                  <td>${escapeHtml(formatDate(item.updated_at))}</td>
                </tr>
              `).join("") : `<tr><td colspan="6" class="muted">暂无长期记忆条目。</td></tr>`}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

function renderWikiDraftReview(draft) {
  const rules = draft.rules_draft?.rules || [];
  const chunks = draft.chunk_summaries || [];
  const wikiPages = draft.wiki_pages || [];
  const coverage = draft.coverage_report || {};
  const review = draft.llm_review || {};
  const quality = draft.document_quality || {};
  const profile = draft.document_profile || {};
  const plan = draft.ingestion_plan || {};
  const extraction = draft.structured_extraction || {};
  const extractionStats = extraction.stats || draft.structured_extraction_stats || {};
  const needsReviewPayload = state.selectedWikiNeedsReview;
  const draftNeedsReview = needsReviewPayload?.draft_needs_review || extraction.needs_review || [];
  const dbNeedsReview = needsReviewPayload?.db_needs_review || [];
  const needsReviewItems = [...draftNeedsReview, ...dbNeedsReview];
  const uncovered = coverage.uncovered_pages || draft.uncovered_pages || [];
  return `
    <div class="draft-review">
      <div class="draft-review-head">
        <div>
          <div class="panel-title">${escapeHtml(draft.title || "LLMWiki 草案")}</div>
          <div class="section-help">
            来源：${escapeHtml(draft.source_name || "-")}；
            状态：${escapeHtml(draft.status || "-")}；
            模式：${escapeHtml(draft.ingestion_mode || "-")}；
            分块：${escapeHtml(`${draft.chunk_success_count ?? 0} / ${draft.chunk_count ?? 0}`)}；
            Wiki 页面：${escapeHtml(draft.wiki_page_count ?? wikiPages.length ?? 0)}
          </div>
        </div>
        <div class="draft-review-actions">
          ${statusBadgeFromText(draft.status || "draft")}
          ${review.overall_status ? statusBadgeFromText(review.overall_status) : ""}
          ${draft.wiki_target_path ? `<span class="tag">${escapeHtml(draft.wiki_target_path)}</span>` : ""}
          <button class="secondary-button compact" type="button" data-regenerate-wiki-plan="${escapeAttr(draft.draft_id || "")}">重新画像</button>
          <button class="secondary-button compact" type="button" data-extract-wiki-structured="${escapeAttr(draft.draft_id || "")}">重抽结构化</button>
          <button class="secondary-button compact" type="button" data-review-wiki-draft="${escapeAttr(draft.draft_id || "")}">LLM 复核</button>
          <button class="secondary-button compact" type="button" data-load-wiki-needs-review="${escapeAttr(draft.draft_id || "")}">查看待审</button>
          <button class="secondary-button compact" type="button" data-approve-wiki-draft="${escapeAttr(draft.draft_id || "")}">确认发布</button>
        </div>
      </div>
      ${draft.llm_error ? `<div class="notice danger"><div class="label">生成诊断</div><div>${escapeHtml(draft.llm_error)}</div></div>` : ""}
      <div class="two-col">
        <div class="notice">
          <div class="label">文档质量</div>
          <div class="tag-row" style="margin-top:10px;">
            <span class="tag">状态 ${escapeHtml(quality.quality_status || draft.document_quality_status || "-")}</span>
            <span class="tag">页数 ${escapeHtml(quality.page_count ?? draft.page_count ?? "-")}</span>
            <span class="tag">可读页 ${escapeHtml(quality.readable_page_count ?? "-")}</span>
            <span class="tag">低文本页 ${escapeHtml(quality.low_text_page_count ?? "-")}</span>
            ${quality.needs_ocr ? `<span class="tag danger">需要 OCR</span>` : ""}
          </div>
          ${(quality.issues || []).length ? `<div class="hint" style="margin-top:8px;">问题：${(quality.issues || []).map((item) => escapeHtml(item)).join("；")}</div>` : ""}
        </div>
        <div class="notice">
          <div class="label">AI 文档画像</div>
          <div class="tag-row" style="margin-top:10px;">
            <span class="tag">类型 ${escapeHtml(profile.document_type || draft.document_profile_type || "-")}</span>
            <span class="tag">置信度 ${escapeHtml(profile.confidence ?? "-")}</span>
            <span class="tag">建议状态 ${escapeHtml(profile.suggested_status || "-")}</span>
          </div>
          ${(profile.knowledge_types || []).length ? `<div class="hint" style="margin-top:8px;">知识类型：${(profile.knowledge_types || []).map((item) => escapeHtml(item)).join("；")}</div>` : ""}
          ${profile.summary ? `<div class="hint" style="margin-top:8px;">${escapeHtml(profile.summary)}</div>` : ""}
        </div>
      </div>
      <div class="notice">
        <div class="label">入库计划与结构化抽取</div>
        <div class="tag-row" style="margin-top:10px;">
          <span class="tag">blocks ${escapeHtml((plan.blocks || []).length)}</span>
          ${Object.entries(extractionStats).map(([key, value]) => `<span class="tag">${escapeHtml(key)} ${escapeHtml(value)}</span>`).join("")}
          <span class="tag">needs_review ${escapeHtml(draft.needs_review_count ?? draftNeedsReview.length)}</span>
          ${needsReviewPayload ? `<span class="tag">已加载待审 ${escapeHtml(needsReviewPayload.total ?? needsReviewItems.length)}</span>` : ""}
        </div>
      </div>
      ${coverage.total_pages ? `
        <div class="notice">
          <div class="label">覆盖率报告</div>
          <div class="tag-row" style="margin-top:10px;">
            <span class="tag">总页数 ${escapeHtml(coverage.total_pages)}</span>
            <span class="tag">已覆盖 ${escapeHtml((coverage.covered_pages || []).length)}</span>
            <span class="tag">Wiki ${escapeHtml((coverage.wiki_covered_pages || []).length)}</span>
            <span class="tag">RAG ${escapeHtml((coverage.rag_covered_pages || []).length)}</span>
            <span class="tag">结构化 ${escapeHtml((coverage.structured_covered_pages || []).length)}</span>
            <span class="tag">跳过 ${escapeHtml((coverage.skipped_pages || []).length)}</span>
            <span class="tag">未覆盖 ${escapeHtml(uncovered.length)}</span>
          </div>
          ${uncovered.length ? `<div class="hint" style="margin-top:8px;">未覆盖页：${escapeHtml(uncovered.join(", "))}</div>` : `<div class="hint" style="margin-top:8px;">未覆盖页：无。</div>`}
        </div>
        ${renderCoverageReportTable(coverage)}
      ` : ""}
      ${review.overall_status ? `
        <div class="notice">
          <div class="label">LLM 复核报告</div>
          <div class="tag-row" style="margin-top:10px;">
            ${statusBadgeFromText(review.overall_status)}
            <span class="tag">${escapeHtml(review.review_method || "review")}</span>
            <span class="tag">置信度 ${escapeHtml(review.confidence ?? "-")}</span>
          </div>
          ${(review.issues || []).length ? `<div class="hint" style="margin-top:8px;">问题：${(review.issues || []).map((item) => escapeHtml(item)).join("；")}</div>` : ""}
          ${(review.recommendations || []).length ? `<div class="hint" style="margin-top:8px;">建议：${(review.recommendations || []).map((item) => escapeHtml(item)).join("；")}</div>` : ""}
        </div>
      ` : ""}
      ${needsReviewItems.length ? `
        <div class="draft-review-content">
          <div class="label" style="margin-bottom:10px;">待人工复核项</div>
          ${renderNeedsReviewTable(needsReviewItems.slice(0, 40))}
        </div>
      ` : needsReviewPayload ? `
        <div class="notice"><div class="label">待人工复核项</div><div class="hint">当前没有待审项。</div></div>
      ` : ""}
      <div class="two-col">
        <div class="draft-review-content">
          <div class="label" style="margin-bottom:10px;">Wiki 总索引草案</div>
          <pre>${escapeHtml(draft.wiki_markdown || "")}</pre>
        </div>
        <div class="draft-review-content">
          <div class="label" style="margin-bottom:10px;">规则草案</div>
          <pre>${escapeHtml(JSON.stringify({ rules }, null, 2))}</pre>
        </div>
      </div>
      ${wikiPages.length ? `
        <div class="draft-review-content">
          <div class="label" style="margin-bottom:10px;">多页 Wiki 预览</div>
          <div class="scroll-panel">
            <table>
              <thead>
                <tr>
                  <th>页面</th>
                  <th>来源页码</th>
                  <th>预览</th>
                </tr>
              </thead>
              <tbody>
                ${wikiPages.map((page) => `
                  <tr>
                    <td><strong>${escapeHtml(page.title || page.slug || "-")}</strong><div class="muted">${escapeHtml(page.path || page.slug || "")}</div></td>
                    <td>${escapeHtml((page.source_pages || []).join(", ") || "-")}</td>
                    <td><pre style="max-height:220px;">${escapeHtml(page.markdown || "")}</pre></td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      ` : ""}
      ${chunks.length ? `
        <div class="draft-review-content">
          <div class="label" style="margin-bottom:10px;">分块摘要</div>
          <pre>${escapeHtml(JSON.stringify(chunks, null, 2))}</pre>
        </div>
      ` : ""}
    </div>
  `;
}

function renderNeedsReviewTable(items) {
  return `
    <div class="scroll-panel" style="max-height:320px;">
      <table>
        <thead>
          <tr>
            <th>页码</th>
            <th>类型</th>
            <th>状态</th>
            <th>错误</th>
            <th>原文</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          ${items.map((item) => `
            <tr>
              <td>p.${escapeHtml(item.page_start ?? "-")}-${escapeHtml(item.page_end ?? item.page_start ?? "-")}</td>
              <td>${escapeHtml(item.block_type || "-")}</td>
              <td>${statusBadgeFromText(item.review_status || "pending")}</td>
              <td>${escapeHtml(item.schema_errors || item.error || "-")}</td>
              <td><pre style="max-height:160px;">${escapeHtml(item.raw_text || item.llm_output || "")}</pre></td>
              <td>
                ${item.review_id ? `
                  <div class="stacked-actions">
                    <button class="secondary-button compact" type="button" data-resolve-needs-review="${escapeAttr(item.review_id)}" data-review-status="approved">通过</button>
                    <button class="secondary-button compact" type="button" data-resolve-needs-review="${escapeAttr(item.review_id)}" data-review-status="discarded">丢弃</button>
                    <button class="secondary-button compact" type="button" data-resolve-needs-review="${escapeAttr(item.review_id)}" data-review-status="pending">待审</button>
                  </div>
                ` : `<span class="muted">草案待发布后可操作</span>`}
              </td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderCoverageReportTable(coverage) {
  const routes = coverage.page_routes || [];
  if (!routes.length) return "";
  return `
    <div class="scroll-panel" style="max-height:260px;">
      <table>
        <thead>
          <tr>
            <th>页码</th>
            <th>去向</th>
            <th>跳过原因</th>
          </tr>
        </thead>
        <tbody>
          ${routes.map((item) => `
            <tr>
              <td>p.${escapeHtml(item.page ?? "-")}</td>
              <td>${escapeHtml((item.routes || []).join(" / ") || "未覆盖")}</td>
              <td>${escapeHtml(item.skip_reason || "-")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderRagDocumentReview(document) {
  const chunks = state.selectedRagChunks || [];
  const searchResults = state.ragSearchResults || [];
  const structuredReview =
    state.structuredReviewResult?.document?.document_id === document.document_id
      ? state.structuredReviewResult
      : document.structured_review;
  return `
    <div class="draft-review">
      <div class="draft-review-head">
        <div>
          <div class="panel-title">${escapeHtml(document.title || document.original_name || "RAG 文档")}</div>
          <div class="section-help">
            文件：${escapeHtml(document.original_name || "-")}；
            状态：${escapeHtml(document.status || "-")}；
            页数：${escapeHtml(document.page_count ?? "-")}；
            字符：${escapeHtml(document.char_count ?? "-")}；
            片段：${escapeHtml(document.chunk_count ?? 0)}；
            向量：${escapeHtml(document.embedded_count ?? 0)}
          </div>
        </div>
        <div class="draft-review-actions">
          ${statusBadgeFromText(document.status || "uploaded")}
          <button class="secondary-button compact" type="button" data-review-structured="${escapeAttr(document.document_id || "")}">LLM 复核结构化抽取</button>
          <button class="secondary-button compact" type="button" data-approve-structured="${escapeAttr(document.document_id || "")}">标记抽取通过</button>
          <button class="secondary-button compact" type="button" data-index-rag="${escapeAttr(document.document_id || "")}">重建索引</button>
        </div>
      </div>
      ${document.error_message ? `<div class="notice danger"><div class="label">索引诊断</div><div>${escapeHtml(document.error_message)}</div></div>` : ""}
      ${renderStructuredReviewPanel(structuredReview)}
      <form id="rag-search-form" class="search-row">
        <input name="rag_query" value="${escapeAttr(state.ragSearchQuery || "")}" placeholder="测试检索，例如：糖尿病午餐怎么控制碳水" />
        <button class="secondary-button" type="submit">检索 RAG</button>
      </form>
      ${searchResults.length ? `
        <div class="scroll-panel">
          <table>
            <thead>
              <tr>
                <th>引用</th>
                <th>相关度</th>
                <th>片段</th>
              </tr>
            </thead>
            <tbody>
              ${searchResults.map((item) => `
                <tr>
                  <td>${escapeHtml(item.citation || "-")}</td>
                  <td>${escapeHtml(Number(item.score || 0).toFixed(3))}</td>
                  <td>${escapeHtml((item.text || "").slice(0, 260))}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      ` : `<div class="hint">输入问题后可以检查 RAG 是否命中文档中的正确页码和片段。</div>`}
      <div class="scroll-panel">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>页码</th>
              <th>章节</th>
              <th>片段</th>
            </tr>
          </thead>
          <tbody>
            ${chunks.length ? chunks.map((chunk) => `
              <tr>
                <td>${escapeHtml(chunk.chunk_index ?? "-")}</td>
                <td>${escapeHtml(chunk.page_start === chunk.page_end ? `p.${chunk.page_start}` : `pp.${chunk.page_start}-${chunk.page_end}`)}</td>
                <td>${escapeHtml(chunk.section_title || "-")}</td>
                <td>${escapeHtml((chunk.text || "").slice(0, 420))}</td>
              </tr>
            `).join("") : `<tr><td colspan="4" class="muted">还没有分块。请先建立索引；如果状态是 needs_ocr，说明这个 PDF 需要先做 OCR。</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderStructuredReviewPanel(review) {
  if (!review) {
    return `
      <div class="notice">
        <div class="label">结构化抽取复核</div>
        <div class="hint" style="margin-top:8px;">这篇文档的表格、食谱、食养方、MET 等结构化记录还没有复核。可以先点“LLM 复核结构化抽取”，再人工确认“标记抽取通过”。</div>
      </div>
    `;
  }
  const document = review.document || {};
  const status = document.review_status || review.review_status || "auto_extracted";
  const method = document.review_method || review.review_method || "";
  const reviewedAt = document.reviewed_at || review.reviewed_at || "";
  const counts = review.counts || {};
  const payload = review.review_payload || parseJsonMaybe(document.review_summary || review.review_summary);
  const issues = Array.isArray(payload?.issues) ? payload.issues : [];
  const recommendations = Array.isArray(payload?.recommendations) ? payload.recommendations : [];
  return `
    <div class="notice">
      <div class="label">结构化抽取复核</div>
      <div class="tag-row" style="margin-top:10px;">
        ${statusBadgeFromText(status)}
        ${method ? `<span class="tag">${escapeHtml(method)}</span>` : ""}
        ${reviewedAt ? `<span class="tag">${escapeHtml(formatDate(reviewedAt))}</span>` : ""}
        ${counts.guide_tables !== undefined ? `<span class="tag">表格 ${escapeHtml(counts.guide_tables)}</span>` : ""}
        ${counts.recipe_plans !== undefined ? `<span class="tag">食谱 ${escapeHtml(counts.recipe_plans)}</span>` : ""}
        ${counts.therapeutic_recipes !== undefined ? `<span class="tag">食养方 ${escapeHtml(counts.therapeutic_recipes)}</span>` : ""}
        ${counts.activity_mets !== undefined ? `<span class="tag">MET ${escapeHtml(counts.activity_mets)}</span>` : ""}
      </div>
      ${payload ? `
        <div class="hint" style="margin-top:10px;">
          结论：${escapeHtml(payload.overall_status || "-")}；
          置信度：${escapeHtml(payload.confidence ?? "-")}；
          ${payload.approved_to_use ? "LLM 判断可继续使用" : "LLM 建议人工复核"}
        </div>
        ${issues.length ? `<div class="hint" style="margin-top:8px;">问题：${issues.map((item) => escapeHtml(item)).join("；")}</div>` : ""}
        ${recommendations.length ? `<div class="hint" style="margin-top:8px;">建议：${recommendations.map((item) => escapeHtml(item)).join("；")}</div>` : ""}
      ` : `<div class="hint" style="margin-top:10px;">暂无详细复核报告。</div>`}
    </div>
  `;
}

function renderHistoryPage() {
  const sessions = state.historySessions || [];
  const session = state.selectedSession;
  const tabs = [
    ["chat", "聊天视图"],
    ["tools", "工具视图"],
    ["raw", "原始消息"],
    ["debug", "调试链路"],
  ];

  return `
    <div class="page-stack">
      <section class="panel conversation-pane">
        <div class="panel-head">
          <div class="panel-title">历史对话</div>
          <div class="tabs">
            ${tabs.map(([key, label]) => `
              <button class="tab-button ${state.historyView === key ? "active" : ""}" data-history-view="${key}" type="button">${label}</button>
            `).join("")}
          </div>
        </div>
        <div class="panel-body split">
          <div class="list scroll-panel">
            ${sessions.length ? sessions.map((item) => `
              <button class="list-item ${item.session_id === state.selectedSessionId ? "active" : ""}" data-session-id="${escapeAttr(item.session_id)}" type="button">
                <div class="list-item-title">${escapeHtml(item.title || "未命名日期")}</div>
                <div class="list-item-meta">${escapeHtml(item.preview || "")}</div>
                <div class="tag-row" style="margin-top:10px;">
                  <span class="tag">${escapeHtml(item.session_count || 1)} 次对话</span>
                  <span class="tag">${escapeHtml(item.message_count || 0)} 条消息</span>
                  <span class="tag">更新 ${escapeHtml(formatTime(item.updated_at))}</span>
                  ${item.has_tool_calls ? `<span class="tag">工具</span>` : ""}
                  ${item.has_vision ? `<span class="tag">视觉</span>` : ""}
                </div>
              </button>
            `).join("") : `<div class="empty">当前用户还没有历史对话。</div>`}
          </div>
          <div>
            ${session ? renderHistoryDetail(session) : `<div class="empty">请选择左侧会话。</div>`}
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderHistoryDetail(session) {
  const messages = Array.isArray(session.messages) ? session.messages : [];
  if (state.historyView === "raw") {
    return `
      <div class="scroll-panel">
        <pre>${escapeHtml(JSON.stringify(session, null, 2))}</pre>
      </div>
    `;
  }

  if (state.historyView === "tools") {
    const toolMessages = messages.filter((item) => item.role === "tool" || item.tool_calls);
    return toolMessages.length
      ? `<div class="message-list">${toolMessages.map(renderHistoryToolBlock).join("")}</div>`
      : `<div class="empty">这一天的对话里没有记录到工具调用。</div>`;
  }

  if (state.historyView === "debug") {
    const shortSummary = state.memory?.short_term_summary;
    const profileScalars = state.profile?.scalars || {};
    return `
      <div class="page-stack">
        <div class="notice">
          <div class="label">日期会话信息</div>
          <div class="tag-row" style="margin-top:8px;">
            <span class="tag">${escapeHtml(session.title || session.session_id)}</span>
            <span class="tag">用户 ${escapeHtml(session.user_id)}</span>
            <span class="tag">${escapeHtml(session.session_count || 1)} 次连接</span>
            <span class="tag">${escapeHtml(session.message_count ?? messages.length)} 条消息</span>
          </div>
        </div>
        <div class="notice">
          <div class="label">短期记忆摘要</div>
          <div style="margin-top:8px;">${escapeHtml(shortSummary?.summary || "暂无")}</div>
        </div>
        <div class="notice">
          <div class="label">健康档案注入线索</div>
          <div style="margin-top:8px;">${escapeHtml(formatObjectInline(profileScalars) || "暂无结构化档案")}</div>
        </div>
        <div class="notice">
          <div class="label">工具 / 视觉痕迹</div>
          <div style="margin-top:8px;">${escapeHtml(messages.filter((item) => item.role === "tool" || item.tool_calls).map(summarizeMessage).join(" | ") || "未记录")}</div>
        </div>
        <div class="notice">
          <div class="label">最终消息回放</div>
          <div style="margin-top:8px;">${escapeHtml(messages.slice(-4).map(summarizeMessage).join("\n"))}</div>
        </div>
      </div>
    `;
  }

  const visibleMessages = messages.filter(
    (item) => !item.is_temporary && (item.role === "user" || item.role === "assistant"),
  );
  return `
    <div class="page-stack">
      <div class="notice">
        <div class="label">日期</div>
        <div class="tag-row" style="margin-top:8px;">
          <span class="tag">${escapeHtml(session.title || session.session_id)}</span>
          <span class="tag">${escapeHtml(session.session_count || 1)} 次连接</span>
          <span class="tag">${escapeHtml(visibleMessages.length)} 条对话</span>
        </div>
      </div>
      <div class="message-list scroll-panel">
        ${visibleMessages.length ? visibleMessages.map(renderHistoryBubble).join("") : `<div class="empty">会话内容为空。</div>`}
      </div>
    </div>
  `;
}

function renderHistoryBubble(item) {
  const role = item.role || "assistant";
  const displayText = extractHistoryMessageText(item) || roleSummary(role);
  return `
    <article class="bubble ${escapeAttr(role)}">
      <div class="bubble-head">${escapeHtml(formatHistoryRole(role))} · ${escapeHtml(formatDate(item.created_at))}</div>
      <div class="bubble-body">${escapeHtml(displayText)}</div>
    </article>
  `;
}

function renderHistoryToolBlock(item) {
  const displayPayload = item.tool_calls || item;
  return `
    <div class="notice">
      <div class="label">${escapeHtml(formatHistoryRole(item.role || "tool"))} · ${escapeHtml(formatDate(item.created_at))}</div>
      <pre style="margin-top:10px;">${escapeHtml(JSON.stringify(displayPayload, null, 2))}</pre>
    </div>
  `;
}

function bindSettingsEvents() {
  const form = document.getElementById("agent-settings-form");
  if (!form) return;
  form.querySelectorAll('input[type="range"]').forEach((input) => {
    const output = form.querySelector(`[data-slider-output="${input.name}"]`);
    const sync = () => {
      if (output) output.textContent = Number(input.value).toFixed(2);
    };
    input.addEventListener("input", sync);
    sync();
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = serializeForm(form);
    try {
      await apiPost("/console/api/agent-settings", payload);
      showToast("Agent 设置已保存，重启服务后会完全生效。", "success");
      await Promise.all([loadAgentSettings(), loadUserScopedData()]);
      renderAll();
    } catch (error) {
      showToast(error.message || "保存失败", "error");
    }
  });
}

function bindModelSettingsEvents() {
  const form = document.getElementById("model-config-form");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = serializeForm(form);
    try {
      await apiPost("/console/api/model-config", payload);
      showToast("模型配置已保存，重启服务后会完全生效。", "success");
      await loadModelConfig();
      renderAll();
    } catch (error) {
      showToast(error.message || "保存失败", "error");
    }
  });
}

function bindProfileEvents() {
  pageContentEl.querySelectorAll("[data-resolve-review]").forEach((button) => {
    button.addEventListener("click", async () => {
      const reviewId = button.dataset.resolveReview || "";
      const decision = button.dataset.decision || "";
      if (!reviewId || !decision) return;
      button.disabled = true;
      try {
        await apiPost(
          `/console/api/profile-review/${encodeURIComponent(reviewId)}/resolve`,
          { decision },
        );
        showToast(decision === "accept" ? "档案更新已确认。" : "档案冲突已忽略。", "success");
        await loadUserScopedData();
        await loadSummary();
        renderAll();
      } catch (error) {
        showToast(error.message || "处理待确认项失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });
}

function bindKnowledgeEvents() {
  const uploadForm = document.getElementById("upload-form");
  const foodForm = document.getElementById("food-search-form");
  const mealForm = document.getElementById("meal-analysis-form");

  if (uploadForm) {
    const input = document.getElementById("upload-input");
    const fileNameEl = document.getElementById("upload-file-name");
    const syncSelectedFiles = () => {
      if (!fileNameEl || !input) return;
      const files = input.files ? Array.from(input.files) : [];
      if (!files.length) {
        fileNameEl.textContent = "未选择文件";
        return;
      }
      fileNameEl.textContent = files.length === 1 ? files[0].name : `已选择 ${files.length} 个文件`;
    };
    input?.addEventListener("change", syncSelectedFiles);
    syncSelectedFiles();

    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const files = input?.files ? Array.from(input.files) : [];
      if (!files.length) {
        showToast("先选择要上传的文档。", "error");
        return;
      }
      const formData = new FormData();
      files.forEach((file) => formData.append("file", file, file.name));
      try {
        await apiUpload("/console/api/knowledge/upload", formData);
        showToast("文档已上传到素材层。", "success");
        input.value = "";
        syncSelectedFiles();
        await Promise.all([loadKnowledgeData(), loadSummary()]);
        renderAll();
      } catch (error) {
        showToast(error.message || "上传失败", "error");
      }
    });
  }

  pageContentEl.querySelectorAll("[data-ingest-file]").forEach((button) => {
    button.addEventListener("click", async () => {
      const relativePath = button.dataset.ingestFile || "";
      if (!relativePath) return;
      button.disabled = true;
      try {
        const payload = await apiPost("/console/api/knowledge/ingest", { relative_path: relativePath });
        const document = payload.document || null;
        const draft = payload.draft || null;
        if (document?.document_id) {
          state.selectedRagDocumentId = document.document_id;
          state.selectedRagChunks = [];
        }
        if (draft?.draft_id) {
          state.selectedWikiDraftId = draft.draft_id;
          state.selectedWikiDraft = draft;
        }
        const warnings = [payload.draft_error, payload.rag_error].filter(Boolean);
        showToast(
          warnings.length ? `部分完成：${warnings.join("；")}` : "已生成 Wiki 草案并启动 RAG 索引。",
          warnings.length ? "error" : "success"
        );
        await loadKnowledgeData();
        if (state.selectedWikiDraftId) {
          await loadWikiDraft(state.selectedWikiDraftId);
        }
        if (state.selectedRagDocumentId) {
          await loadRagChunks(state.selectedRagDocumentId);
        }
        renderAll();
      } catch (error) {
        showToast(error.message || "Wiki + RAG 入库失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-view-wiki-draft]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.viewWikiDraft || "";
      if (!draftId) return;
      button.disabled = true;
      try {
        await loadWikiDraft(draftId);
        renderAll();
      } catch (error) {
        showToast(error.message || "读取 Wiki 草案失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-regenerate-wiki-plan]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.regenerateWikiPlan || "";
      if (!draftId) return;
      button.disabled = true;
      try {
        const payload = await apiPost(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}/regenerate-plan`, {});
        showToast(`已重新生成入库计划，待审 ${payload.needs_review_count ?? 0} 项。`, "success");
        state.selectedWikiDraftId = draftId;
        await loadWikiDraft(draftId);
        await loadKnowledgeData();
        renderAll();
      } catch (error) {
        showToast(error.message || "重新画像失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-extract-wiki-structured]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.extractWikiStructured || "";
      if (!draftId) return;
      button.disabled = true;
      try {
        const payload = await apiPost(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}/extract-structured`, {});
        showToast(`结构化抽取已完成，待审 ${payload.needs_review_count ?? 0} 项。`, "success");
        state.selectedWikiDraftId = draftId;
        await loadWikiDraft(draftId);
        await loadWikiDraftNeedsReview(draftId);
        await loadKnowledgeData();
        renderAll();
      } catch (error) {
        showToast(error.message || "结构化抽取失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-review-wiki-draft]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.reviewWikiDraft || "";
      if (!draftId) return;
      button.disabled = true;
      try {
        const payload = await apiPost(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}/review`, {});
        state.selectedWikiDraftId = draftId;
        state.selectedWikiDraft = payload.draft || null;
        showToast(`LLM 复核完成：${payload.review?.overall_status || "已完成"}`, "success");
        await loadKnowledgeData();
        renderAll();
      } catch (error) {
        showToast(error.message || "LLM 复核失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-load-wiki-needs-review]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.loadWikiNeedsReview || "";
      if (!draftId) return;
      button.disabled = true;
      try {
        await loadWikiDraftNeedsReview(draftId);
        showToast("待审项已加载。", "success");
        renderAll();
      } catch (error) {
        showToast(error.message || "读取待审项失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-resolve-needs-review]").forEach((button) => {
    button.addEventListener("click", async () => {
      const reviewId = button.dataset.resolveNeedsReview || "";
      const reviewStatus = button.dataset.reviewStatus || "";
      const draftId = state.selectedWikiDraftId || "";
      if (!reviewId || !reviewStatus) return;
      button.disabled = true;
      try {
        await apiPost(`/console/api/clinical-knowledge/needs-review/${encodeURIComponent(reviewId)}/resolve`, {
          status: reviewStatus,
          reviewer_notes: "console_manual_review",
        });
        if (draftId) {
          await loadWikiDraftNeedsReview(draftId);
        }
        showToast(`待审项已标记为 ${reviewStatus}。`, "success");
        renderAll();
      } catch (error) {
        showToast(error.message || "更新待审状态失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-approve-wiki-draft]").forEach((button) => {
    button.addEventListener("click", async () => {
      const draftId = button.dataset.approveWikiDraft || "";
      if (!draftId) return;
      if (!window.confirm("确认把这个 Wiki 草案写入正式知识库，并把规则草案追加到待审规则库？")) {
        return;
      }
      button.disabled = true;
      try {
        const payload = await apiPost(`/console/api/knowledge/ingestion/drafts/${encodeURIComponent(draftId)}/approve`, {});
        state.selectedWikiDraftId = draftId;
        state.selectedWikiDraft = payload.draft || null;
        showToast("Wiki 草案已确认入库。", "success");
        await Promise.all([loadKnowledgeData(), loadSummary()]);
        if (state.selectedWikiDraftId) {
          await loadWikiDraft(state.selectedWikiDraftId);
        }
        renderAll();
      } catch (error) {
        showToast(error.message || "确认入库失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-index-rag]").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.dataset.indexRag || "";
      if (!documentId) return;
      button.disabled = true;
      try {
        await apiPost(`/console/api/rag/documents/${encodeURIComponent(documentId)}/index`, {});
        state.selectedRagDocumentId = documentId;
        showToast("已启动 RAG 重建索引任务。", "success");
        await loadKnowledgeData();
        await loadRagChunks(documentId);
        renderAll();
      } catch (error) {
        showToast(error.message || "重建索引失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-view-rag]").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.dataset.viewRag || "";
      if (!documentId) return;
      button.disabled = true;
      try {
        await loadRagChunks(documentId);
        renderAll();
      } catch (error) {
        showToast(error.message || "读取 RAG 片段失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-review-structured]").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.dataset.reviewStructured || "";
      if (!documentId) return;
      button.disabled = true;
      try {
        showToast("正在调用文档入库模型复核结构化抽取结果。", "info");
        const payload = await apiPost(`/console/api/clinical-knowledge/documents/${encodeURIComponent(documentId)}/llm-review`, {});
        state.structuredReviewResult = payload.review || null;
        await Promise.all([loadKnowledgeData(), loadSummary()]);
        await loadRagChunks(documentId);
        showToast("结构化抽取复核已完成。", "success");
        renderAll();
      } catch (error) {
        showToast(error.message || "LLM 复核失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-approve-structured]").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.dataset.approveStructured || "";
      if (!documentId) return;
      if (!window.confirm("确认把这篇文档的结构化抽取结果标记为已通过？这不会改写原文，只会更新审核状态。")) {
        return;
      }
      button.disabled = true;
      try {
        const payload = await apiPost(`/console/api/clinical-knowledge/documents/${encodeURIComponent(documentId)}/approve`, {});
        state.structuredReviewResult = payload.review || null;
        await Promise.all([loadKnowledgeData(), loadSummary()]);
        await loadRagChunks(documentId);
        showToast("结构化抽取结果已标记通过。", "success");
        renderAll();
      } catch (error) {
        showToast(error.message || "标记通过失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  pageContentEl.querySelectorAll("[data-delete-rag]").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.dataset.deleteRag || "";
      if (!documentId) return;
      if (!window.confirm("确认删除这个 RAG 文档及其所有片段和向量索引？")) {
        return;
      }
      button.disabled = true;
      try {
        await apiDelete(`/console/api/rag/documents/${encodeURIComponent(documentId)}`);
        if (state.selectedRagDocumentId === documentId) {
          state.selectedRagDocumentId = "";
          state.selectedRagChunks = [];
          state.ragSearchResults = [];
        }
        showToast("RAG 文档已删除。", "success");
        await Promise.all([loadKnowledgeData(), loadSummary()]);
        renderAll();
      } catch (error) {
        showToast(error.message || "删除 RAG 文档失败", "error");
      } finally {
        button.disabled = false;
      }
    });
  });

  const ragSearchForm = pageContentEl.querySelector("#rag-search-form");
  if (ragSearchForm) {
    ragSearchForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = ragSearchForm.elements.rag_query.value.trim();
      state.ragSearchQuery = query;
      if (!query) {
        state.ragSearchResults = [];
        renderAll();
        return;
      }
      try {
        const payload = await apiPost("/console/api/rag/search", {
          question: query,
          top_k: 6,
        });
        state.ragSearchResults = payload.results || [];
        renderAll();
      } catch (error) {
        showToast(error.message || "RAG 检索失败", "error");
      }
    });
  }

  if (foodForm) {
    foodForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = foodForm.elements.food_query.value.trim();
      state.foodQuery = query;
      if (!query) {
        state.foodResults = [];
        renderAll();
        return;
      }
      try {
        const payload = await apiGet(`/console/api/food?q=${encodeURIComponent(query)}&limit=12`);
        state.foodResults = payload.foods || [];
        renderAll();
      } catch (error) {
        showToast(error.message || "搜索失败", "error");
      }
    });
  }

  if (mealForm) {
    mealForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const mealText = mealForm.elements.meal_text.value.trim();
      const record = Boolean(mealForm.elements.record?.checked);
      state.mealText = mealText;
      state.mealRecord = record;
      if (!mealText) {
        showToast("先输入要分析的一餐。", "error");
        return;
      }
      try {
        const payload = await apiPost("/console/api/meal/analyze", {
          user_id: state.currentUserId,
          meal_text: mealText,
          record,
        });
        state.mealAnalysis = payload;
        if (record) {
          await Promise.all([loadUserScopedData(), loadSummary()]);
        }
        renderAll();
        showToast(record ? "整餐已分析并写入每日曲线。" : "整餐分析完成。", "success");
      } catch (error) {
        showToast(error.message || "整餐分析失败", "error");
      }
    });
  }
}

function bindHistoryEvents() {
  pageContentEl.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await loadSelectedSession(button.dataset.sessionId);
        renderAll();
      } catch (error) {
        showToast(error.message || "读取会话失败", "error");
      }
    });
  });
  pageContentEl.querySelectorAll("[data-history-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.historyView = button.dataset.historyView;
      renderAll();
    });
  });
}

function renderLoading(text) {
  pageContentEl.innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
}

function renderError(error) {
  pageContentEl.innerHTML = `
    <div class="notice danger">
      <div class="panel-title">控制台加载失败</div>
      <div style="margin-top:8px;">${escapeHtml(error.message || String(error))}</div>
    </div>
  `;
}

function renderModelEditorSection(title, note, rootKey, data) {
  return `
    <div class="notice">
      <div class="label">${escapeHtml(title)}</div>
      <div class="hint" style="margin-top:6px;">${escapeHtml(note)}</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>模块</span>
          <select name="${rootKey}.module">${optionList(data?.available_modules || [], data?.module)}</select>
        </label>
        <label class="field">
          <span>模型名</span>
          <input name="${rootKey}.model_name" value="${escapeAttr(data?.model_name || "")}" />
        </label>
        <label class="field">
          <span>${escapeHtml(data?.endpoint_field || "base_url")}</span>
          <input name="${rootKey}.endpoint_url" value="${escapeAttr(data?.endpoint_url || "")}" />
        </label>
        <label class="field">
          <span>API Key</span>
          <input name="${rootKey}.api_key" type="password" placeholder="${data?.api_key_configured ? "已配置，留空则不改" : "未配置"}" />
        </label>
      </div>
    </div>
  `;
}

function renderIntentSection(data) {
  return `
    <div class="notice">
      <div class="label">Intent LLM / 工具决策</div>
      <div class="hint" style="margin-top:6px;">负责判断是否调用营养查询、知识库、视觉或其他工具。</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>Intent 模块</span>
          <select name="intent.module">${optionList(data?.available_modules || [], data?.module)}</select>
        </label>
        <label class="field">
          <span>独立 Intent LLM</span>
          <select name="intent.dedicated_llm">${optionList(data?.available_llm_modules || [], data?.dedicated_llm)}</select>
        </label>
      </div>
    </div>
  `;
}

function renderSimpleModelSection(title, note, rootKey, data) {
  return `
    <div class="notice">
      <div class="label">${escapeHtml(title)}</div>
      <div class="hint" style="margin-top:6px;">${escapeHtml(note)}</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>provider</span>
          <input name="${rootKey}.provider" value="${escapeAttr(data?.provider || "")}" />
        </label>
        <label class="field">
          <span>model</span>
          <input name="${rootKey}.model" value="${escapeAttr(data?.model || "")}" />
        </label>
        <label class="field">
          <span>openai_base_url</span>
          <input name="${rootKey}.openai_base_url" value="${escapeAttr(data?.openai_base_url || "")}" />
        </label>
        <label class="field">
          <span>API Key</span>
          <input name="${rootKey}.api_key" type="password" placeholder="${data?.api_key_configured ? "已配置，留空则不改" : "未配置"}" />
        </label>
      </div>
    </div>
  `;
}

function renderEmbeddingSection(data) {
  return `
    <div class="notice">
      <div class="label">Embedding</div>
      <div class="hint" style="margin-top:6px;">负责向量检索，不直接负责聊天。</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>provider</span>
          <input name="embedding.provider" value="${escapeAttr(data?.provider || "")}" />
        </label>
        <label class="field">
          <span>model</span>
          <input name="embedding.model" value="${escapeAttr(data?.model || "")}" />
        </label>
        <label class="field">
          <span>openai_base_url</span>
          <input name="embedding.openai_base_url" value="${escapeAttr(data?.openai_base_url || "")}" />
        </label>
        <label class="field">
          <span>embedding_dims</span>
          <input name="embedding.embedding_dims" value="${escapeAttr(data?.embedding_dims ?? "")}" />
        </label>
        <label class="field">
          <span>API Key</span>
          <input name="embedding.api_key" type="password" placeholder="${data?.api_key_configured ? "已配置，留空则不改" : "未配置"}" />
        </label>
      </div>
    </div>
  `;
}

function renderSimpleMem0Section(data) {
  return `
    <div class="notice">
      <div class="label">mem0</div>
      <div class="hint" style="margin-top:6px;">当前长期记忆引擎的 mem0 接入层。</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>mode</span>
          <input name="mem0.mode" value="${escapeAttr(data?.mode || "")}" />
        </label>
        <label class="field">
          <span>host</span>
          <input name="mem0.host" value="${escapeAttr(data?.host || "")}" />
        </label>
        <label class="field">
          <span>API Key</span>
          <input name="mem0.api_key" type="password" placeholder="${data?.api_key_configured ? "已配置，留空则不改" : "未配置"}" />
        </label>
      </div>
    </div>
  `;
}

function renderVectorStoreSection(data) {
  return `
    <div class="notice">
      <div class="label">Vector Store</div>
      <div class="hint" style="margin-top:6px;">PowerMem 检索层当前使用的向量库配置。</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>provider</span>
          <input name="vector_store.provider" value="${escapeAttr(data?.provider || "")}" />
        </label>
        <label class="field">
          <span>database_path</span>
          <input name="vector_store.database_path" value="${escapeAttr(data?.database_path || "")}" />
        </label>
        <label class="field">
          <span>collection_name</span>
          <input name="vector_store.collection_name" value="${escapeAttr(data?.collection_name || "")}" />
        </label>
        <label class="field">
          <span>embedding_model_dims</span>
          <input name="vector_store.embedding_model_dims" value="${escapeAttr(data?.embedding_model_dims ?? "")}" />
        </label>
      </div>
    </div>
  `;
}

function renderRagSettingsSection(data) {
  return `
    <div class="notice">
      <div class="label">Clinical RAG / 文档索引</div>
      <div class="hint" style="margin-top:6px;">负责把上传资料解析、分块、向量化，并在对话中返回带页码的证据片段。</div>
      <div class="band" style="margin-top:12px;">
        <label class="field field-check">
          <input type="checkbox" name="clinical_rag.enabled" ${data?.enabled ? "checked" : ""} />
          <span>启用本地 Clinical RAG</span>
        </label>
        <div class="two-col">
          <label class="field">
            <span>db_path</span>
            <input name="clinical_rag.db_path" value="${escapeAttr(data?.db_path || "")}" />
          </label>
          <label class="field">
            <span>chunk_chars</span>
            <input name="clinical_rag.chunk_chars" value="${escapeAttr(data?.chunk_chars || "")}" />
          </label>
          <label class="field">
            <span>chunk_overlap_chars</span>
            <input name="clinical_rag.chunk_overlap_chars" value="${escapeAttr(data?.chunk_overlap_chars || "")}" />
          </label>
          <label class="field">
            <span>top_k</span>
            <input name="clinical_rag.top_k" value="${escapeAttr(data?.top_k || "")}" />
          </label>
          <label class="field">
            <span>embedding_provider</span>
            <input name="clinical_rag.embedding_provider" value="${escapeAttr(data?.embedding_provider || "")}" />
          </label>
          <label class="field">
            <span>embedding_model</span>
            <input name="clinical_rag.embedding_model" value="${escapeAttr(data?.embedding_model || "")}" />
          </label>
          <label class="field">
            <span>embedding_openai_base_url</span>
            <input name="clinical_rag.embedding_openai_base_url" value="${escapeAttr(data?.embedding_openai_base_url || "")}" />
          </label>
          <label class="field">
            <span>embedding_dimensions</span>
            <input name="clinical_rag.embedding_dimensions" value="${escapeAttr(data?.embedding_dimensions || "")}" />
          </label>
          <label class="field">
            <span>Embedding API Key</span>
            <input name="clinical_rag.api_key" type="password" placeholder="${data?.api_key_configured ? "已配置，留空则不改" : "未配置"}" />
          </label>
        </div>
      </div>
    </div>
  `;
}

function renderRuntimeModelSection(title, note, rootKey, data) {
  return `
    <div class="notice">
      <div class="label">${escapeHtml(title)}</div>
      <div class="hint" style="margin-top:6px;">${escapeHtml(note)}</div>
      <div class="two-col" style="margin-top:12px;">
        <label class="field">
          <span>模块</span>
          <select name="${rootKey}.module">${optionList(data?.available_modules || [], data?.module)}</select>
        </label>
        <label class="field">
          <span>model_name</span>
          <input name="${rootKey}.model_name" value="${escapeAttr(data?.model_name || "")}" />
        </label>
        <label class="field">
          <span>model</span>
          <input name="${rootKey}.model" value="${escapeAttr(data?.model || "")}" />
        </label>
        <label class="field">
          <span>${escapeHtml(data?.endpoint_field || "base_url")}</span>
          <input name="${rootKey}.endpoint_url" value="${escapeAttr(data?.endpoint_url || "")}" />
        </label>
        <label class="field">
          <span>voice</span>
          <input name="${rootKey}.voice" value="${escapeAttr(data?.voice || "")}" />
        </label>
        <label class="field">
          <span>speaker</span>
          <input name="${rootKey}.speaker" value="${escapeAttr(data?.speaker || "")}" />
        </label>
      </div>
    </div>
  `;
}

function sliderBox(label, name, value, meta) {
  return `
    <div class="slider-box">
      <div class="slider-head">
        <span class="label">${escapeHtml(label)}</span>
        <span class="slider-value" data-slider-output="${escapeAttr(name)}">${Number(value).toFixed(2)}</span>
      </div>
      <input
        type="range"
        name="${escapeAttr(name)}"
        min="${meta.min}"
        max="${meta.max}"
        step="${meta.step}"
        value="${escapeAttr(value)}"
      />
      <div class="hint" style="margin-top:8px;">字段：${escapeHtml(meta.field)}</div>
    </div>
  `;
}

function sliderMeta(field, kind) {
  const defaults = {
    rate: { min: 0.5, max: 2.0, step: 0.05 },
    pitch: { min: 0.5, max: 2.0, step: 0.05 },
    volume: { min: 0.0, max: 2.0, step: 0.05 },
  };
  if (field === "pitch_factor" || field === "pitch") {
    return { field, min: -12, max: 12, step: 0.5 };
  }
  if (field === "volume_change_dB") {
    return { field, min: -12, max: 12, step: 0.5 };
  }
  return { field: field || kind, ...defaults[kind] };
}

function metricCard(label, value, note) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-note">${escapeHtml(note)}</div>
    </div>
  `;
}

function roleSummary(role) {
  if (role === "tool") return "[tool output]";
  if (role === "system") return "[system prompt]";
  return "";
}

function statusBadgeFromText(text) {
  const lowered = String(text || "").toLowerCase();
  let cls = "info";
  if (lowered.includes("启用") || lowered.includes("direct") || lowered.includes("configured") || lowered.includes("ok") || lowered.includes("indexed") || lowered.includes("passed") || lowered.includes("approved")) cls = "ok";
  if (lowered.includes("warn") || lowered.includes("待") || lowered.includes("mock") || lowered.includes("缺") || lowered.includes("fallback") || lowered.includes("needs_review") || lowered.includes("unavailable")) cls = "warn";
  if (lowered.includes("error") || lowered.includes("danger") || lowered.includes("failed")) cls = "danger";
  return `<span class="status-pill ${cls}">${escapeHtml(text || "-")}</span>`;
}

function optionList(options, selected, labels = {}) {
  const unique = Array.from(new Set((options || []).filter(Boolean)));
  if (selected && !unique.includes(selected)) unique.unshift(selected);
  return unique
    .map((value) => {
      const label = labels[value] || value;
      return `<option value="${escapeAttr(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(label)}</option>`;
    })
    .join("");
}

function serializeForm(form) {
  const payload = {};
  form.querySelectorAll("[name]").forEach((element) => {
    const { name } = element;
    if (!name) return;
    let value;
    if (element.type === "checkbox") {
      value = element.checked;
    } else {
      value = element.value;
    }
    assignByPath(payload, name, value);
  });
  return payload;
}

function assignByPath(target, path, value) {
  const parts = path.split(".");
  let cursor = target;
  parts.forEach((part, index) => {
    if (index === parts.length - 1) {
      cursor[part] = value;
      return;
    }
    if (!cursor[part] || typeof cursor[part] !== "object") {
      cursor[part] = {};
    }
    cursor = cursor[part];
  });
}

function formatGlucoseType(type) {
  const labels = {
    fasting: "空腹",
    pre_meal: "餐前",
    post_meal: "餐后",
    postprandial_2h: "餐后2小时",
    bedtime: "睡前",
    random: "随机",
  };
  return labels[type] || type || "-";
}

function formatGlucoseValue(value) {
  const number = Number(value);
  if (Number.isNaN(number)) return "-";
  return `${number.toFixed(1).replace(/\.0$/, "")} mmol/L`;
}

function formatReviewName(item) {
  if (!item) return "-";
  if (item.field_type === "scalar") {
    return formatDisplayKey(item.name || "");
  }
  return `${formatProfileCategory(item.category || "other")} / ${item.name || "-"}`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function formatDateOnly(value) {
  if (!value) return "未分组日期";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10) || "未分组日期";
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function formatTime(value) {
  if (!value) return "--:--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function pad(value) {
  return String(value).padStart(2, "0");
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") return "-";
  if (Array.isArray(value)) {
    const items = value.map((item) => formatValue(item)).filter((item) => item && item !== "-");
    return items.length ? items.join("、") : "-";
  }
  if (typeof value === "object") {
    const pairs = Object.entries(value).filter(([, item]) => {
      if (item === undefined || item === null || item === "") return false;
      if (Array.isArray(item)) return item.length > 0;
      if (typeof item === "object") return Object.keys(item).length > 0;
      return true;
    });
    if (!pairs.length) return "-";
    return pairs
      .map(([key, item]) => `${formatDisplayKey(key)}：${formatValue(item)}`)
      .join("；");
  }
  return String(value);
}

function formatObjectInline(object) {
  if (!object || typeof object !== "object") return "";
  const parts = Object.entries(object)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${formatDisplayKey(key)}: ${formatValue(value)}`);
  return parts.join("; ");
}

function formatBytes(bytes) {
  const size = Number(bytes || 0);
  if (!size) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 10 || unit === "B" ? 0 : 1)} ${unit}`;
}

function parseJsonMaybe(value) {
  if (!value) return null;
  try {
    return JSON.parse(String(value));
  } catch {
    return null;
  }
}

function summarizeMessage(item) {
  const role = item.role || "assistant";
  const content = extractHistoryMessageText(item) || (item.tool_calls ? "[tool calls]" : "");
  return `${role}: ${content}`.slice(0, 140);
}

function buildHistoryBuckets(records) {
  const buckets = new Map();
  [...(records || [])]
    .sort((left, right) => compareDateValue(right.updated_at || right.created_at, left.updated_at || left.created_at))
    .forEach((record) => {
      const dateKey = formatDateOnly(record.updated_at || record.created_at);
      const cleanPreview = cleanHistorySnippet(record.preview || "") || cleanHistorySnippet(record.title || "");
      const existing = buckets.get(dateKey) || {
        session_id: dateKey,
        title: dateKey,
        preview: "",
        user_id: record.user_id || state.currentUserId,
        device_id: record.device_id || "",
        created_at: record.created_at || record.updated_at || "",
        updated_at: record.updated_at || record.created_at || "",
        message_count: 0,
        has_tool_calls: false,
        has_vision: false,
        session_count: 0,
        session_ids: [],
      };

      existing.session_ids.push(record.session_id);
      existing.session_count += 1;
      existing.message_count += Number(record.message_count || 0);
      existing.has_tool_calls = existing.has_tool_calls || Boolean(record.has_tool_calls);
      existing.has_vision = existing.has_vision || Boolean(record.has_vision);
      if (record.updated_at && compareDateValue(record.updated_at, existing.updated_at) > 0) {
        existing.updated_at = record.updated_at;
      }
      if (record.created_at && compareDateValue(record.created_at, existing.created_at) < 0) {
        existing.created_at = record.created_at;
      }
      if (!existing.preview && cleanPreview) {
        existing.preview = cleanPreview;
      }
      buckets.set(dateKey, existing);
    });

  return [...buckets.values()]
    .map((bucket) => ({
      ...bucket,
      preview: bucket.preview || `${bucket.session_count} 次对话记录`,
    }))
    .sort((left, right) => compareDateValue(right.updated_at || right.created_at, left.updated_at || left.created_at));
}

function compareDateValue(left, right) {
  const leftTime = left ? new Date(left).getTime() : 0;
  const rightTime = right ? new Date(right).getTime() : 0;
  const safeLeft = Number.isNaN(leftTime) ? 0 : leftTime;
  const safeRight = Number.isNaN(rightTime) ? 0 : rightTime;
  return safeLeft - safeRight;
}

function cleanHistorySnippet(value) {
  const text = extractStructuredText(value);
  if (!text) return "";
  const compact = text.replace(/\s+/g, " ").trim();
  if (!compact) return "";
  if (compact.startsWith("You are ") || compact.startsWith("你是“") || compact.startsWith("你是\"")) {
    return "";
  }
  if (compact.length <= 36) return compact;
  return `${compact.slice(0, 36).trimEnd()}...`;
}

function extractHistoryMessageText(item) {
  if (!item) return "";
  return extractStructuredText(item.content);
}

function extractStructuredText(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "object") {
    if (typeof value.content === "string" && value.content.trim()) {
      return stripMarkdown(value.content.trim());
    }
    if (typeof value.response === "string" && value.response.trim()) {
      return stripMarkdown(value.response.trim());
    }
    return stripMarkdown(formatValue(value));
  }

  const raw = String(value).trim();
  if (!raw) return "";

  const parsed = tryParseJson(raw);
  if (parsed && typeof parsed === "object") {
    if (typeof parsed.content === "string" && parsed.content.trim()) {
      return stripMarkdown(parsed.content.trim());
    }
    if (typeof parsed.response === "string" && parsed.response.trim()) {
      return stripMarkdown(parsed.response.trim());
    }
  }

  return stripMarkdown(raw);
}

function tryParseJson(value) {
  if (!value || typeof value !== "string") return null;
  const text = value.trim();
  if (!(text.startsWith("{") || text.startsWith("["))) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function stripMarkdown(value) {
  return String(value || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/`{1,3}(.*?)`{1,3}/g, "$1")
    .replace(/^#+\s*/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function formatHistoryRole(role) {
  if (role === "user") return "用户";
  if (role === "assistant") return "Agent";
  if (role === "tool") return "工具";
  if (role === "system") return "系统";
  return role || "-";
}

function formatProfileCategory(category) {
  return PROFILE_CATEGORY_LABELS[category] || formatDisplayKey(category || "其他");
}

function formatProfileSource(source) {
  return PROFILE_SOURCE_LABELS[source] || formatDisplayKey(source || "-");
}

function formatProfileValue(value) {
  return formatValue(value);
}

function formatNutritionTarget(target) {
  if (!target || target.value === undefined || target.value === null || target.value === "") {
    return "-";
  }
  return `${formatCompactNumber(target.value)} ${target.unit || ""}`.trim();
}

function formatNutrientWithPercent(value, unit, percent) {
  return `${formatCompactNumber(value)} ${unit} · ${formatCompactNumber(percent)}%`;
}

function formatCompactNumber(value) {
  const number = numberValue(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number - Math.round(number)) < 0.05) {
    return String(Math.round(number));
  }
  return number.toFixed(1);
}

function formatNutritionMethod(method) {
  const labels = {
    mifflin_st_jeor: "Mifflin-St Jeor",
    sex_weight_kcal_per_kg: "性别+体重粗估",
  };
  return labels[method] || method || "-";
}

function formatActivityBucket(activity) {
  const labels = {
    sedentary: "久坐",
    light: "轻体力",
    moderate: "中等活动",
    active: "高活动量",
  };
  return labels[activity] || activity || "-";
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function percentOf(value, target) {
  if (!target || target <= 0) return 0;
  return (numberValue(value) / target) * 100;
}

function formatDisplayKey(value) {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const labels = {
    age_years: "年龄",
    sex: "性别",
    height_cm: "身高",
    weight_kg: "体重",
    target_energy_kcal: "目标热量",
    target_carbohydrate_g_per_meal: "每餐碳水目标",
    target_protein_g_per_day: "每日蛋白目标",
    target_fat_g_per_day: "每日脂肪目标",
  };
  if (labels[raw]) return labels[raw];
  return raw.replaceAll("_", " ");
}

function groupBy(items, getKey) {
  return (items || []).reduce((acc, item) => {
    const key = getKey(item);
    if (!acc[key]) acc[key] = [];
    acc[key].push(item);
    return acc;
  }, {});
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

async function apiGet(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function apiUpload(url, body) {
  const response = await fetch(url, {
    method: "POST",
    body,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Upload failed: ${response.status}`);
  }
  return payload;
}

async function apiDelete(url) {
  const response = await fetch(url, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Delete failed: ${response.status}`);
  }
  return payload;
}

function showToast(message, type = "success") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  toastRootEl.appendChild(node);
  setTimeout(() => node.remove(), 3200);
}
