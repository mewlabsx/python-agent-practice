# Python Agent Practice

一个用于学习 Agent 核心执行机制的 Python POC，包含多轮对话、工具调用、进程内记忆和结构化 RAG。适合作为代码阅读与面试演示项目；尚不是生产级 Agent 平台。

## 核心实现

- `agent.py`：最多 5 步的工具执行循环，保留最近 2 轮模型上下文。
- `runtime.py` / `shape.py`：工具结果封装、异常转换、JSON 与 SDK 对象边界。
- `memory.py`：由模型提取记忆，按类型和 key 存入进程内字典。
- `ingestion.py` / `rag.py`：Markdown 分块、指标和章节角色元数据、规则解析与模型回退、查询改写、过滤和余弦相似度排序。
- `tools.py`：计算器、虚构项目查询和知识检索工具。
- `provider.py`：OpenAI-compatible API，有限次数重试、指数退避与抖动。

执行路径：`main → agent → provider / runtime → tools → rag → ingestion / embedding`。
`knowledge/demo_knowledge.md` 是简短教学示例，不代表真实业务要求或验收结论。项目查询返回的是静态模拟数据。

## 安装与离线演示

本次本地验证环境为 Python 3.14.3；其他 Python 版本尚未验证。建议新建虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python agent_trace.py
```

轨迹脚本使用预设模型响应，运行真实工具和循环逻辑，不需要 API Key 或嵌入模型，不证明真实 LLM 的效果。

## 真实模型与 RAG

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
$env:AGENT_MODEL = "your-provider-model-id"
# 第一次运行允许下载嵌入模型，之后可恢复为 1，仅使用本地缓存。
$env:EMBEDDING_LOCAL_ONLY = "0"
python main.py
```

模型 ID 应按你的服务账户可用模型设置，代码中的默认 ID 不保证可用。真实请求会发送对话、检索片段或记忆抽取内容，并可能计费。示例配置见 `.env.example`；程序不会自动读取 `.env`。默认嵌入模型使用 CPU，下载后的模型保存在模型库的缓存目录中，不应提交到仓库。模型权重不包含在本项目中。

## 测试

```powershell
python -m pytest test/test_shape.py test/test_runtime.py test/test_memory.py
# RAG 测试需要嵌入模型缓存；未知查询测试使用显式模拟模型响应。
python -m pytest test/test_rag.py
```

测试结果和验证边界见 `VALIDATION.md`。

## 已知限制

- 步数耗尽时循环隐式返回 None；这也是轨迹脚本展示的异常路径。
- 记忆不持久化，检索主要围绕固定偏好 key；历史消息列表仍会增长。
- 检索索引保存在内存中，启动时重新构建；知识库样本很小。
- 查询理解可能调用模型；没有真实模型端到端验收或检索质量指标。
- 工具没有超时、权限隔离或沙箱；未绑定知识库时直接调用默认检索工具会失败。
- 当前无并发、多用户、部署、成本预算或生产级可观测性保证。

## License

采用 MIT 许可证，见 `LICENSE`。本仓库不包含第三方模型权重；其使用条款应单独确认。

