# 本地验证记录

日期：2026-10-08。

- 环境：复用原项目本地虚拟环境，Python 3.14.3，openai 3.14.0，sentence-transformers 6.0.1，pytest 9.1.1。
- 基础测试：15 passed。
- 离线轨迹：退出码 0；计算工具返回 56088，循环步数耗尽路径返回 None。
- 首次 RAG 测试：14 passed、1 failed。未知查询测试隐式调用模型，缺少凭据失败。
- 整理后未知查询测试显式模拟模型返回空解析对象，只验证未知字段不会错误过滤所有候选。
- 最终全量测试：30 passed（35.32 秒）。嵌入使用本地缓存的 BAAI/bge-small-zh-v1.5，开启 HF_HUB_OFFLINE；不使用真实 API Key。

验证不代表真实 LLM 端到端验收、RAG 质量评估、新环境依赖安装或跨平台兼容。原先空的 test_rag_e2e.py 未纳入候选。

