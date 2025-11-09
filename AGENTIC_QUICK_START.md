# 🎯 Agentic 检索 - 快速开始

## 一分钟了解

Agentic 检索是一种 **LLM 引导的智能多轮检索**方法，自动判断检索结果是否充分，并在必要时生成改进查询进行第二轮检索。

```python
# 一行代码启用 Agentic 检索
result = await memory_manager.retrieve_agentic(
    query="用户喜欢吃什么？",
    group_id="美食群",
    llm_provider=llm,
)
```

---

## 快速使用

### 方法 1: 对话界面（推荐）

```bash
# 启动对话应用
uv run python src/bootstrap.py demo/chat_with_memory.py

# 选择检索模式时，输入 4（Agentic 检索）
```

### 方法 2: Python 代码

```python
from agentic_layer.memory_manager import MemoryManager
from memory_layer.llm.llm_provider import LLMProvider

# 1. 初始化 LLM
llm = LLMProvider(
    provider_type="openai",
    model="gpt-4",
    api_key="sk-...",
)

# 2. 初始化 Memory Manager
manager = MemoryManager()

# 3. 执行检索
result = await manager.retrieve_agentic(
    query="用户最喜欢的菜系是什么？",
    group_id="美食爱好者群",
    llm_provider=llm,
)

# 4. 查看结果
print(f"检索到 {result['count']} 条记忆")
print(f"LLM 判断: {'充分' if result['metadata']['is_sufficient'] else '不充分'}")

if result['metadata']['is_multi_round']:
    print(f"改进查询: {result['metadata']['refined_queries']}")
```

---

## 核心优势

| 对比项 | Lightweight 检索 | **Agentic 检索** |
|--------|----------------|----------------|
| 延迟 | 0.5-2s | 5-10s |
| 召回率 | 中 | **高** ⭐ |
| 精度 | 中 | **高** ⭐ |
| 复杂查询 | 一般 | **优秀** ⭐ |
| LLM 调用 | 无 | 1-2 次 |
| 成本 | 低 | 中 (~$0.001-0.003) |

---

## 工作流程

```
用户输入查询
    ↓
Round 1: 混合检索 (Embedding + BM25 + RRF)
    ↓
Rerank → Top 5
    ↓
LLM 判断：这些记忆足够回答查询吗？
    ↓
  ┌─────────────────┬─────────────────┐
  │    充分 ✅      │   不充分 ❌    │
  │  返回 Top 20   │  进入 Round 2  │
  └─────────────────┴─────────────────┘
                          ↓
                  LLM 生成 2-3 个改进查询
                          ↓
                  并行检索所有查询
                          ↓
                  多查询 RRF 融合
                          ↓
                  去重 + 合并
                          ↓
                  Rerank → Top 20 ✅
```

---

## 配置选项

### 基础配置（使用默认值）

```python
result = await manager.retrieve_agentic(
    query="...",
    group_id="...",
    llm_provider=llm,
)
```

### 高级配置（自定义参数）

```python
from agentic_layer.agentic_utils import AgenticConfig

config = AgenticConfig(
    # 是否使用 Reranker（提升精度）
    use_reranker=True,
    
    # 是否启用多查询（提升召回）
    enable_multi_query=True,
    
    # 生成查询数量（2-3 个）
    num_queries=3,
    
    # Round 1 返回数量
    round1_top_n=20,
    
    # 最终返回数量
    final_top_n=20,
)

result = await manager.retrieve_agentic(
    query="...",
    group_id="...",
    llm_provider=llm,
    agentic_config=config,
)
```

---

## 适用场景

### ✅ 适合使用

1. **复杂多维查询**
   ```python
   "用户最喜欢的川菜是什么？有什么忌口吗？"
   "团队讨论过哪些技术方案？优缺点是什么？"
   ```

2. **信息分散**
   - 相关记忆散落在不同时间点
   - 需要多个角度的信息

3. **高质量要求**
   - 对召回率和精度要求高
   - 可以接受 5-10 秒延迟

### ❌ 不适合使用

1. **简单查询**
   ```python
   "用户的名字是什么？"
   "今天星期几？"
   ```

2. **对延迟敏感**
   - 要求 < 1 秒响应
   - 实时聊天场景

3. **成本敏感**
   - 无法承担 LLM API 费用
   - 高频调用场景

---

## 返回结果示例

```python
{
    "memories": [
        {
            "event_id": "mem_123",
            "timestamp": "2024-01-15T10:30:00",
            "episode": "用户说他最喜欢吃川菜，尤其是麻婆豆腐",
            "score": 0.95
        },
        # ... 更多记忆
    ],
    "count": 20,
    "metadata": {
        # 基本信息
        "retrieval_mode": "agentic",
        "is_multi_round": True,
        "total_latency_ms": 3500,
        
        # LLM 判断
        "is_sufficient": False,
        "reasoning": "缺少用户的口味偏好信息",
        "missing_info": ["口味偏好", "忌口信息"],
        
        # 改进查询（仅在多轮时存在）
        "refined_queries": [
            "用户最喜欢的菜系是什么？",
            "用户喜欢什么口味？",
            "用户有什么饮食禁忌？"
        ],
        
        # 详细统计
        "round1_count": 20,
        "round2_count": 40,
        "final_count": 20
    }
}
```

---

## 故障排查

### 问题：LLM API 调用失败

**症状**：返回 `agentic_fallback` 模式

**解决**：
1. 检查 `.env` 文件中的 API Key
2. 确认网络连接
3. 查看日志中的详细错误

### 问题：延迟过高（> 10 秒）

**原因**：
- LLM 响应慢
- 候选记忆过多
- Reranker 超时

**解决**：
```python
config = AgenticConfig(
    use_reranker=False,  # 禁用 Reranker（降低延迟）
)

result = await manager.retrieve_agentic(
    ...,
    time_range_days=30,  # 减少时间范围（减少候选数）
    agentic_config=config,
)
```

### 问题：检索质量不佳

**解决**：
1. 使用更强的 LLM 模型（如 GPT-4）
2. 增加 `round1_rerank_top_n`（给 LLM 更多样本）
3. 调整 Prompt 模板（在 `agentic_utils.py`）

---

## 更多文档

- 📖 **完整指南**: [docs/dev_docs/agentic_retrieval_guide.md](docs/dev_docs/agentic_retrieval_guide.md)
- 📊 **实现总结**: [AGENTIC_IMPLEMENTATION_SUMMARY.md](AGENTIC_IMPLEMENTATION_SUMMARY.md)
- 🎯 **API 文档**: [docs/api_docs/agentic_v3_api.md](docs/api_docs/agentic_v3_api.md)

---

## 常见问题 FAQ

**Q: Agentic 检索比 Lightweight 检索慢多少？**  
A: 单轮约 2-5 秒，多轮约 5-10 秒。Lightweight 约 0.5-2 秒。

**Q: 每次调用的成本是多少？**  
A: 基于 GPT-4，单轮约 $0.001，多轮约 $0.003。

**Q: 可以使用其他 LLM 吗？**  
A: 可以！支持任何兼容 Memory Layer LLMProvider 的模型。

**Q: 如何监控 LLM 判断是否准确？**  
A: 查看返回的 `metadata.reasoning` 和日志输出。

**Q: 能否在生产环境使用？**  
A: 可以，但建议先进行充分测试，并监控成本和延迟。

---

**开始使用吧！** 🚀

有问题？查看 [完整指南](docs/dev_docs/agentic_retrieval_guide.md) 或提交 Issue。

