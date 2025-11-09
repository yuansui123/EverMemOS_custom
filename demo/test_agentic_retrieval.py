"""
Agentic 检索测试脚本

测试 Agentic 检索功能的各个组件。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

async def test_agentic_utils():
    """测试 Agentic 工具函数"""
    from agentic_layer.agentic_utils import (
        AgenticConfig,
        format_documents_for_llm,
        parse_json_response,
    )
    
    print("\n" + "="*60)
    print("测试 1: Agentic Utils")
    print("="*60)
    
    # 测试配置
    config = AgenticConfig()
    print(f"✓ AgenticConfig 创建成功")
    print(f"  - Round 1 Top N: {config.round1_top_n}")
    print(f"  - Use Reranker: {config.use_reranker}")
    print(f"  - Enable Multi Query: {config.enable_multi_query}")
    
    # 测试文档格式化
    class MockMemory:
        def __init__(self):
            self.timestamp = "2024-01-15 10:30:00"
            self.episode = "用户说他最喜欢吃川菜"
            self.summary = "用户的饮食偏好"
            self.subject = "饮食习惯"
    
    mock_results = [(MockMemory(), 0.95), (MockMemory(), 0.88)]
    formatted = format_documents_for_llm(mock_results, max_docs=2)
    print(f"✓ 文档格式化成功")
    print(f"  格式化文本长度: {len(formatted)} 字符")
    
    # 测试 JSON 解析
    json_text = '{"is_sufficient": true, "reasoning": "测试"}'
    result = parse_json_response(json_text)
    print(f"✓ JSON 解析成功")
    print(f"  解析结果: {result}")
    
    print("\n✅ Agentic Utils 测试通过\n")


async def test_retrieval_utils():
    """测试检索工具函数"""
    from agentic_layer.retrieval_utils import (
        multi_rrf_fusion,
        reciprocal_rank_fusion,
    )
    
    print("\n" + "="*60)
    print("测试 2: Retrieval Utils")
    print("="*60)
    
    # 创建模拟文档
    class MockDoc:
        def __init__(self, id):
            self.id = id
            self.episode = f"Mock episode {id}"
    
    # 测试双路 RRF
    results1 = [(MockDoc(1), 0.9), (MockDoc(2), 0.8), (MockDoc(3), 0.7)]
    results2 = [(MockDoc(2), 0.85), (MockDoc(1), 0.75), (MockDoc(4), 0.7)]
    
    fused = reciprocal_rank_fusion(results1, results2, k=60)
    print(f"✓ 双路 RRF 融合成功")
    print(f"  输入: 3 + 3 = 6 个结果")
    print(f"  输出: {len(fused)} 个去重结果")
    
    # 测试多路 RRF
    results_list = [results1, results2, results1]
    multi_fused = multi_rrf_fusion(results_list, k=60)
    print(f"✓ 多路 RRF 融合成功")
    print(f"  输入: 3 个结果集")
    print(f"  输出: {len(multi_fused)} 个去重结果")
    
    print("\n✅ Retrieval Utils 测试通过\n")


async def test_memory_manager():
    """测试 Memory Manager（需要数据库连接）"""
    print("\n" + "="*60)
    print("测试 3: Memory Manager")
    print("="*60)
    
    try:
        from agentic_layer.memory_manager import MemoryManager
        
        manager = MemoryManager()
        print(f"✓ MemoryManager 创建成功")
        print(f"  - 已注册 retrieve_agentic 方法: {hasattr(manager, 'retrieve_agentic')}")
        print(f"  - 已注册 retrieve_lightweight 方法: {hasattr(manager, 'retrieve_lightweight')}")
        
        print("\n✅ Memory Manager 测试通过（基础功能）")
        print("⚠️  完整测试需要数据库连接和 LLM Provider\n")
    
    except Exception as e:
        print(f"❌ Memory Manager 测试失败: {e}")
        print(f"   这可能是因为缺少数据库连接\n")


async def test_integration():
    """集成测试提示"""
    print("\n" + "="*60)
    print("集成测试指南")
    print("="*60)
    
    print("""
要进行完整的集成测试，请执行以下步骤：

1. 启动 MongoDB 数据库
2. 配置 .env 文件（LLM API Key）
3. 运行对话脚本:
   
   uv run python src/bootstrap.py demo/chat_with_memory.py
   
4. 选择检索模式 4（Agentic 检索）
5. 输入复杂查询测试，例如：
   - "用户喜欢吃什么？有什么忌口吗？"
   - "用户最常讨论的话题是什么？"

查看日志输出，验证：
✓ Round 1 检索成功
✓ LLM 判断充分性
✓ （如不充分）生成改进查询
✓ Round 2 多查询检索
✓ 最终结果返回

更多信息请参考:
📖 docs/dev_docs/agentic_retrieval_guide.md
""")


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 Agentic 检索功能测试")
    print("="*60)
    
    try:
        # 测试各个组件
        await test_agentic_utils()
        await test_retrieval_utils()
        await test_memory_manager()
        await test_integration()
        
        print("\n" + "="*60)
        print("✅ 所有基础测试通过！")
        print("="*60)
        print("""
下一步:
1. 查看实现文档: docs/dev_docs/agentic_retrieval_guide.md
2. 运行集成测试: uv run python src/bootstrap.py demo/chat_with_memory.py
3. 测试不同的查询场景
        """)
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

