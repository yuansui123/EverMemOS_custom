"""
CLI 入口

评测框架的命令行接口。

Usage:
    python -m evaluation.cli --dataset locomo --system evermemos
    python -m evaluation.cli --dataset locomo --system evermemos --smoke 10
    python -m evaluation.cli --dataset locomo --system evermemos --stages search answer evaluate
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path

# ===== 环境初始化 =====
# 必须在导入任何 EverMemOS 组件之前完成
# 参考 src/bootstrap.py 的初始化逻辑

# 1. 添加项目路径
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 2. 加载环境变量
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

# 3. 初始化依赖注入容器（如果需要的话）
# 注：当前 adapter 手动创建对象，暂不需要 DI 容器
# 但保留此注释，未来如果需要可以启用：
# from application_startup import setup_all
# setup_all()

# ===== 现在可以安全地导入 EverMemOS 组件 =====
from evaluation.src.core.loaders import load_dataset
from evaluation.src.core.pipeline import Pipeline
from evaluation.src.adapters.registry import create_adapter
from evaluation.src.evaluators.registry import create_evaluator
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.logger import get_console

from memory_layer.llm.llm_provider import LLMProvider


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    深度合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge_config(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    return result


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Memory System Evaluation Framework")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., locomo)"
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="System name (e.g., evermemos)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (add, search, answer, evaluate). Default: all"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode (process small dataset for quick validation)"
    )
    parser.add_argument(
        "--smoke-messages",
        type=int,
        default=10,
        help="Smoke test: number of messages to process (use 0 for all). Default: 10"
    )
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=3,
        help="Smoke test: number of questions to test (use 0 for all). Default: 3"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name/version for distinguishing multiple runs (e.g., 'v1', 'baseline', '20241104')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/{dataset}-{system}[-{run_name}]"
    )
    
    args = parser.parse_args()
    
    console = get_console()
    
    # ===== 加载配置 =====
    console.print("\n[bold cyan]Loading configurations...[/bold cyan]")
    
    evaluation_root = Path(__file__).parent
    
    # 加载数据集配置
    dataset_config_path = evaluation_root / "config" / "datasets" / f"{args.dataset}.yaml"
    if not dataset_config_path.exists():
        console.print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return
    
    dataset_config = load_yaml(str(dataset_config_path))
    console.print(f"  ✅ Loaded dataset config: {args.dataset}")
    
    # 加载系统配置
    system_config_path = evaluation_root / "config" / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        console.print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return
    
    system_config = load_yaml(str(system_config_path))
    console.print(f"  ✅ Loaded system config: {args.system}")
    
    # 应用数据集特定的配置覆盖
    if "dataset_overrides" in system_config and args.dataset in system_config["dataset_overrides"]:
        overrides = system_config["dataset_overrides"][args.dataset]
        # 深度合并覆盖配置（支持嵌套字段覆盖）
        system_config = deep_merge_config(system_config, overrides)
        console.print(f"  🔧 Applied dataset overrides for {args.dataset}: {list(overrides.keys())}")
    
    # ===== 加载数据集 =====
    console.print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")
    
    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        # 优先从 evaluation/data/ 加载，如果不存在则从项目根目录加载
        eval_data_path = evaluation_root / "data" / data_path
        root_data_path = evaluation_root.parent / data_path
        
        if eval_data_path.exists():
            data_path = eval_data_path
            console.print(f"  📂 Using evaluation/data/{data_path}")
        elif root_data_path.exists():
            data_path = root_data_path
            console.print(f"  📂 Using project root data/{data_path}")
        else:
            console.print(f"[red]❌ Data not found in evaluation/data/ or project root data/[/red]")
            return
    
    # 智能加载（自动转换）
    dataset = load_dataset(args.dataset, str(data_path))
    
    console.print(f"  ✅ Loaded {len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} QA pairs")
    
    # ===== 确定输出目录 =====
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 根据是否有 run_name 生成输出目录名
        if args.run_name:
            output_dir = evaluation_root / "results" / f"{args.dataset}-{args.system}-{args.run_name}"
        else:
            output_dir = evaluation_root / "results" / f"{args.dataset}-{args.system}"
    
    # ===== 创建组件 =====
    console.print(f"\n[bold cyan]Initializing components...[/bold cyan]")
    
    # 创建适配器（传递 output_dir 用于持久化）
    adapter = create_adapter(
        system_config["adapter"],
        system_config,
        output_dir=output_dir
    )
    console.print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")
    
    # 创建评估器
    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"],
        dataset_config["evaluation"]
    )
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")
    
    # 创建 LLM Provider（用于答案生成）
    llm_config = system_config.get("llm", {})
    llm_provider = LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 32768),
    )
    console.print(f"  ✅ Created LLM provider: {llm_config.get('model')}")
    
    # ===== 创建 Pipeline =====
    # 从数据集配置中读取需要过滤的问题类别
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])
    
    pipeline = Pipeline(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        filter_categories=filter_categories
    )
    
    console.print(f"  ✅ Created pipeline, output: {output_dir}")
    if filter_categories:
        console.print(f"  📋 Filter categories: {filter_categories}")
    
    # ===== 运行 Pipeline =====
    try:
        results = await pipeline.run(
            dataset=dataset,
            stages=args.stages,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
        )
        
        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")
    
    finally:
        # ===== 清理资源 =====
        # 只有使用了 rerank 的系统才需要清理
        systems_need_rerank = ["evermemos"]
        if args.system in systems_need_rerank:
            try:
                from agentic_layer import rerank_service
                reranker = rerank_service.get_rerank_service()
                if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
                    await reranker.close()
                    console.print("[dim]🧹 Cleaned up rerank service resources[/dim]")
            except Exception as e:
                # 如果清理失败也不影响主流程
                console.print(f"[dim]⚠️  Failed to cleanup resources: {e}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

