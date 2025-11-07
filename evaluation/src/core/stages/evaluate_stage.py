"""
Evaluate 阶段

负责评估答案质量。
"""
from typing import List, Optional
from logging import Logger

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_evaluate_stage(
    evaluator: BaseEvaluator,
    answer_results: List[AnswerResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> EvaluationResult:
    """
    执行 Evaluate 阶段
    
    Args:
        evaluator: 评估器
        answer_results: 答案结果列表
        checkpoint_manager: 断点续传管理器
        logger: 日志器
        
    Returns:
        评估结果
    """
    logger.info("Starting Stage 4: Evaluate")
    
    eval_result = await evaluator.evaluate(answer_results)
    
    logger.info("✅ Stage 4 completed")
    
    return eval_result


