"""
Exact Match 评估器

用于直接对比答案是否完全匹配，适用于选择题等场景。
"""
import re
from typing import List

from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.core.data_models import AnswerResult, EvaluationResult


@register_evaluator("exact_match")
class ExactMatch(BaseEvaluator):
    """精确匹配评估器"""
    
    def __init__(self, config: dict):
        """
        初始化评估器
        
        Args:
            config: 评估配置（可选参数：case_sensitive, normalize_whitespace）
        """
        super().__init__(config)
        
        # 配置选项
        self.case_sensitive = config.get("case_sensitive", False)  # 默认不区分大小写
        self.normalize_whitespace = config.get("normalize_whitespace", True)  # 默认规范化空白字符
        self.extract_choice = config.get("extract_choice", True)  # 默认提取选项（如 (a), (b), (c)）
    
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        使用精确匹配评估答案
        
        Args:
            answer_results: 答案结果列表
            
        Returns:
            EvaluationResult: 评估结果
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: Exact Match")
        print(f"  - Case sensitive: {self.case_sensitive}")
        print(f"  - Normalize whitespace: {self.normalize_whitespace}")
        print(f"  - Extract choice: {self.extract_choice}")
        print(f"{'='*60}")
        
        detailed_results = []
        total_correct = 0
        
        # 评估每个答案
        for answer_result in answer_results:
            is_correct = self._check_match(
                answer_result.golden_answer,
                answer_result.answer
            )
            
            if is_correct:
                total_correct += 1
            
            detailed_results.append({
                "question_id": answer_result.question_id,
                "question": answer_result.question,
                "golden_answer": answer_result.golden_answer,
                "generated_answer": answer_result.answer,
                "is_correct": is_correct,
                "category": answer_result.category,
            })
        
        accuracy = total_correct / len(answer_results) if answer_results else 0.0
        
        print(f"\n✅ 评估完成:")
        print(f"   - 总问题数: {len(answer_results)}")
        print(f"   - 正确: {total_correct}")
        print(f"   - 准确率: {accuracy:.2%}")
        
        return EvaluationResult(
            total_questions=len(answer_results),
            correct=total_correct,
            accuracy=accuracy,
            detailed_results=detailed_results,
            metadata={
                "evaluator": "exact_match",
                "case_sensitive": self.case_sensitive,
                "normalize_whitespace": self.normalize_whitespace,
                "extract_choice": self.extract_choice,
            }
        )
    
    def _check_match(self, golden: str, generated: str) -> bool:
        """
        检查两个答案是否匹配
        
        Args:
            golden: 标准答案
            generated: 生成的答案
            
        Returns:
            是否匹配
        """
        # 预处理
        golden_processed = self._preprocess(golden)
        generated_processed = self._preprocess(generated)
        
        # 如果启用选项提取，尝试从生成的答案中提取选项
        if self.extract_choice:
            extracted_choice = self._extract_choice(generated_processed)
            if extracted_choice:
                generated_processed = extracted_choice
        
        # 比较
        if self.case_sensitive:
            return golden_processed == generated_processed
        else:
            return golden_processed.lower() == generated_processed.lower()
    
    def _preprocess(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # 规范化空白字符
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_choice(self, text: str) -> str:
        """
        从文本中提取选项（如 (a), (b), (c), (d)）
        
        支持多种格式：
        - (a)
        - a)
        - a.
        - A
        - option a
        
        Args:
            text: 文本
            
        Returns:
            提取的选项（如 "(a)"），如果未找到则返回空字符串
        """
        # 尝试匹配 (a), (b), (c), (d) 等格式
        match = re.search(r'\(([a-zA-Z])\)', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # 尝试匹配 a), b), c), d) 等格式
        match = re.search(r'\b([a-zA-Z])\)', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # 尝试匹配 a., b., c., d. 等格式
        match = re.search(r'\b([a-zA-Z])\.', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # 尝试匹配单独的字母（在句首或被空白包围）
        match = re.search(r'(?:^|\s)([a-zA-Z])(?:\s|$)', text)
        if match:
            letter = match.group(1).lower()
            # 只接受 a-z 的前几个字母（通常选项不超过 f）
            if letter in 'abcdefgh':
                return f"({letter})"
        
        # 如果都没匹配到，返回空字符串
        return ""


