"""
核心数据模型

定义评测框架中使用的标准数据格式，确保不同系统和数据集之间的互操作性。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Message:
    """标准消息格式"""
    speaker_id: str
    speaker_name: str
    content: str
    timestamp: Optional[datetime] = None  # 时间戳可选，某些数据集（如 PersonaMem）没有时间信息
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """标准对话格式"""
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    """
    标准 QA 对格式
    
    注意：category 字段统一为字符串类型：
    - LoCoMo: "1", "2", "3", "5" (原始为整数，加载时转为字符串)
    - LongMemEval: "single-session-user", "multi-session-user"
    - PersonaMem: "recall_user_shared_facts", "suggest_new_ideas" 等
    """
    question_id: str
    question: str
    answer: str
    category: Optional[str] = None  # 统一为字符串类型
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """标准数据集格式"""
    dataset_name: str
    conversations: List[Conversation]
    qa_pairs: List[QAPair]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """标准检索结果格式"""
    query: str
    conversation_id: str
    results: List[Dict[str, Any]]  # [{"content": str, "score": float, "metadata": dict}]
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """标准答案结果格式"""
    question_id: str
    question: str
    answer: str
    golden_answer: str
    category: Optional[int] = None
    conversation_id: str = ""
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """标准评估结果格式"""
    total_questions: int
    correct: int
    accuracy: float
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

