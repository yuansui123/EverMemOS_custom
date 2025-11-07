"""
数据加载器

提供不同数据集的加载功能。
支持自动转换非 Locomo 格式的数据集。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from evaluation.src.core.data_models import Dataset, Conversation, Message, QAPair
from evaluation.src.converters.registry import get_converter


def load_dataset(dataset_name: str, data_path: str) -> Dataset:
    """
    智能加载数据集（支持自动转换）
    
    Args:
        dataset_name: 数据集名称（如 "locomo", "longmemeval", "personamem"）
        data_path: 数据文件路径或目录路径
        
    Returns:
        Dataset: 标准格式数据集
    """
    data_path_obj = Path(data_path)
    
    # 检查是否需要转换
    converter = get_converter(dataset_name)
    
    if converter:
        # 需要转换的数据集
        if data_path_obj.is_file():
            # 如果给的是文件路径，取其父目录
            data_dir = data_path_obj.parent
        else:
            data_dir = data_path_obj
        
        # 检查是否需要转换
        if converter.needs_conversion(data_dir):
            print(f"📝 Converted file not found, converting {dataset_name}...")
            
            # 构建输入文件路径
            input_files = converter.get_input_files()
            input_paths = {
                key: str(data_dir / filename)
                for key, filename in input_files.items()
            }
            
            # 执行转换
            output_path = str(converter.get_converted_path(data_dir))
            converter.convert(input_paths, output_path)
        
        # 使用 converted 文件
        locomo_file = converter.get_converted_path(data_dir)
    else:
        # 原生 Locomo 格式，直接使用
        if data_path_obj.is_file():
            locomo_file = data_path_obj
        else:
            # 如果是目录，尝试找到 .json 文件
            json_files = list(data_path_obj.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {data_path_obj}")
            locomo_file = json_files[0]
    
    return load_locomo_dataset(str(locomo_file), dataset_name=dataset_name)


def load_locomo_dataset(data_path: str, dataset_name: str = "locomo") -> Dataset:
    """
    加载 LoCoMo 格式的数据集
    
    Args:
        data_path: Locomo 格式数据文件路径
        dataset_name: 数据集名称（默认为 "locomo"，转换后的数据集应传入原始名称）
        
    Returns:
        Dataset: 标准格式数据集
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    conversations = []
    qa_pairs = []
    
    for idx, item in enumerate(raw_data):
        # 🔥 添加数据集前缀，避免不同数据集间的 conversation_id 冲突
        # 例如：locomo_0, longmemeval_0, personamem_0
        conv_id = f"{dataset_name}_{idx}"
        conversation_data = item.get("conversation", {})
        qa_data = item.get("qa", [])
        
        # 转换对话
        conversation = _convert_locomo_conversation(conversation_data, conv_id)
        conversations.append(conversation)
        
        # 转换 QA 对
        for qa_idx, qa_item in enumerate(qa_data):
            qa_pair = _convert_locomo_qa_pair(qa_item, conv_id, qa_idx)
            qa_pairs.append(qa_pair)
    
    return Dataset(
        dataset_name=dataset_name,
        conversations=conversations,
        qa_pairs=qa_pairs,
        metadata={"total_conversations": len(conversations)}
    )


def _convert_locomo_conversation(conversation_data: dict, conv_id: str) -> Conversation:
    """转换 LoCoMo 对话"""
    messages = []
    
    # 获取所有 session keys
    session_keys = sorted([
        key for key in conversation_data.keys()
        if key.startswith("session_") and not key.endswith("_date_time")
    ])
    
    # 🔥 为没有时间戳的数据生成伪造的起始时间（用于 online API）
    # 使用一个固定的基准时间：2024-01-01 00:00:00
    fake_base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 🔥 第一步：解析所有 session 的时间戳
    session_times = []
    for session_idx, session_key in enumerate(session_keys):
        session_time_key = f"{session_key}_date_time"
        if session_time_key in conversation_data:
            session_time = _parse_locomo_timestamp(conversation_data[session_time_key])
            
            # 如果解析失败或为 "Unknown"，生成伪造时间戳
            is_fake = (session_time is None)
            if is_fake:
                session_time = fake_base_time + timedelta(hours=session_idx)
            
            session_times.append({
                "time": session_time,
                "is_fake": is_fake
            })
        else:
            # 没有 date_time 字段，生成伪造时间戳
            session_times.append({
                "time": fake_base_time + timedelta(hours=session_idx),
                "is_fake": True
            })
    
    # 🔥 第二步：为每个 session 分配消息时间戳
    for session_idx, session_key in enumerate(session_keys):
        session_messages = conversation_data[session_key]
        
        if not session_messages:
            continue
        
        # 获取当前 session 的起始时间
        current_session_time = session_times[session_idx]["time"]
        is_fake_timestamp = session_times[session_idx]["is_fake"]
        
        # 🔥 计算消息时间间隔
        # 策略：优先使用30秒间隔，只有在会超出下一个session时才缩小间隔
        num_messages = len(session_messages)
        default_interval = 30  # 默认30秒间隔
        
        if num_messages > 1:
            # 计算使用默认间隔需要的总时长
            required_duration = (num_messages - 1) * default_interval
            
            # 获取可用的时间跨度
            if session_idx + 1 < len(session_times):
                # 有下一个 session：计算到下一个 session 的时间
                next_session_time = session_times[session_idx + 1]["time"]
                available_duration = (next_session_time - current_session_time).total_seconds()
                
                # 如果时间跨度为负或太小（说明数据有问题），使用默认间隔
                if available_duration <= 0:
                    time_interval = default_interval
                # 留出10%缓冲，避免最后一条消息太接近下一个 session
                elif required_duration > available_duration * 0.9:
                    # 需要缩小间隔才能放下所有消息
                    time_interval = (available_duration * 0.9) / (num_messages - 1)
                else:
                    # 可以使用默认间隔
                    time_interval = default_interval
            else:
                # 最后一个 session：直接使用默认间隔
                time_interval = default_interval
        else:
            # 只有一条消息，放在 session 开始时
            time_interval = 0
        
        # 转换每条消息
        for msg_idx, msg in enumerate(session_messages):
            msg_timestamp = current_session_time + timedelta(seconds=msg_idx * time_interval)
            
            message = Message(
                speaker_id=f"{msg['speaker'].lower().replace(' ', '_')}_{conv_id}",
                speaker_name=msg['speaker'],
                content=msg['text'],
                timestamp=msg_timestamp,
                metadata={
                    "session": session_key,
                    "dia_id": msg.get("dia_id"),
                    "img_url": msg.get("img_url"),
                    "blip_caption": msg.get("blip_caption"),
                    "is_fake_timestamp": is_fake_timestamp,  # 标记是否为伪造时间戳
                }
            )
            messages.append(message)
    
    return Conversation(
        conversation_id=conv_id,
        messages=messages,
        metadata={
            "speaker_a": conversation_data.get("speaker_a"),
            "speaker_b": conversation_data.get("speaker_b"),
        }
    )


def _convert_locomo_qa_pair(qa_item: dict, conv_id: str, qa_idx: int) -> QAPair:
    """转换 LoCoMo QA 对"""
    # 提取额外的字段到 metadata
    metadata = {"conversation_id": conv_id}
    
    # 如果有 all_options（PersonaMem 选择题），保存到 metadata
    if "all_options" in qa_item:
        metadata["all_options"] = qa_item["all_options"]
    
    # 优先使用数据中的 question_id（如果存在），否则生成一个唯一的 ID
    question_id = qa_item.get("question_id")
    if not question_id:
        # 使用 conv_id + qa_idx 生成唯一 ID，确保不会冲突
        question_id = f"{conv_id}_qa{qa_idx}"
    
    # 统一将 category 转换为字符串（兼容 int 和 str）
    category = qa_item.get("category")
    if category is not None:
        category = str(category)
    
    return QAPair(
        question_id=question_id,
        question=qa_item.get("question", ""),
        answer=qa_item.get("answer", ""),
        category=category,
        evidence=qa_item.get("evidence", []),
        metadata=metadata
    )


def _parse_locomo_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    解析 LoCoMo 的时间格式
    
    输入格式: "6:07 pm on 13 January, 2023"
    特殊值: "Unknown" 或无法解析时返回 None
    输出: datetime 对象或 None
    """
    # 清理字符串
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    
    # 处理特殊情况：Unknown 或空字符串
    if timestamp_str.lower() == "unknown" or not timestamp_str:
        # 没有时间信息，返回 None
        return None
    
    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        # 如果解析失败，返回 None 并输出警告
        print(f"⚠️  Warning: Failed to parse timestamp '{timestamp_str}', no timestamp will be set")
        return None

