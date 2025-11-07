"""
Checkpoint 管理模块

支持断点续传，避免评测中途失败需要重新运行。

两层 Checkpoint 机制：
1. 跨阶段 Checkpoint：记录哪些阶段（add/search/answer/evaluate）已完成
2. 阶段内 Checkpoint：记录单个阶段内的细粒度进度（每个会话/每N个问题）
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime


class CheckpointManager:
    """Checkpoint 管理器
    
    - Stage 3 (search): 每处理完一个会话就保存检查点
    - Stage 4 (answer): 每 SAVE_INTERVAL 个问题保存一次
    """
    
    def __init__(self, output_dir: Path, run_name: str = "default"):
        """
        初始化 Checkpoint 管理器
        
        Args:
            output_dir: 输出目录
            run_name: 运行名称
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        
        # 跨阶段检查点（记录哪些阶段已完成）
        self.checkpoint_file = self.output_dir / f"checkpoint_{run_name}.json"
        
        # 细粒度检查点（每个阶段一个，记录阶段内进度）
        self.search_checkpoint = self.output_dir / f"search_results_checkpoint.json"
        self.answer_checkpoint = self.output_dir / f"answer_results_checkpoint.json"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        加载已有的 checkpoint
        
        Returns:
            checkpoint 数据，如果不存在则返回 None
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            print(f"\n🔄 发现 checkpoint 文件: {self.checkpoint_file.name}")
            print(f"   上次运行时间: {checkpoint.get('last_updated', 'Unknown')}")
            print(f"   已完成阶段: {', '.join(checkpoint.get('completed_stages', []))}")
            
            if 'search_results' in checkpoint:
                completed_convs = len(checkpoint['search_results'])
                print(f"   已处理对话数: {completed_convs}")
            
            return checkpoint
            
        except Exception as e:
            print(f"⚠️ 加载 checkpoint 失败: {e}")
            print(f"   将从头开始运行")
            return None
    
    def save_checkpoint(
        self, 
        completed_stages: Set[str],
        search_results: Optional[Dict] = None,
        answer_results: Optional[Dict] = None,
        eval_results: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        保存 checkpoint
        
        Args:
            completed_stages: 已完成的阶段集合
            search_results: 搜索结果（可选）
            answer_results: 答案结果（可选）
            eval_results: 评测结果（可选）
            metadata: 其他元数据（可选）
        """
        checkpoint = {
            "run_name": self.run_name,
            "last_updated": datetime.now().isoformat(),
            "completed_stages": list(completed_stages),
        }
        
        if search_results is not None:
            checkpoint["search_results"] = search_results
        
        if answer_results is not None:
            checkpoint["answer_results"] = answer_results
        
        if eval_results is not None:
            checkpoint["eval_results"] = eval_results
        
        if metadata is not None:
            checkpoint["metadata"] = metadata
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint 已保存: {self.checkpoint_file.name}")
            
        except Exception as e:
            print(f"⚠️ 保存 checkpoint 失败: {e}")
    
    def get_completed_conversations(self) -> Set[str]:
        """
        获取已完成的对话 ID 集合
        
        Returns:
            已完成的对话 ID 集合
        """
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return set()
        
        completed = set()
        
        # 从 search_results 中获取
        if 'search_results' in checkpoint:
            completed.update(checkpoint['search_results'].keys())
        
        return completed
    
    def should_skip_stage(self, stage: str) -> bool:
        """
        检查是否应该跳过某个阶段
        
        Args:
            stage: 阶段名称 (add, search, answer, evaluate)
            
        Returns:
            True 如果应该跳过
        """
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        
        completed_stages = set(checkpoint.get('completed_stages', []))
        return stage in completed_stages
    
    def delete_checkpoint(self):
        """删除 checkpoint 文件"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                print(f"🗑️  Checkpoint 已删除: {self.checkpoint_file.name}")
            except Exception as e:
                print(f"⚠️ 删除 checkpoint 失败: {e}")
    
    def get_search_results(self) -> Optional[Dict]:
        """获取已保存的搜索结果"""
        checkpoint = self.load_checkpoint()
        if checkpoint and 'search_results' in checkpoint:
            return checkpoint['search_results']
        return None
    
    def get_answer_results(self) -> Optional[Dict]:
        """获取已保存的答案结果"""
        checkpoint = self.load_checkpoint()
        if checkpoint and 'answer_results' in checkpoint:
            return checkpoint['answer_results']
        return None
    
    # ==================== 细粒度 Checkpoint 方法 ====================
    
    def save_add_progress(self, completed_convs: set, memcells_dir: Path):
        """
        保存 Add 阶段的细粒度进度（记录已完成的会话 ID）
        
        Args:
            completed_convs: 已完成的会话 ID 集合
            memcells_dir: MemCells 保存目录（用于检查文件是否存在）
        """
        # Add 阶段的 checkpoint 策略：
        # 每处理完一个会话，将 MemCells 保存到 {output_dir}/memcells/{conv_id}.json
        # 不需要额外的 checkpoint 文件，直接检查 memcells 目录即可
        pass  # 文件本身就是 checkpoint
    
    def load_add_progress(self, memcells_dir: Path, all_conv_ids: list) -> set:
        """
        加载 Add 阶段的细粒度进度（检查哪些会话已完成）
                
        Args:
            memcells_dir: MemCells 保存目录
            all_conv_ids: 所有会话 ID 列表
            
        Returns:
            已完成的会话 ID 集合
        """
        import json
        
        completed_convs = set()
        
        if not memcells_dir.exists():
            print(f"\n🆕 No previous memcells found, starting from scratch")
            return completed_convs
        
        print(f"\n🔍 Checking for completed conversations in: {memcells_dir}")
        
        for conv_id in all_conv_ids:
            # 🔥 修复：匹配 stage1 实际保存的文件名格式
            output_file = memcells_dir / f"memcell_list_conv_{conv_id}.json"
            if output_file.exists():
                # 验证文件有效性（非空且可解析）
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data and len(data) > 0:  # 确保有数据
                            completed_convs.add(conv_id)
                            print(f"✅ 跳过已完成的会话: {conv_id} ({len(data)} memcells)")
                except Exception as e:
                    print(f"⚠️  会话 {conv_id} 文件损坏，将重新处理: {e}")
        
        if completed_convs:
            print(f"\n📊 发现 {len(completed_convs)}/{len(all_conv_ids)} 个已完成的会话")
        
        return completed_convs
    
    def save_search_progress(self, search_results: Dict[str, Any]):
        """
        保存 Search 阶段的细粒度进度（每处理完一个会话就保存）
        
        Args:
            search_results: 当前累积的所有搜索结果
                格式: {conv_id: [{"question_id": ..., "results": ...}, ...], ...}
        """
        try:
            with open(self.search_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {len(search_results)} conversations")
            
        except Exception as e:
            print(f"⚠️  Failed to save search checkpoint: {e}")
    
    def load_search_progress(self) -> Dict[str, Any]:
        """
        加载 Search 阶段的细粒度进度
        
        Returns:
            已保存的搜索结果，如果不存在则返回空字典
        """
        if not self.search_checkpoint.exists():
            print(f"\n🆕 No checkpoint found, starting from scratch")
            return {}
        
        try:
            print(f"\n🔄 Found checkpoint file: {self.search_checkpoint}")
            with open(self.search_checkpoint, 'r', encoding='utf-8') as f:
                search_results = json.load(f)
            
            print(f"✅ Loaded {len(search_results)} conversations from checkpoint")
            print(f"   Already processed: {sorted(search_results.keys())}")
            
            return search_results
            
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"   Starting from scratch...")
            return {}
    
    def delete_search_checkpoint(self):
        """删除 Search 阶段的细粒度检查点"""
        if self.search_checkpoint.exists():
            try:
                self.search_checkpoint.unlink()
                print(f"🗑️  Checkpoint file removed (task completed)")
            except Exception as e:
                print(f"⚠️  Failed to remove checkpoint: {e}")
    
    def save_answer_progress(self, answer_results: Dict[str, Any], completed: int, total: int):
        """
        保存 Answer 阶段的细粒度进度（每 SAVE_INTERVAL 个问题保存一次）
        
        Args:
            answer_results: 当前累积的所有答案结果
            completed: 已完成的问题数
            total: 总问题数
        """
        try:
            checkpoint_path = self.output_dir / f"responses_checkpoint_{completed}.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(answer_results, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
            
        except Exception as e:
            print(f"⚠️  Failed to save answer checkpoint: {e}")
    
    def load_answer_progress(self) -> Dict[str, Any]:
        """
        加载 Answer 阶段的细粒度进度（查找最新的检查点文件）
        
        Returns:
            已保存的答案结果，如果不存在则返回空字典
        """
        # 查找所有 responses_checkpoint_*.json 文件
        checkpoint_files = list(self.output_dir.glob("responses_checkpoint_*.json"))
        
        if not checkpoint_files:
            print(f"\n🆕 No answer checkpoint found, starting from scratch")
            return {}
        
        # 找到最新的检查点文件（按文件名中的数字排序）
        try:
            latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
            
            print(f"\n🔄 Found checkpoint file: {latest_checkpoint.name}")
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                answer_results = json.load(f)
            
            print(f"✅ Loaded {len(answer_results)} answers from checkpoint")
            
            return answer_results
            
        except Exception as e:
            print(f"⚠️  Failed to load answer checkpoint: {e}")
            print(f"   Starting from scratch...")
            return {}
    
    def delete_answer_checkpoints(self):
        """删除 Answer 阶段的所有细粒度检查点"""
        checkpoint_files = list(self.output_dir.glob("responses_checkpoint_*.json"))
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
                print(f"  🗑️  Removed checkpoint: {checkpoint_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to remove checkpoint {checkpoint_file.name}: {e}")

