"""
Base Converter

定义数据集转换器的基类接口。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseConverter(ABC):
    """数据集转换器基类"""
    
    @abstractmethod
    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        将数据集转换为 Locomo 格式
        
        Args:
            input_paths: 输入文件路径字典，如 {"raw": "path/to/raw.json"}
            output_path: 输出文件路径
        """
        pass
    
    @abstractmethod
    def get_input_files(self) -> Dict[str, str]:
        """
        返回需要的输入文件列表
        
        Returns:
            文件名映射，如 {"raw": "longmemeval_s_cleaned.json"}
        """
        pass
    
    def get_output_filename(self) -> str:
        """
        返回输出文件名（converted 版本）
        
        Returns:
            文件名，如 "longmemeval_s_locomo_style.json"
        """
        return "converted_locomo_style.json"
    
    def needs_conversion(self, data_dir: Path) -> bool:
        """
        检查是否需要转换（converted 文件是否存在）
        
        Args:
            data_dir: 数据集目录
            
        Returns:
            True 如果需要转换，False 如果已存在 converted 版本
        """
        output_file = data_dir / self.get_output_filename()
        return not output_file.exists()
    
    def get_converted_path(self, data_dir: Path) -> Path:
        """
        获取 converted 文件的路径
        
        Args:
            data_dir: 数据集目录
            
        Returns:
            converted 文件路径
        """
        return data_dir / self.get_output_filename()


