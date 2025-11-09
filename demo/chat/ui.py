"""终端 UI 工具

提供美观的终端输出格式。
"""

from typing import List, Dict, Any, Optional

from demo.i18n_texts import I18nTexts
from common_utils.cli_ui import CLIUI


class ChatUI:
    """终端界面工具类"""
    
    @staticmethod
    def _ui() -> CLIUI:
        """获取 UI 实例"""
        return CLIUI()
    
    @staticmethod
    def clear_screen():
        """清空屏幕"""
        print("\033[2J\033[H", end="")
        import sys
        sys.stdout.flush()
    
    @staticmethod
    def print_banner(texts: I18nTexts):
        """打印欢迎横幅"""
        ui = ChatUI._ui()
        print()
        ui.banner(texts.get("banner_title"), subtitle=texts.get("banner_subtitle"))
        print()
    
    @staticmethod
    def print_group_list(groups: List[Dict[str, Any]], texts: I18nTexts):
        """显示群组列表"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("groups_available_title"))
        
        rows = []
        for group in groups:
            index = group["index"]
            group_id = group["group_id"]
            name = group.get("name", group_id)
            count = group["memcell_count"]
            count_text = f"💾 {count} " + ("memories" if texts.language == "en" else "条记忆")
            rows.append([f"[{index}]", group_id, f'📝 "{name}"', count_text])
        
        headers = [
            texts.get("table_header_index"),
            texts.get("table_header_group"),
            texts.get("table_header_name"),
            texts.get("table_header_count"),
        ]
        ui.table(headers=headers, rows=rows, aligns=["right", "left", "left", "right"])
    
    @staticmethod
    def print_retrieved_memories(
        memories: List[Dict[str, Any]],
        texts: I18nTexts,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
    ):
        """显示检索到的记忆"""
        ui = ChatUI._ui()
        
        heading = f"🔍 {texts.get('retrieval_complete')}"
        shown_count = len(memories)
        if shown_count > 0:
            heading += f" - {texts.get('retrieval_showing', shown=shown_count)}"
        
        # 显示检索模式和耗时
        if retrieval_metadata:
            retrieval_mode = retrieval_metadata.get("retrieval_mode", "rrf")
            latency_ms = retrieval_metadata.get("total_latency_ms", 0.0)
            
            mode_map = {
                "rrf": "RRF融合",
                "embedding": "纯向量",
                "bm25": "纯BM25",
                "agentic": "Agentic",
                "agentic_fallback": "Agentic(降级)",
            }
            mode_text = mode_map.get(retrieval_mode, retrieval_mode)
            heading += f" | {mode_text} | {int(latency_ms)}ms"
        
        ui.section_heading(heading)
        
        # 🔥 Agentic 检索特殊信息显示
        if retrieval_metadata and retrieval_metadata.get("retrieval_mode") == "agentic":
            agentic_info = []
            
            # LLM 判断结果
            is_sufficient = retrieval_metadata.get("is_sufficient")
            if is_sufficient is not None:
                status = "✅ 充分" if is_sufficient else "❌ 不充分"
                agentic_info.append(f"LLM 判断: {status}")
            
            # 是否多轮
            is_multi_round = retrieval_metadata.get("is_multi_round", False)
            if is_multi_round:
                agentic_info.append("🔄 多轮检索")
                
                # 改进查询
                refined_queries = retrieval_metadata.get("refined_queries", [])
                if refined_queries:
                    agentic_info.append(f"生成查询: {len(refined_queries)} 个")
            else:
                agentic_info.append("⚡ 单轮检索")
            
            # Round 统计
            round1_count = retrieval_metadata.get("round1_count", 0)
            round2_count = retrieval_metadata.get("round2_count", 0)
            if round1_count:
                agentic_info.append(f"R1: {round1_count} 条")
            if round2_count:
                agentic_info.append(f"R2: {round2_count} 条")
            
            if agentic_info:
                print()
                ui.note(" | ".join(agentic_info), icon="🤖")
                
                # 显示 LLM 推理
                reasoning = retrieval_metadata.get("reasoning")
                if reasoning:
                    print(f"   💭 {reasoning}")
                
                # 显示改进查询
                if is_multi_round:
                    refined_queries = retrieval_metadata.get("refined_queries", [])
                    if refined_queries:
                        print(f"   📝 改进查询:")
                        for i, q in enumerate(refined_queries[:3], 1):
                            print(f"      {i}. {q[:60]}{'...' if len(q) > 60 else ''}")
        
        # 显示记忆列表
        lines = []
        for i, mem in enumerate(memories, start=1):
            timestamp = mem.get("timestamp", "")[:10]
            subject = mem.get("subject", "")
            summary = mem.get("summary", "")
            content = subject or summary or ""
            lines.append(f"📌 [{i:2d}]  {timestamp}  │  {content}")
        
        if lines:
            print()
            ui.panel(lines)
    
    @staticmethod
    def print_generating_indicator(texts: I18nTexts):
        """显示生成进度提示"""
        ui = ChatUI._ui()
        print()
        ui.note(f"🤔 {texts.get('chat_generating')}", icon="⏳")
    
    @staticmethod
    def print_generation_complete(texts: I18nTexts):
        """清除生成提示并显示完成标识"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
        ui = ChatUI._ui()
        ui.success(f"✓ {texts.get('chat_generation_complete')}")
    
    @staticmethod
    def clear_progress_indicator():
        """清除进度提示"""
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")
    
    @staticmethod
    def print_assistant_response(response: str, texts: I18nTexts):
        """显示助手回答
        
        优化后的显示：
        - 主要显示 answer（大标题）
        - references 和 confidence 作为元数据（小字）
        - 隐藏 reasoning
        """
        ui = ChatUI._ui()
        print()
        
        # 尝试解析 JSON 响应
        try:
            import json
            data = json.loads(response)
            
            # 提取字段
            answer = data.get("answer", "")
            references = data.get("references", [])
            confidence = data.get("confidence", "")
            
            # 显示主回答（大标题）
            ui.panel([answer], title=f"🤖 {texts.get('response_assistant_title')}")
            
            # 显示元数据（小字，弱化显示）
            metadata_parts = []
            if references:
                ref_text = ", ".join(references)
                metadata_parts.append(f"📚 {ref_text}")
            if confidence:
                confidence_icon = {"high": "✓", "medium": "~", "low": "?"}.get(confidence, "")
                metadata_parts.append(f"{confidence_icon} {confidence}")
            
            if metadata_parts:
                metadata_line = "  │  ".join(metadata_parts)
                print(f"  {metadata_line}")
            
        except (json.JSONDecodeError, ValueError):
            # 如果不是 JSON 格式，直接显示原始回答
            ui.panel([response], title=f"🤖 {texts.get('response_assistant_title')}")
        
        ui.rule()
        print()
    
    @staticmethod
    def print_help(texts: I18nTexts):
        """显示帮助信息"""
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("cmd_help_title"))
        lines = [
            f"🚪  {texts.get('cmd_exit')}",
            f"🧹  {texts.get('cmd_clear')}",
            f"🔄  {texts.get('cmd_reload')}",
            f"❓  {texts.get('cmd_help')}",
        ]
        ui.panel(lines)
        print()
    
    @staticmethod
    def print_info(message: str, texts: I18nTexts):
        """显示信息提示"""
        ui = ChatUI._ui()
        print()
        ui.success(f"✓ {message}")
        print()
    
    @staticmethod
    def print_error(message: str, texts: I18nTexts):
        """显示错误信息"""
        ui = ChatUI._ui()
        print()
        ui.error(f"✗ {message}")
        print()

