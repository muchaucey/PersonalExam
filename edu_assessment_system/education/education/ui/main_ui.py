"""
UI界面模块
使用Gradio构建Web界面
"""

import gradio as gr
import logging
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EducationSystemUI:
    
    def __init__(self, system_core):

        self.system = system_core
        self.current_session = None
        
        logger.info("UI界面初始化完成")
    
    def create_interface(self) -> gr.Blocks:

        with gr.Blocks(title="教育评估", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""

    
            """)
            
            with gr.Tabs():
                # Tab 1: 学生测评
                with gr.Tab("📝 学生测评"):
                    self._create_student_tab()
                
                # Tab 2: 教师管理
                with gr.Tab("👨‍🏫 教师管理"):
                    self._create_teacher_tab()
                
                # Tab 3: 知识图谱
                with gr.Tab("🕸️ 知识图谱"):
                    self._create_kg_tab()
                
                # Tab 4: 系统设置
                with gr.Tab("⚙️ 系统设置"):
                    self._create_settings_tab()
        
        return interface
    
    def _create_student_tab(self):

        gr.Markdown("### 选择知识点并开始测评")
        
        # 知识点选择
        with gr.Row():
            knowledge_dropdown = gr.Dropdown(
                choices=list(self.system.get_knowledge_points()),
                label="选择知识点",
                value=list(self.system.get_knowledge_points())[0] if self.system.get_knowledge_points() else None
            )
            student_id_input = gr.Textbox(
                label="学生ID",
                placeholder="请输入学生ID（可选）",
                value="student_001"
            )
            num_questions = gr.Slider(
                minimum=3,
                maximum=15,
                value=5,
                step=1,
                label="题目数量"
            )
        
        start_btn = gr.Button("🚀 开始测评", variant="primary")
        
        # 测评区域
        gr.Markdown("---")
        
        session_state = gr.State(value=None)  # 存储会话状态
        
        with gr.Column(visible=False) as quiz_area:
            question_display = gr.Markdown("### 题目加载中...")
            
            with gr.Row():
                current_q_num = gr.Number(label="当前题号", value=1, interactive=False)
                total_q_num = gr.Number(label="总题数", value=5, interactive=False)
            
            question_text = gr.Textbox(
                label="题目",
                lines=5,
                interactive=False
            )
            
            answer_input = gr.Textbox(
                label="你的答案",
                lines=3,
                placeholder="请输入你的答案..."
            )
            
            with gr.Row():
                submit_answer_btn = gr.Button("✓ 提交答案", variant="primary")
                next_question_btn = gr.Button("→ 下一题", visible=False)
            
            feedback_text = gr.Markdown("", visible=False)
        
        # 评估报告区域
        with gr.Column(visible=False) as report_area:
            gr.Markdown("### 📊 评估报告")
            report_display = gr.Textbox(
                label="详细报告",
                lines=20,
                interactive=False
            )
            restart_btn = gr.Button("🔄 重新开始")
        
        # 事件绑定
        start_btn.click(
            fn=self._start_assessment,
            inputs=[knowledge_dropdown, student_id_input, num_questions],
            outputs=[session_state, quiz_area, question_text, 
                    current_q_num, total_q_num, answer_input]
        )
        
        submit_answer_btn.click(
            fn=self._submit_answer,
            inputs=[session_state, answer_input],
            outputs=[session_state, feedback_text, submit_answer_btn, 
                    next_question_btn, answer_input]
        )
        
        next_question_btn.click(
            fn=self._next_question,
            inputs=[session_state],
            outputs=[session_state, question_text, current_q_num,
                    feedback_text, submit_answer_btn, next_question_btn,
                    answer_input, quiz_area, report_area, report_display]
        )
        
        restart_btn.click(
            fn=lambda: (None, gr.update(visible=False), gr.update(visible=False), ""),
            outputs=[session_state, quiz_area, report_area, answer_input]
        )
    
    def _create_teacher_tab(self):
        """创建教师管理标签页"""
        gr.Markdown("### 题库管理")
        
        with gr.Tab("📥 导入题目"):
            gr.Markdown("#### 从JSON文件导入题目")
            
            json_file = gr.File(label="选择JSON文件", file_types=[".json"])
            import_btn = gr.Button("导入", variant="primary")
            import_status = gr.Textbox(label="导入状态", interactive=False)
            
            import_btn.click(
                fn=self._import_questions,
                inputs=[json_file],
                outputs=[import_status]
            )
        
        with gr.Tab("➕ 添加单题"):
            gr.Markdown("#### 手动添加单个题目")
            
            with gr.Row():
                q_knowledge = gr.Dropdown(
                    choices=list(self.system.get_knowledge_points()),
                    label="知识点"
                )
                q_difficulty = gr.Dropdown(
                    choices=["简单", "中等", "困难"],
                    label="难度"
                )
            
            q_question = gr.Textbox(label="题目", lines=3)
            q_answer = gr.Textbox(label="答案", lines=2)
            q_explanation = gr.Textbox(label="解析", lines=4)
            
            add_btn = gr.Button("添加题目", variant="primary")
            add_status = gr.Textbox(label="添加状态", interactive=False)
            
            add_btn.click(
                fn=self._add_single_question,
                inputs=[q_knowledge, q_difficulty, q_question, 
                       q_answer, q_explanation],
                outputs=[add_status]
            )
        
        with gr.Tab("🔍 查看题库"):
            gr.Markdown("#### 题库统计与浏览")
            
            refresh_btn = gr.Button("🔄 刷新统计")
            stats_display = gr.Textbox(
                label="题库统计",
                lines=10,
                interactive=False
            )
            
            with gr.Row():
                filter_knowledge = gr.Dropdown(
                    choices=["全部"] + list(self.system.get_knowledge_points()),
                    label="筛选知识点",
                    value="全部"
                )
                filter_difficulty = gr.Dropdown(
                    choices=["全部", "简单", "中等", "困难"],
                    label="筛选难度",
                    value="全部"
                )
            
            search_btn = gr.Button("搜索")
            questions_display = gr.Dataframe(
                headers=["题号", "知识点", "难度", "问题"],
                interactive=False
            )
            
            refresh_btn.click(
                fn=self._get_database_stats,
                outputs=[stats_display]
            )
            
            search_btn.click(
                fn=self._search_questions,
                inputs=[filter_knowledge, filter_difficulty],
                outputs=[questions_display]
            )
    
    def _create_kg_tab(self):
        """创建知识图谱标签页"""
        gr.Markdown("### 知识图谱可视化")
        
        with gr.Row():
            layout_choice = gr.Dropdown(
                choices=["spring", "circular", "kamada_kawai"],
                label="布局算法",
                value="spring"
            )
            generate_btn = gr.Button("🎨 生成图谱", variant="primary")
        
        # 使用Plotly组件直接显示交互式图表
        kg_display = gr.Plot(label="知识图谱", show_label=True)
        
        download_btn = gr.Button("💾 下载图谱HTML")
        download_file = gr.File(label="下载", visible=False)
        
        generate_btn.click(
            fn=self._generate_kg_plotly,
            inputs=[layout_choice],
            outputs=[kg_display]
        )
        
        download_btn.click(
            fn=self._download_kg_html,
            outputs=[download_file]
        )
    
    def _create_settings_tab(self):
        """创建系统设置标签页"""
        gr.Markdown("### 系统信息")
        
        system_info = gr.Textbox(
            label="系统状态",
            value=self._get_system_info(),
            lines=15,
            interactive=False
        )
        
        gr.Markdown("### 模型管理")
        
        with gr.Row():
            reload_models_btn = gr.Button("🔄 重新加载模型")
            clear_cache_btn = gr.Button("🗑️ 清除缓存")
        
        model_status = gr.Textbox(label="操作状态", interactive=False)
        
        reload_models_btn.click(
            fn=self._reload_models,
            outputs=[model_status]
        )
        
        clear_cache_btn.click(
            fn=self._clear_cache,
            outputs=[model_status]
        )
    
    # 回调函数实现
    def _start_assessment(self, knowledge: str, student_id: str, num: int):
        """开始测评"""
        try:
            session = self.system.start_assessment(knowledge, student_id, int(num))
            
            if session is None:
                return None, gr.update(visible=False), "无法开始测评", 1, num, ""
            
            question = session['current_question']
            
            return (
                session,
                gr.update(visible=True),
                f"**题目 {session['current_index']}/{session['total_questions']}**\n\n{question['问题']}",
                session['current_index'],
                session['total_questions'],
                ""
            )
        except Exception as e:
            logger.error(f"开始测评失败: {e}")
            return None, gr.update(visible=False), f"错误: {str(e)}", 1, num, ""
    
    def _submit_answer(self, session, answer):
        """提交答案"""
        if session is None:
            return session, "请先开始测评", gr.update(), gr.update(), ""
        
        try:
            session = self.system.submit_answer(session, answer)
            
            feedback = f"""
### 答题反馈

**你的答案:** {answer}

**标准答案:** {session['last_result']['question']['答案']}

**判定结果:** {'✓ 正确!' if session['last_result']['is_correct'] else '✗ 错误'}

**解析:** {session['last_result']['question']['解析']}
"""
            
            return (
                session,
                gr.update(value=feedback, visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                ""
            )
        except Exception as e:
            logger.error(f"提交答案失败: {e}")
            return session, f"错误: {str(e)}", gr.update(), gr.update(), answer
    
    def _next_question(self, session):
        """下一题"""
        if session is None:
            return None, "", 1, "", gr.update(), gr.update(), "", gr.update(), gr.update(), ""
        
        try:
            # 检查是否还有题目
            if session['current_index'] >= session['total_questions']:
                # 生成评估报告
                report = self.system.generate_report(session)
                
                return (
                    session,
                    "",
                    session['current_index'],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    report
                )
            
            # 加载下一题
            session = self.system.next_question(session)
            question = session['current_question']
            
            return (
                session,
                f"**题目 {session['current_index']}/{session['total_questions']}**\n\n{question['问题']}",
                session['current_index'],
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                gr.update(visible=True),
                gr.update(visible=False),
                ""
            )
        except Exception as e:
            logger.error(f"加载下一题失败: {e}")
            return session, f"错误: {str(e)}", 1, "", gr.update(), gr.update(), "", gr.update(), gr.update(), ""
    
    def _import_questions(self, file_obj):
        """导入题目"""
        if file_obj is None:
            return "请选择文件"
        
        try:
            result = self.system.import_questions(file_obj.name)
            return f"成功导入 {result} 道题目"
        except Exception as e:
            logger.error(f"导入失败: {e}")
            return f"导入失败: {str(e)}"
    
    def _add_single_question(self, knowledge, difficulty, question, answer, explanation):
        """添加单个题目"""
        try:
            question_data = {
                "知识点": knowledge,
                "难度": difficulty,
                "问题": question,
                "答案": answer,
                "解析": explanation
            }
            
            success = self.system.add_question(question_data)
            
            if success:
                return "题目添加成功!"
            else:
                return "题目添加失败"
        except Exception as e:
            logger.error(f"添加题目失败: {e}")
            return f"添加失败: {str(e)}"
    
    def _get_database_stats(self):
        """获取数据库统计"""
        try:
            stats = self.system.get_database_statistics()
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"获取统计失败: {str(e)}"
    
    def _search_questions(self, knowledge, difficulty):
        """搜索题目"""
        try:
            kp = None if knowledge == "全部" else knowledge
            diff = None if difficulty == "全部" else difficulty
            
            questions = self.system.search_questions(kp, diff)
            
            # 格式化为表格数据
            data = []
            for q in questions:
                data.append([
                    q.get('题号', 'N/A'),
                    q.get('知识点', ''),
                    q.get('难度', ''),
                    q.get('问题', '')[:50] + '...'
                ])
            
            return data
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def _generate_kg_visualization(self, layout):
        """生成知识图谱可视化"""
        try:
            html_content = self.system.generate_kg_visualization(layout)
            return html_content
        except Exception as e:
            logger.error(f"生成图谱失败: {e}")
            return f"<p>生成失败: {str(e)}</p>"
    
    def _generate_kg_plotly(self, layout):
        """生成Plotly知识图谱图表"""
        try:
            # 获取Plotly图表对象
            fig = self.system.generate_kg_plotly(layout)
            return fig
        except Exception as e:
            logger.error(f"生成Plotly图谱失败: {e}")
            # 返回一个空的图表
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="知识图谱生成失败<br>请检查题库数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
    
    def _download_kg_html(self):
        """下载知识图谱HTML"""
        try:
            file_path = self.system.export_kg_html()
            return gr.update(value=file_path, visible=True)
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return gr.update(visible=False)
    
    def _get_system_info(self):
        """获取系统信息"""
        return self.system.get_system_info()
    
    def _reload_models(self):
        """重新加载模型"""
        try:
            self.system.reload_models()
            return "模型重新加载成功"
        except Exception as e:
            return f"重新加载失败: {str(e)}"
    
    def _clear_cache(self):
        """清除缓存"""
        try:
            self.system.clear_cache()
            return "缓存已清除"
        except Exception as e:
            return f"清除失败: {str(e)}"


def create_ui(system_core) -> gr.Blocks:
    """
    工厂函数:创建UI界面
    
    Args:
        system_core: 系统核心实例
        
    Returns:
        Gradio Blocks对象
    """
    ui = EducationSystemUI(system_core)
    return ui.create_interface()


if __name__ == "__main__":
    print("请从主程序运行UI")
