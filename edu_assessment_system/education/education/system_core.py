"""
系统核心模块
整合所有功能组件,提供统一的接口
优化版：实现真正的自适应出题、实时题目调整、智能题目选择
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import plotly.graph_objects as go
import random

logger = logging.getLogger(__name__)


class EducationSystemCore:
    """教育评估系统核心（优化版）"""
    
    def __init__(self, config):
        """
        初始化系统核心
        
        Args:
            config: 配置模块
        """
        self.config = config
        
        # 初始化各个组件
        self.question_db = None
        self.embedding_model = None
        self.pangu_model = None
        self.evaluator = None
        self.visualizer = None
        self.rag_engine = None
        self.bkt_algorithm = None
        
        # 运行时状态
        self.models_loaded = False
        
        logger.info("✅ 系统核心初始化完成（自适应增强版）")
    
    def initialize(self):
        """初始化所有组件"""
        logger.info("🔄 正在初始化系统组件...")
        
        try:
            # 导入必要的模块
            from models import create_llm_model, create_embedding_model
            from data_management.question_db import create_question_database
            from utils.evaluator import create_evaluator
            from visualization.kg_visualizer import create_visualizer
            
            # 1. 初始化题库
            logger.info("📚 初始化题库...")
            self.question_db = create_question_database(str(self.config.QUESTION_DB))
            
            # 2. 初始化模型
            logger.info("🚀 初始化盘古7B模型（单例模式）...")
            
            self.embedding_model = create_embedding_model(
                self.config.BGE_M3_MODEL_PATH,
                self.config.EMBEDDING_MODEL_CONFIG
            )
            
            self.pangu_model = create_llm_model(
                'pangu',
                self.config.PANGU_MODEL_PATH,
                self.config.PANGU_MODEL_CONFIG
            )
            
            logger.info("🔄 预加载盘古7B模型...")
            self.pangu_model.load_model()
            logger.info("✅ 盘古7B模型预加载完成")
            
            # 3. 初始化功能组件
            logger.info("⚙️  初始化功能组件...")
            
            # BKT算法（增强版，支持持久化）
            from utils.bkt_algorithm import create_bkt_algorithm
            self.bkt_algorithm = create_bkt_algorithm(
                storage_path=str(self.config.DATA_DIR / "student_states.json")
            )
            
            # 个性化评估器（需要BKT算法实例）
            self.evaluator = create_evaluator(
                self.pangu_model,
                self.bkt_algorithm,
                self.config.EVALUATION_CONFIG
            )
            
            self.visualizer = create_visualizer(
                self.config.VISUALIZATION_CONFIG
            )
            
            self.models_loaded = True
            logger.info("✅ 系统初始化完成 - 深度个性化自适应学习版")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            raise RuntimeError(f"系统初始化失败: {e}")
    
    def get_knowledge_points(self) -> List[str]:
        """获取所有知识点"""
        return list(self.config.QUESTION_TYPES.keys())
    
    def _select_adaptive_question(self, student_id: str, knowledge_point: str,
                                 current_mastery: float, available_questions: List[Dict[str, Any]],
                                 used_questions: set) -> Optional[Dict[str, Any]]:
        """
        根据学生当前掌握度智能选择题目（核心自适应逻辑）
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            current_mastery: 当前掌握概率
            available_questions: 可用题目列表
            used_questions: 已使用的题目ID集合
            
        Returns:
            选中的题目，如果没有合适的题目则返回None
        """
        # 过滤掉已使用的题目
        candidates = [q for q in available_questions if q.get('题号') not in used_questions]
        
        if not candidates:
            logger.warning(f"⚠️  没有可用的题目了")
            return None
        
        # 根据掌握度确定目标难度
        if current_mastery < 0.3:
            # 基础薄弱，选择简单题目
            target_difficulty = "简单"
            fallback_difficulties = ["中等"]
            logger.debug(f"🎯 掌握度 {current_mastery:.3f} < 0.3，目标难度：简单")
        elif current_mastery < 0.7:
            # 中等水平，选择中等题目
            target_difficulty = "中等"
            fallback_difficulties = ["简单", "困难"]
            logger.debug(f"🎯 掌握度 {current_mastery:.3f} 在 [0.3, 0.7)，目标难度：中等")
        else:
            # 掌握良好，选择困难题目
            target_difficulty = "困难"
            fallback_difficulties = ["中等"]
            logger.debug(f"🎯 掌握度 {current_mastery:.3f} ≥ 0.7，目标难度：困难")
        
        # 先尝试目标难度
        target_candidates = [q for q in candidates if q.get('难度') == target_difficulty]
        
        if target_candidates:
            selected = random.choice(target_candidates)
            logger.info(f"✅ 选中题目 {selected.get('题号')} (难度: {target_difficulty})")
            return selected
        
        # 如果目标难度题目不足，尝试备选难度
        logger.debug(f"⚠️  目标难度 {target_difficulty} 题目不足，尝试备选难度")
        for fallback_diff in fallback_difficulties:
            fallback_candidates = [q for q in candidates if q.get('难度') == fallback_diff]
            if fallback_candidates:
                selected = random.choice(fallback_candidates)
                logger.info(f"✅ 使用备选难度，选中题目 {selected.get('题号')} (难度: {fallback_diff})")
                return selected
        
        # 如果所有难度都试过了，随机选择一个
        logger.warning(f"⚠️  无法按难度筛选，随机选择题目")
        selected = random.choice(candidates)
        logger.info(f"✅ 随机选中题目 {selected.get('题号')} (难度: {selected.get('难度')})")
        return selected
    
    def start_assessment(self, knowledge_point: str, 
                        student_id: str = "default_student",
                        num_questions: int = 10) -> Optional[Dict[str, Any]]:
        """
        开始测评（真正的自适应版本）
        
        Args:
            knowledge_point: 知识点
            student_id: 学生ID
            num_questions: 题目数量
            
        Returns:
            会话状态字典
        """
        try:
            logger.info(f"🎯 开始自适应测评: {knowledge_point}, 学生: {student_id}, 数量: {num_questions}")
            
            # 检查题库
            all_available_questions = self.question_db.get_questions_filtered(
                knowledge_point=knowledge_point
            )
            
            if not all_available_questions:
                logger.error(f"❌ 题库中没有任何关于'{knowledge_point}'的题目")
                return None
            
            if len(all_available_questions) < num_questions:
                logger.warning(f"⚠️  题库题目数({len(all_available_questions)})少于需求({num_questions})")
                num_questions = len(all_available_questions)
            
            # 获取学生当前状态
            state = self.bkt_algorithm.get_student_state(student_id, knowledge_point)
            current_mastery = state.mastery_prob
            
            logger.info(f"📊 学生 {student_id} 在 {knowledge_point} 的当前掌握度: {current_mastery:.3f}")
            
            # 智能选择第一题
            used_question_ids = set()
            first_question = self._select_adaptive_question(
                student_id, knowledge_point, current_mastery,
                all_available_questions, used_question_ids
            )
            
            if not first_question:
                logger.error(f"❌ 无法选择第一题")
                return None
            
            used_question_ids.add(first_question.get('题号'))
            
            # 创建会话
            session = {
                'knowledge_point': knowledge_point,
                'student_id': student_id,
                'total_questions': num_questions,
                'current_index': 1,
                'current_question': first_question,
                'questions': [first_question],  # 已选题目列表
                'answer_records': [],
                'last_result': None,
                'used_question_ids': used_question_ids,
                'all_available_questions': all_available_questions,
                'current_mastery': current_mastery,
                'initial_mastery': current_mastery
            }
            
            logger.info(f"✅ 测评开始，第1题: {first_question.get('问题', '')[:50]}... (难度: {first_question.get('难度')})")
            return session
            
        except Exception as e:
            logger.error(f"❌ 开始测评失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def submit_answer(self, session: Dict[str, Any], 
                     student_answer: str) -> Dict[str, Any]:
        """
        提交答案（优化版，自动调整后续题目）
        
        Args:
            session: 会话状态
            student_answer: 学生答案
            
        Returns:
            更新后的会话状态
        """
        try:
            question = session['current_question']
            
            logger.info(f"✍️  评估答案（题目 {session['current_index']}/{session['total_questions']}）...")
            
            # 检查答案
            is_correct, reason = self.evaluator.check_answer(
                question,
                student_answer,
                self.config.PROMPTS['answer_check']
            )
            
            logger.info(f"✅ 答案评估完成: {'✓ 正确' if is_correct else '✗ 错误'}")
            
            # ⭐ 关键：记录答题到BKT算法，获取更新后的掌握度
            bkt_result = self.bkt_algorithm.record_answer(
                session['student_id'],
                session['knowledge_point'],
                question,
                is_correct
            )
            
            new_mastery = bkt_result['current_mastery']
            mastery_change = bkt_result['mastery_change']
            recommended_difficulty = bkt_result['recommended_difficulty']
            
            logger.info(f"📊 BKT更新: 掌握度 {bkt_result['previous_mastery']:.3f} → {new_mastery:.3f} "
                       f"(变化: {mastery_change:+.3f}), 推荐难度: {recommended_difficulty}")
            
            # 更新会话中的掌握度
            session['current_mastery'] = new_mastery
            
            # 记录答题
            record = {
                'question': question,
                'student_answer': student_answer,
                'is_correct': is_correct,
                'check_reason': reason,
                'mastery_before': bkt_result['previous_mastery'],
                'mastery_after': new_mastery,
                'mastery_change': mastery_change
            }
            
            session['answer_records'].append(record)
            session['last_result'] = record
            
            # ⭐⭐ 核心自适应逻辑：如果还有后续题目，根据新的掌握度选择下一题
            if session['current_index'] < session['total_questions']:
                logger.info(f"🔄 根据新掌握度 {new_mastery:.3f} 动态选择下一题...")
                
                next_question = self._select_adaptive_question(
                    session['student_id'],
                    session['knowledge_point'],
                    new_mastery,
                    session['all_available_questions'],
                    session['used_question_ids']
                )
                
                if next_question:
                    session['questions'].append(next_question)
                    session['used_question_ids'].add(next_question.get('题号'))
                    logger.info(f"✅ 已准备下一题 (难度: {next_question.get('难度')})")
                else:
                    logger.warning(f"⚠️  无法选择下一题，提前结束测评")
                    session['total_questions'] = session['current_index']
            
            return session
            
        except Exception as e:
            logger.error(f"❌ 提交答案失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            session['last_result'] = {
                'question': session['current_question'],
                'student_answer': student_answer,
                'is_correct': False,
                'check_reason': f"处理失败: {str(e)}"
            }
            return session
    
    def next_question(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载下一题
        
        Args:
            session: 会话状态
            
        Returns:
            更新后的会话状态
        """
        session['current_index'] += 1
        
        if session['current_index'] <= len(session['questions']):
            session['current_question'] = session['questions'][session['current_index'] - 1]
            logger.info(f"📄 加载第 {session['current_index']} 题: {session['current_question'].get('问题', '')[:50]}...")
        else:
            logger.info(f"✅ 所有题目已完成")
        
        return session
    
    def generate_report(self, session: Dict[str, Any]) -> str:
        """
        生成评估报告（深度个性化版本）
        
        Args:
            session: 会话状态
            
        Returns:
            个性化评估报告文本
        """
        try:
            logger.info("📝 生成深度个性化评估报告...")
            
            # 使用新的综合报告生成方法
            report = self.evaluator.generate_comprehensive_report(
                session['student_id'],
                session['knowledge_point'],
                session['answer_records']
            )
            
            logger.info("✅ 深度个性化评估报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"报告生成失败: {str(e)}"
    
    def generate_student_profile(self, student_id: str) -> Dict[str, Any]:
        """生成学生评估画像"""
        try:
            if not self.bkt_algorithm:
                return {"error": "BKT算法未初始化"}
            
            profile = self.bkt_algorithm.generate_student_profile(student_id)
            return profile
            
        except Exception as e:
            logger.error(f"❌ 生成学生画像失败: {e}")
            return {"error": str(e)}
    
    def import_questions(self, file_path: str) -> int:
        """导入题目"""
        return self.question_db.import_from_json(file_path)
    
    def add_question(self, question_data: Dict[str, Any]) -> bool:
        """添加题目"""
        return self.question_db.insert_question(question_data)
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        return self.question_db.get_statistics()
    
    def search_questions(self, knowledge_point: Optional[str] = None,
                        difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索题目"""
        return self.question_db.get_questions_filtered(
            knowledge_point=knowledge_point,
            difficulty=difficulty
        )
    
    def generate_kg_visualization(self, layout: str = 'spring') -> str:
        """生成知识图谱可视化"""
        try:
            questions = self.question_db.get_all_questions()
            self.visualizer.build_graph_from_questions(questions)
            fig = self.visualizer.create_plotly_figure(layout, "知识图谱")
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        except Exception as e:
            logger.error(f"❌ 生成图谱可视化失败: {e}")
            return f"<p>生成失败: {str(e)}</p>"
    
    def generate_kg_plotly(self, layout: str = 'spring'):
        """生成知识图谱Plotly图表对象"""
        try:
            questions = self.question_db.get_all_questions()
            self.visualizer.build_graph_from_questions(questions)
            fig = self.visualizer.create_plotly_figure(layout, "知识图谱")
            return fig
        except Exception as e:
            logger.error(f"❌ 生成Plotly图谱失败: {e}")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="知识图谱生成失败<br>请检查题库数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
    
    def export_kg_html(self) -> str:
        """导出知识图谱HTML文件"""
        try:
            questions = self.question_db.get_all_questions()
            self.visualizer.build_graph_from_questions(questions)
            
            output_path = str(self.config.KG_GRAPH_PATH)
            self.visualizer.save_interactive_html(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"❌ 导出图谱失败: {e}")
            raise
    
    def get_system_info(self) -> str:
        """获取系统信息"""
        # 统计学生数据
        student_count = 0
        total_records = 0
        if self.bkt_algorithm and hasattr(self.bkt_algorithm, 'student_states'):
            student_count = len(self.bkt_algorithm.student_states)
            total_records = sum(len(kps) for kps in self.bkt_algorithm.student_states.values())
        
        info = f"""
系统版本: {self.config.SYSTEM_INFO['version']}
作者: {self.config.SYSTEM_INFO['author']}
描述: {self.config.SYSTEM_INFO['description']}
模型: {self.config.SYSTEM_INFO['model']}
设备: {self.config.SYSTEM_INFO['device']}

模型状态:
  - 嵌入模型: {'已加载' if self.embedding_model else '未加载'}
  - 盘古7B模型: {'已加载' if (self.pangu_model and self.pangu_model.is_loaded) else '未加载'}
  - NPU设备数: {len(self.pangu_model.devices) if self.pangu_model else 0}

数据统计:
  - 题库路径: {self.config.QUESTION_DB}
  - 总题目数: {len(self.question_db.get_all_questions()) if self.question_db else 0}
  - 学生数量: {student_count}
  - 学习记录数: {total_records}

自适应功能:
  - BKT算法: {'✅ 已启用' if self.bkt_algorithm else '❌ 未启用'}
  - 状态持久化: {'✅ 已启用' if self.bkt_algorithm else '❌ 未启用'}
  - 智能题目选择: ✅ 已启用
  - 实时难度调整: ✅ 已启用

配置信息:
  - 工作目录: {self.config.WORKING_DIR}
  - 数据目录: {self.config.DATA_DIR}
"""
        return info
    
    def reload_models(self):
        """重新加载模型"""
        logger.info("🔄 重新加载模型...")
        
        if self.embedding_model:
            self.embedding_model.load_model()
        
        if self.pangu_model:
            self.pangu_model.load_model()
        
        logger.info("✅ 模型重新加载完成")
    
    def clear_cache(self):
        """清除缓存"""
        logger.info("🗑️  清除缓存...")
        
        import torch
        
        try:
            import torch_npu
            if torch.npu.is_available():
                for i in range(torch.npu.device_count()):
                    torch.npu.empty_cache()
                logger.info("✅ NPU缓存已清除")
        except:
            pass
        
        logger.info("✅ 缓存清除完成")


def create_system_core(config):
    """
    工厂函数:创建系统核心
    
    Args:
        config: 配置模块
        
    Returns:
        系统核心实例
    """
    core = EducationSystemCore(config)
    core.initialize()
    return core


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import config
    
    logging.basicConfig(level=logging.INFO)
    
    system = create_system_core(config)
    
    print("✅ 系统核心创建成功（自适应增强版）")
    print(system.get_system_info())