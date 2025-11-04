"""
贝叶斯知识追踪(BKT)算法模块
实现基于BKT的学生能力评估和个性化出题策略
优化版：支持状态持久化、个性化参数、详细学生画像
"""

import logging
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BKTParameters:
    """BKT算法参数"""
    # 初始掌握概率
    p_init: float = 0.3
    # 学习概率
    p_learn: float = 0.2
    # 猜测概率
    p_guess: float = 0.3
    # 失误概率
    p_slip: float = 0.1
    # 遗忘概率
    p_forget: float = 0.05


@dataclass
class StudentState:
    """学生状态记录"""
    student_id: str
    knowledge_point: str
    # 当前掌握概率
    mastery_prob: float
    # 历史答题记录
    answer_history: List[Dict[str, Any]]
    # 最近表现
    recent_performance: List[bool]
    # 参数
    params: BKTParameters
    # 创建时间
    created_at: str = ""
    # 最后更新时间
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class BayesianKnowledgeTracing:
    """贝叶斯知识追踪算法实现（优化版）"""
    
    def __init__(self, default_params: Optional[BKTParameters] = None,
                 storage_path: str = "./data/student_states.json"):
        """
        初始化BKT算法
        
        Args:
            default_params: 默认参数
            storage_path: 学生状态存储路径
        """
        self.default_params = default_params or BKTParameters()
        self.storage_path = Path(storage_path)
        self.student_states: Dict[str, Dict[str, StudentState]] = defaultdict(dict)
        
        # 加载已有的学生状态
        self._load_states()
        
        logger.info(f"✅ BKT算法初始化完成（持久化模式，存储路径: {storage_path}）")
    
    def _load_states(self):
        """从文件加载学生状态"""
        if not self.storage_path.exists():
            logger.info("📂 学生状态文件不存在，创建新文件")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 反序列化
            for student_id, knowledge_points in data.items():
                for kp, state_dict in knowledge_points.items():
                    # 重建 BKTParameters 对象
                    params_dict = state_dict.get('params', {})
                    params = BKTParameters(**params_dict)
                    
                    # 重建 StudentState 对象
                    state = StudentState(
                        student_id=state_dict['student_id'],
                        knowledge_point=state_dict['knowledge_point'],
                        mastery_prob=state_dict['mastery_prob'],
                        answer_history=state_dict['answer_history'],
                        recent_performance=state_dict['recent_performance'],
                        params=params,
                        created_at=state_dict.get('created_at', ''),
                        updated_at=state_dict.get('updated_at', '')
                    )
                    
                    self.student_states[student_id][kp] = state
            
            total_students = len(self.student_states)
            total_records = sum(len(kps) for kps in self.student_states.values())
            logger.info(f"✅ 加载学生状态成功: {total_students} 个学生, {total_records} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 加载学生状态失败: {e}")
            self.student_states = defaultdict(dict)
    
    def _save_states(self):
        """保存学生状态到文件"""
        try:
            # 确保目录存在
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 序列化
            data = {}
            for student_id, knowledge_points in self.student_states.items():
                data[student_id] = {}
                for kp, state in knowledge_points.items():
                    state_dict = {
                        'student_id': state.student_id,
                        'knowledge_point': state.knowledge_point,
                        'mastery_prob': state.mastery_prob,
                        'answer_history': state.answer_history,
                        'recent_performance': state.recent_performance,
                        'params': asdict(state.params),
                        'created_at': state.created_at,
                        'updated_at': state.updated_at
                    }
                    data[student_id][kp] = state_dict
            
            # 写入文件
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 学生状态已保存")
            
        except Exception as e:
            logger.error(f"❌ 保存学生状态失败: {e}")
    
    def initialize_student(self, student_id: str, knowledge_point: str, 
                          params: Optional[BKTParameters] = None) -> StudentState:
        """
        初始化学生状态
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            params: BKT参数
            
        Returns:
            学生状态
        """
        if params is None:
            # 尝试使用个性化参数
            params = self._get_personalized_params(student_id)
        
        state = StudentState(
            student_id=student_id,
            knowledge_point=knowledge_point,
            mastery_prob=params.p_init,
            answer_history=[],
            recent_performance=[],
            params=params
        )
        
        self.student_states[student_id][knowledge_point] = state
        self._save_states()
        
        logger.info(f"🆕 初始化学生 {student_id} 在知识点 {knowledge_point} 的状态 (初始掌握度: {params.p_init:.3f})")
        return state
    
    def _get_personalized_params(self, student_id: str) -> BKTParameters:
        """
        获取学生的个性化BKT参数
        
        Args:
            student_id: 学生ID
            
        Returns:
            个性化的BKT参数
        """
        if student_id not in self.student_states:
            return self.default_params
        
        # 收集学生所有答题历史
        all_history = []
        for state in self.student_states[student_id].values():
            all_history.extend(state.answer_history)
        
        if len(all_history) < 10:  # 数据不足，使用默认参数
            return self.default_params
        
        # 分析学生特征
        total = len(all_history)
        correct = sum(1 for r in all_history if r.get('is_correct', False))
        accuracy = correct / total
        
        # 计算学习速度
        learning_speed = self._calculate_learning_speed_from_history(all_history)
        
        # 个性化参数
        params = BKTParameters()
        
        # 基础能力强的学生：提高初始掌握概率
        if accuracy > 0.8:
            params.p_init = 0.5
            logger.debug(f"👍 学生 {student_id} 基础好，初始掌握概率提升至 0.5")
        elif accuracy > 0.6:
            params.p_init = 0.4
        else:
            params.p_init = 0.2
            logger.debug(f"📚 学生 {student_id} 需要加强基础，初始掌握概率降至 0.2")
        
        # 学习速度快的学生：提高学习概率
        if learning_speed > 0.1:
            params.p_learn = 0.3
            logger.debug(f"🚀 学生 {student_id} 学习速度快，学习概率提升至 0.3")
        elif learning_speed > 0.05:
            params.p_learn = 0.2
        else:
            params.p_learn = 0.15
        
        return params
    
    def _calculate_learning_speed_from_history(self, history: List[Dict[str, Any]]) -> float:
        """从答题历史计算学习速度"""
        if len(history) < 3:
            return 0.0
        
        mastery_changes = []
        for i in range(1, len(history)):
            prev_mastery = history[i-1].get('previous_mastery', 0.3)
            curr_mastery = history[i].get('previous_mastery', 0.3)
            change = curr_mastery - prev_mastery
            mastery_changes.append(change)
        
        if mastery_changes:
            return sum(mastery_changes) / len(mastery_changes)
        return 0.0
    
    def get_student_state(self, student_id: str, knowledge_point: str) -> StudentState:
        """
        获取学生状态，如果不存在则初始化
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            
        Returns:
            学生状态
        """
        if student_id not in self.student_states or knowledge_point not in self.student_states[student_id]:
            return self.initialize_student(student_id, knowledge_point)
        
        return self.student_states[student_id][knowledge_point]
    
    def update_mastery_probability(self, state: StudentState, is_correct: bool) -> float:
        """
        更新掌握概率
        
        Args:
            state: 学生状态
            is_correct: 是否答对
            
        Returns:
            更新后的掌握概率
        """
        p_mastery = state.mastery_prob
        p_learn = state.params.p_learn
        p_forget = state.params.p_forget
        p_guess = state.params.p_guess
        p_slip = state.params.p_slip
        
        # 贝叶斯更新公式
        if is_correct:
            # 答对情况
            numerator = p_mastery * (1 - p_slip)
            denominator = numerator + (1 - p_mastery) * p_guess
            p_mastery_given_correct = numerator / denominator if denominator > 0 else p_mastery
            
            # 考虑学习效应
            p_mastery_updated = p_mastery_given_correct + (1 - p_mastery_given_correct) * p_learn
        else:
            # 答错情况
            numerator = p_mastery * p_slip
            denominator = numerator + (1 - p_mastery) * (1 - p_guess)
            p_mastery_given_incorrect = numerator / denominator if denominator > 0 else p_mastery
            
            # 考虑遗忘效应
            p_mastery_updated = p_mastery_given_incorrect * (1 - p_forget)
        
        # 确保概率在合理范围内
        p_mastery_updated = max(0.01, min(0.99, p_mastery_updated))
        
        state.mastery_prob = p_mastery_updated
        state.updated_at = datetime.now().isoformat()
        
        return p_mastery_updated
    
    def record_answer(self, student_id: str, knowledge_point: str, 
                     question: Dict[str, Any], is_correct: bool) -> Dict[str, Any]:
        """
        记录学生答题并更新状态
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            question: 题目信息
            is_correct: 是否答对
            
        Returns:
            更新后的状态信息
        """
        state = self.get_student_state(student_id, knowledge_point)
        
        # 记录答题历史
        answer_record = {
            'question': question,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat(),
            'difficulty': question.get('难度', '中等'),
            'previous_mastery': state.mastery_prob
        }
        
        # 更新掌握概率
        new_mastery = self.update_mastery_probability(state, is_correct)
        
        # 更新最近表现（保留最近10次）
        state.recent_performance.append(is_correct)
        if len(state.recent_performance) > 10:
            state.recent_performance.pop(0)
        
        state.answer_history.append(answer_record)
        
        # 保存状态
        self._save_states()
        
        result = {
            'student_id': student_id,
            'knowledge_point': knowledge_point,
            'current_mastery': new_mastery,
            'previous_mastery': answer_record['previous_mastery'],
            'mastery_change': new_mastery - answer_record['previous_mastery'],
            'answer_record': answer_record,
            'total_answers': len(state.answer_history),
            'recent_accuracy': self._calculate_recent_accuracy(state),
            'recommended_difficulty': self.get_recommended_difficulty(student_id, knowledge_point)
        }
        
        logger.info(f"📝 学生 {student_id} 在知识点 {knowledge_point} 答题记录: "
                   f"掌握度 {answer_record['previous_mastery']:.3f} → {new_mastery:.3f} "
                   f"({'✓' if is_correct else '✗'})")
        
        return result
    
    def _calculate_recent_accuracy(self, state: StudentState) -> float:
        """计算最近表现准确率"""
        if not state.recent_performance:
            return 0.0
        
        return sum(state.recent_performance) / len(state.recent_performance)
    
    def get_recommended_difficulty(self, student_id: str, knowledge_point: str) -> str:
        """
        根据学生掌握程度推荐题目难度
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            
        Returns:
            推荐难度: "简单", "中等", "困难"
        """
        state = self.get_student_state(student_id, knowledge_point)
        mastery = state.mastery_prob
        recent_accuracy = self._calculate_recent_accuracy(state)
        
        # 综合考虑掌握概率和最近表现
        combined_score = 0.7 * mastery + 0.3 * recent_accuracy
        
        if combined_score < 0.3:
            return "简单"
        elif combined_score < 0.7:
            return "中等"
        else:
            return "困难"
    
    def get_adaptive_question_sequence(self, student_id: str, knowledge_point: str, 
                                     total_questions: int = 10) -> List[str]:
        """
        生成自适应题目序列
        
        Args:
            student_id: 学生ID
            knowledge_point: 知识点
            total_questions: 总题目数
            
        Returns:
            难度序列列表
        """
        state = self.get_student_state(student_id, knowledge_point)
        sequence = []
        
        for i in range(total_questions):
            # 根据当前掌握程度动态调整难度
            current_mastery = state.mastery_prob
            
            if i == 0:
                # 第一题使用推荐难度
                difficulty = self.get_recommended_difficulty(student_id, knowledge_point)
            else:
                # 后续题目根据表现调整
                recent_correct = sum(state.recent_performance[-3:]) if len(state.recent_performance) >= 3 else 0
                
                if recent_correct >= 2:  # 最近3题答对2题以上
                    # 提升难度
                    if current_mastery > 0.7:
                        difficulty = "困难"
                    elif current_mastery > 0.4:
                        difficulty = "中等"
                    else:
                        difficulty = "简单"
                elif recent_correct <= 1:  # 最近3题答对1题或更少
                    # 降低难度
                    if current_mastery < 0.3:
                        difficulty = "简单"
                    elif current_mastery < 0.6:
                        difficulty = "中等"
                    else:
                        difficulty = "困难"
                else:
                    # 保持当前难度
                    difficulty = sequence[-1] if sequence else self.get_recommended_difficulty(student_id, knowledge_point)
            
            sequence.append(difficulty)
        
        return sequence
    
    def generate_student_profile(self, student_id: str) -> Dict[str, Any]:
        """
        生成学生评估画像
        
        Args:
            student_id: 学生ID
            
        Returns:
            学生评估画像
        """
        if student_id not in self.student_states:
            return {
                'student_id': student_id,
                'knowledge_points': {},
                'overall_mastery': 0.0,
                'learning_potential': '未知',
                'weak_points': [],
                'strengths': [],
                'total_practice_time': 0,
                'learning_characteristics': {}
            }
        
        states = self.student_states[student_id]
        knowledge_points = {}
        total_mastery = 0.0
        
        for kp, state in states.items():
            knowledge_points[kp] = {
                'mastery': state.mastery_prob,
                'total_answers': len(state.answer_history),
                'recent_accuracy': self._calculate_recent_accuracy(state),
                'learning_trend': self._calculate_learning_trend(state),
                'created_at': state.created_at,
                'updated_at': state.updated_at
            }
            total_mastery += state.mastery_prob
        
        # 计算整体掌握度
        overall_mastery = total_mastery / len(states) if states else 0.0
        
        # 识别薄弱点和优势点
        weak_points = [kp for kp, data in knowledge_points.items() 
                      if data['mastery'] < 0.4]
        strengths = [kp for kp, data in knowledge_points.items() 
                    if data['mastery'] > 0.8]
        
        # 评估学习潜力
        learning_potential = self._assess_learning_potential(states)
        
        # 学习特征分析
        learning_characteristics = self._analyze_learning_characteristics(student_id)
        
        profile = {
            'student_id': student_id,
            'knowledge_points': knowledge_points,
            'overall_mastery': overall_mastery,
            'learning_potential': learning_potential,
            'weak_points': weak_points,
            'strengths': strengths,
            'total_knowledge_points': len(states),
            'total_answers': sum(len(state.answer_history) for state in states.values()),
            'learning_characteristics': learning_characteristics
        }
        
        return profile
    
    def _calculate_learning_trend(self, state: StudentState) -> str:
        """计算学习趋势"""
        if len(state.answer_history) < 5:
            return "数据不足"
        
        # 分析最近5次答题的掌握度变化
        recent_mastery = [record.get('previous_mastery', 0.3) for record in state.answer_history[-5:]]
        if len(recent_mastery) >= 2:
            trend = recent_mastery[-1] - recent_mastery[0]
            if trend > 0.1:
                return "快速提升"
            elif trend > 0.05:
                return "稳步提升"
            elif trend < -0.1:
                return "明显下降"
            elif trend < -0.05:
                return "轻微下降"
            else:
                return "保持稳定"
        
        return "数据不足"
    
    def _assess_learning_potential(self, states: Dict[str, StudentState]) -> str:
        """评估学习潜力"""
        if not states:
            return "未知"
        
        # 分析学习速度和稳定性
        learning_speeds = []
        for state in states.values():
            if len(state.answer_history) >= 3:
                # 计算平均学习速度
                mastery_changes = []
                for i in range(1, len(state.answer_history)):
                    prev_m = state.answer_history[i-1].get('previous_mastery', 0.3)
                    curr_m = state.answer_history[i].get('previous_mastery', 0.3)
                    change = curr_m - prev_m
                    mastery_changes.append(change)
                
                if mastery_changes:
                    avg_speed = sum(mastery_changes) / len(mastery_changes)
                    learning_speeds.append(avg_speed)
        
        if not learning_speeds:
            return "数据不足"
        
        avg_learning_speed = sum(learning_speeds) / len(learning_speeds)
        
        if avg_learning_speed > 0.08:
            return "学习潜力优秀"
        elif avg_learning_speed > 0.04:
            return "学习潜力良好"
        elif avg_learning_speed > 0.01:
            return "学习潜力一般"
        else:
            return "需要更多关注"
    
    def _analyze_learning_characteristics(self, student_id: str) -> Dict[str, Any]:
        """分析学生学习特征"""
        if student_id not in self.student_states:
            return {}
        
        states = self.student_states[student_id]
        all_history = []
        for state in states.values():
            all_history.extend(state.answer_history)
        
        if not all_history:
            return {}
        
        # 难度偏好分析
        difficulty_stats = {'简单': 0, '中等': 0, '困难': 0}
        correct_by_difficulty = {'简单': 0, '中等': 0, '困难': 0}
        
        for record in all_history:
            diff = record.get('difficulty', '中等')
            if diff in difficulty_stats:
                difficulty_stats[diff] += 1
                if record.get('is_correct', False):
                    correct_by_difficulty[diff] += 1
        
        # 计算各难度准确率
        difficulty_accuracy = {}
        for diff in difficulty_stats:
            if difficulty_stats[diff] > 0:
                difficulty_accuracy[diff] = correct_by_difficulty[diff] / difficulty_stats[diff]
            else:
                difficulty_accuracy[diff] = 0.0
        
        # 学习稳定性
        if len(all_history) >= 5:
            recent_results = [r.get('is_correct', False) for r in all_history[-10:]]
            stability = 1.0 - (sum(1 for i in range(1, len(recent_results)) 
                               if recent_results[i] != recent_results[i-1]) / len(recent_results))
        else:
            stability = 0.5
        
        return {
            'difficulty_preference': max(difficulty_accuracy, key=difficulty_accuracy.get) if difficulty_accuracy else '中等',
            'difficulty_accuracy': difficulty_accuracy,
            'learning_stability': stability,
            'total_practice_count': len(all_history)
        }


def create_bkt_algorithm(params: Optional[BKTParameters] = None,
                        storage_path: str = "./data/student_states.json") -> BayesianKnowledgeTracing:
    """
    创建BKT算法实例
    
    Args:
        params: BKT参数
        storage_path: 存储路径
        
    Returns:
        BKT算法实例
    """
    return BayesianKnowledgeTracing(params, storage_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建BKT算法实例
    bkt = create_bkt_algorithm()
    
    # 模拟学生答题
    test_question = {
        '问题': '1+1=?',
        '答案': '2',
        '难度': '简单',
        '知识点': '代数'
    }
    
    # 记录答题
    print("\n=== 模拟学生答题 ===")
    result1 = bkt.record_answer("student_001", "代数", test_question, True)
    print(f"第1题（答对）：掌握度 {result1['previous_mastery']:.3f} → {result1['current_mastery']:.3f}")
    
    result2 = bkt.record_answer("student_001", "代数", test_question, False)
    print(f"第2题（答错）：掌握度 {result2['previous_mastery']:.3f} → {result2['current_mastery']:.3f}")
    
    result3 = bkt.record_answer("student_001", "代数", test_question, True)
    print(f"第3题（答对）：掌握度 {result3['previous_mastery']:.3f} → {result3['current_mastery']:.3f}")
    
    # 获取推荐难度
    difficulty = bkt.get_recommended_difficulty("student_001", "代数")
    print(f"\n推荐难度: {difficulty}")
    
    # 生成题目序列
    sequence = bkt.get_adaptive_question_sequence("student_001", "代数", 5)
    print(f"自适应题目序列: {sequence}")
    
    # 生成学生画像
    profile = bkt.generate_student_profile("student_001")
    print(f"\n学生画像:")
    print(f"  整体掌握度: {profile['overall_mastery']:.1%}")
    print(f"  学习潜力: {profile['learning_potential']}")
    print(f"  学习特征: {profile['learning_characteristics']}")