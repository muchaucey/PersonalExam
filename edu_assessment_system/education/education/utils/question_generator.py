"""
出题生成模块
使用Qwen模型结合知识图谱生成题目
"""

import json
import logging
import random
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QuestionGenerator:
    
    def __init__(self, llm_model, question_db, config: Dict[str, Any]):

        self.llm_model = llm_model
        self.question_db = question_db
        self.config = config
        
        logger.info("题目生成器初始化完成")
    
    def get_reference_questions(self, knowledge_point: str, 
                               difficulty: str = None,
                               count: int = 3) -> List[Dict[str, Any]]:
        # 先按知识点筛选
        questions = self.question_db.get_questions_by_knowledge(knowledge_point)
        
        # 如果指定难度,进一步筛选
        if difficulty:
            questions = [q for q in questions if q.get('难度') == difficulty]
        
        # 随机选择
        if len(questions) > count:
            questions = random.sample(questions, count)
        
        return questions
    
    def format_reference_examples(self, questions: List[Dict[str, Any]]) -> str:
        if not questions:
            return "无参考示例"
        
        examples = []
        for i, q in enumerate(questions, 1):
            example = f"""
示例{i}:
问题: {q.get('问题', '')}
答案: {q.get('答案', '')}
解析: {q.get('解析', '')}
"""
            examples.append(example.strip())
        
        return "\n\n".join(examples)
    
    def generate_question_prompt(self, knowledge_point: str,
                                difficulty: str,
                                question_type: str,
                                prompt_template: str) -> str:

        # 获取参考题目
        reference_questions = self.get_reference_questions(
            knowledge_point, difficulty, count=2
        )
        reference_text = self.format_reference_examples(reference_questions)
        
        # 填充模板
        prompt = prompt_template.format(
            knowledge_point=knowledge_point,
            difficulty=difficulty,
            question_type=question_type,
            reference_examples=reference_text
        )
        
        return prompt
    
    def parse_generated_question(self, response: str) -> Optional[Dict[str, Any]]:

        try:
            # 尝试找到JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("响应中未找到JSON格式")
                return None
            
            json_str = response[start_idx:end_idx]
            
            # 尝试直接解析
            try:
                question = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"直接JSON解析失败: {e}, 尝试修复格式")
                
                # 尝试修复常见的JSON格式问题
                # 1. 修复属性名缺少双引号的问题
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                
                # 2. 修复单引号问题
                json_str = json_str.replace("'", '"')
                
                # 3. 尝试再次解析
                try:
                    question = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    logger.warning(f"修复后JSON解析失败: {e2}, 尝试提取关键信息")
                    
                    # 如果JSON解析仍然失败，尝试从文本中提取关键信息
                    question = self.extract_question_from_text(response)
                    if not question:
                        return None
            
            # 验证必要字段
            required_fields = ['问题', '答案', '解析', '难度', '知识点']
            for field in required_fields:
                if field not in question:
                    logger.error(f"生成的题目缺少字段: {field}")
                    return None
            
            return question
            
        except Exception as e:
            logger.error(f"解析题目失败: {e}")
            return None
    
    def extract_question_from_text(self, response: str) -> Optional[Dict[str, Any]]:
        """从文本中提取题目信息"""
        try:
            question = {}
            
            # 提取问题
            problem_match = re.search(r'问题[:：]\s*([^\n]+)', response)
            if problem_match:
                question['问题'] = problem_match.group(1).strip()
            
            # 提取答案
            answer_match = re.search(r'答案[:：]\s*([^\n]+)', response)
            if answer_match:
                question['答案'] = answer_match.group(1).strip()
            
            # 提取解析
            analysis_match = re.search(r'解析[:：]\s*([^\n]+)', response)
            if analysis_match:
                question['解析'] = analysis_match.group(1).strip()
            
            # 提取难度
            difficulty_match = re.search(r'难度[:：]\s*([^\n]+)', response)
            if difficulty_match:
                question['难度'] = difficulty_match.group(1).strip()
            
            # 提取知识点
            knowledge_match = re.search(r'知识点[:：]\s*([^\n]+)', response)
            if knowledge_match:
                question['知识点'] = knowledge_match.group(1).strip()
            
            # 检查是否提取到足够的信息
            if len(question) >= 3:  # 至少需要问题、答案、解析
                logger.info("从文本中成功提取题目信息")
                return question
            else:
                logger.error("从文本中提取的题目信息不完整")
                return None
                
        except Exception as e:
            logger.error(f"从文本提取题目信息失败: {e}")
            return None
    
    def generate_single_question(self, knowledge_point: str,
                                difficulty: str,
                                question_type: str,
                                prompt_template: str,
                                max_retries: int = 3) -> Optional[Dict[str, Any]]:

        for attempt in range(max_retries):
            try:
                # 生成提示词
                prompt = self.generate_question_prompt(
                    knowledge_point, difficulty, question_type, prompt_template
                )
                
                # 调用LLM生成
                logger.info(f"正在生成题目 (尝试 {attempt+1}/{max_retries})...")
                response = self.llm_model.generate(prompt)
                
                # 解析响应
                question = self.parse_generated_question(response)
                
                if question:
                    logger.info("题目生成成功")
                    return question
                else:
                    logger.warning(f"题目解析失败,重试...")
                    
            except Exception as e:
                logger.error(f"生成题目出错: {e}")
        
        logger.error(f"生成题目失败,已尝试 {max_retries} 次")
        return None
    
    def generate_question_set(self, knowledge_point: str,
                            count: int,
                            difficulty_distribution: Dict[str, float] = None,
                            prompt_template: str = None) -> List[Dict[str, Any]]:

        if difficulty_distribution is None:
            difficulty_distribution = {'简单': 0.4, '中等': 0.4, '困难': 0.2}
        
        if prompt_template is None:
            from config import PROMPTS
            prompt_template = PROMPTS['question_generation']
        
        # 计算每个难度的题目数量
        difficulty_counts = {}
        remaining = count
        
        for difficulty, ratio in difficulty_distribution.items():
            num = int(count * ratio)
            difficulty_counts[difficulty] = num
            remaining -= num
        
        # 将剩余的分配给第一个难度
        if remaining > 0:
            first_difficulty = list(difficulty_counts.keys())[0]
            difficulty_counts[first_difficulty] += remaining
        
        # 生成题目
        generated_questions = []
        
        for difficulty, num in difficulty_counts.items():
            logger.info(f"正在生成 {num} 道{difficulty}难度的{knowledge_point}题目...")
            
            for i in range(num):
                question = self.generate_single_question(
                    knowledge_point=knowledge_point,
                    difficulty=difficulty,
                    question_type=knowledge_point,
                    prompt_template=prompt_template
                )
                
                if question:
                    generated_questions.append(question)
                    logger.info(f"进度: {len(generated_questions)}/{count}")
        
        logger.info(f"题目生成完成,成功 {len(generated_questions)}/{count} 道")
        return generated_questions
    
    def generate_from_knowledge_point(self, knowledge_point: str) -> List[Dict[str, Any]]:

        count = self.config.get('questions_per_type', 5)
        return self.generate_question_set(knowledge_point, count)


class MockQuestionGenerator:
    
    def __init__(self, llm_model, question_db, config: Dict[str, Any]):
        self.question_db = question_db
        self.config = config
        logger.info("使用模拟题目生成器")
    
    def generate_from_knowledge_point(self, knowledge_point: str) -> List[Dict[str, Any]]:

        count = self.config.get('questions_per_type', 5)
        questions = self.question_db.get_questions_by_knowledge(knowledge_point)
        
        if len(questions) > count:
            questions = random.sample(questions, count)
        
        logger.info(f"从题库抽取 {len(questions)} 道{knowledge_point}题目")
        return questions
    
    def generate_question_set(self, knowledge_point: str, count: int, **kwargs):
        """生成题目集"""
        return self.generate_from_knowledge_point(knowledge_point)[:count]


def create_question_generator(llm_model, question_db, config: Dict[str, Any],
                             use_mock: bool = False):

    if use_mock:
        return MockQuestionGenerator(llm_model, question_db, config)
    return QuestionGenerator(llm_model, question_db, config)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..")
    from config import (PANGU_MODEL_PATH, QUESTION_MODEL_CONFIG,
                       QUESTION_DB, QUESTION_CONFIG, PROMPTS)
    from models import create_llm_model
    from data_management.question_db import create_question_database
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型和数据库
    pangu_model = create_llm_model('pangu', PANGU_MODEL_PATH, QUESTION_MODEL_CONFIG)
    question_db = create_question_database(str(QUESTION_DB))
    
    # 创建生成器
    generator = create_question_generator(
        pangu_model, question_db, QUESTION_CONFIG, use_mock=True
    )
    
    # 生成题目
    questions = generator.generate_from_knowledge_point("代数")
    print(f"生成了 {len(questions)} 道题目")
    for i, q in enumerate(questions, 1):
        print(f"\n题目{i}: {q.get('问题', '')}")
