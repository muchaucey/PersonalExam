"""
数据管理模块
负责题库数据的管理,包括插入、查询、更新、删除等操作
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)  #创建日志记录


class QuestionDatabase:
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)  # 将字符串路径转换为Path对象
        self.questions = []  # 初始化一个空列表来存储所有题目
        self.load_database()  # 调用load_database方法，从文件加载现有数据
        
        logger.info(f"题库数据库初始化完成,路径: {db_path}")
    
    def load_database(self):  # 从JSON文件加载数据到内存
        if self.db_path.exists():  # 检查文件是否存在
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.questions = json.load(f)   # json.load() 从文件对象f读取JSON数据并转换为Python对象
                logger.info(f"加载题库成功,共 {len(self.questions)} 道题目")
            except Exception as e:
                logger.error(f"加载题库失败: {e}")  # 记录错误
                self.questions = []
        else:
            logger.warning(f"题库文件不存在: {self.db_path}")
            self.questions = []
            self.save_database()
    
    def save_database(self):  # 将内存中的题目数据保存到文件
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)  # 获取文件所在目录的Path对象
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.questions, f, ensure_ascii=False, indent=2)  # 将Python对象转换为JSON写入文件
            logger.info(f"题库已保存,共 {len(self.questions)} 道题目")
        except Exception as e:
            logger.error(f"保存题库失败: {e}")
            raise
    
    def insert_question(self, question: Dict[str, Any]) -> bool:
        try:
            if '题号' not in question:  # 如果不包含，自动生成题号
                question['题号'] = len(self.questions) + 1
            question['创建时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.questions.append(question)  # 将题目字典添加到题目列表中
            self.save_database()  # 保存到文件，确保数据持久化
            
            logger.info(f"插入题目成功,题号: {question['题号']}")
            return True
        except Exception as e:
            logger.error(f"插入题目失败: {e}")
            return False
    
    def insert_questions_batch(self, questions: List[Dict[str, Any]]) -> int:  # 批量插入多个题目
        success_count = 0
        for q in questions:  # 遍历题目列表
            if self.insert_question(q):  # 调用单个插入方法
                success_count += 1  # 插入成功计数加1
        
        logger.info(f"批量插入完成,成功 {success_count}/{len(questions)} 道题目")
        return success_count
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict[str, Any]]:  # 根据题号查找题目

        for q in self.questions:
            if q.get('题号') == question_id:
                return q
        return None
    
    def get_questions_by_knowledge(self, knowledge_point: str) -> List[Dict[str, Any]]:  # 根据知识点筛选题目

        return [q for q in self.questions if q.get('知识点') == knowledge_point]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:  # 多条件筛选题目

        return [q for q in self.questions if q.get('难度') == difficulty]
    
    def get_questions_filtered(self, knowledge_point: Optional[str] = None,
                              difficulty: Optional[str] = None,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:

        filtered = self.questions.copy()
        
        if knowledge_point:
            filtered = [q for q in filtered if q.get('知识点') == knowledge_point]
        
        if difficulty:
            filtered = [q for q in filtered if q.get('难度') == difficulty]
        
        if limit:
            filtered = filtered[:limit]
        
        logger.info(f"筛选结果: {len(filtered)} 道题目")
        return filtered
    
    def update_question(self, question_id: int, updates: Dict[str, Any]) -> bool:  # 更新题目信息
        for i, q in enumerate(self.questions):
            if q.get('题号') == question_id:  # 找到匹配的题目
                q.update(updates)  # 更新题目信息
                q['更新时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.questions[i] = q
                self.save_database()  # 保存更改到文件
                logger.info(f"更新题目成功,题号: {question_id}")
                return True
        
        logger.warning(f"未找到题号 {question_id} 的题目")
        return False
    
    def delete_question(self, question_id: int) -> bool:  # 删除题目

        original_count = len(self.questions)  # 记录删除前的题目数量
        self.questions = [q for q in self.questions if q.get('题号') != question_id]  # 使用列表推导式创建新列表，只包含题号不匹配的题目
        
        if len(self.questions) < original_count:  # 如果删除后数量减少了，说明确实删除了题目
            self.save_database()
            logger.info(f"删除题目成功,题号: {question_id}")
            return True
        
        logger.warning(f"未找到题号 {question_id} 的题目")
        return False
    
    def get_all_questions(self) -> List[Dict[str, Any]]:
        return self.questions.copy()
    
    def get_statistics(self) -> Dict[str, Any]:  # 获取题库统计信息
        stats = {
            "总题目数": len(self.questions),
            "知识点分布": {},
            "难度分布": {}
        }
        
        # 统计知识点分布
        for q in self.questions:
            kp = q.get('知识点', '未分类')  # 获取知识点，如果没有就使用'未分类'
            stats["知识点分布"][kp] = stats["知识点分布"].get(kp, 0) + 1
        
        # 统计难度分布
        for q in self.questions:
            diff = q.get('难度', '未知')
            stats["难度分布"][diff] = stats["难度分布"].get(diff, 0) + 1
        
        return stats
    
    def search_questions(self, keyword: str) -> List[Dict[str, Any]]:  # 全文搜索功能
        results = []  # 存储搜索结果
        keyword_lower = keyword.lower()
        
        for q in self.questions:  # 在问题的各个字段中搜索关键词
            if keyword_lower in q.get('问题', '').lower():
                results.append(q)
                continue  # 找到就添加到结果，跳过当前循环的后续检查
            if keyword_lower in q.get('答案', '').lower():  # 在'答案'字段中搜索
                results.append(q)
                continue
            if keyword_lower in q.get('解析', '').lower():  # 在'解析'字段中搜索
                results.append(q)
                continue
        
        logger.info(f"搜索关键词'{keyword}',找到 {len(results)} 道题目")
        return results
    
    def import_from_json(self, json_path: str) -> int:  # 从JSON文件导入题目
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                new_questions = json.load(f)  # 加载JSON数据
            
            if not isinstance(new_questions, list):  # 检查加载的数据是否是列表类型
                logger.error("JSON文件格式错误,应为题目数组")
                return 0
            
            return self.insert_questions_batch(new_questions)
            
        except Exception as e:
            logger.error(f"导入题目失败: {e}")
            return 0
    
    def export_to_json(self, export_path: str,   # 导出题目到JSON文件
                      knowledge_point: Optional[str] = None,
                      difficulty: Optional[str] = None) -> bool:
        try:
            # 使用筛选功能获取要导出的题目
            questions_to_export = self.get_questions_filtered(
                knowledge_point=knowledge_point,
                difficulty=difficulty
            )
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(questions_to_export, f, ensure_ascii=False, indent=2)
            
            logger.info(f"导出 {len(questions_to_export)} 道题目到 {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出题目失败: {e}")
            return False


def create_question_database(db_path: str) -> QuestionDatabase:
    return QuestionDatabase(db_path)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config import QUESTION_DB
    
    logging.basicConfig(level=logging.INFO)

    db = create_question_database(str(QUESTION_DB))

    test_question = {
        "问题": "求解方程 x^2 - 5x + 6 = 0",
        "答案": "x = 2 或 x = 3",
        "解析": "因式分解: (x-2)(x-3) = 0",
        "难度": "简单",
        "知识点": "代数"
    }
    db.insert_question(test_question)

    stats = db.get_statistics()
    print(f"题库统计: {stats}")
