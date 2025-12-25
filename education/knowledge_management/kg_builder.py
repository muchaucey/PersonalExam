# -*- coding: utf-8 -*-
"""
知识图谱构建器
使用盘古7B分析题目，构建全局知识图谱
"""

import logging
import json
import re
import networkx as nx
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    
    def __init__(self, llm_model, cache_path: str = "./data/knowledge_graph.pkl"):
        self.llm_model = llm_model
        self.cache_path = Path(cache_path)
        self.graph = nx.DiGraph()  # 有向图
        
        logger.info("初始化完成")
    
    def build_from_questions(self, questions: List[Dict[str, Any]], 
                           force_rebuild: bool = False) -> nx.DiGraph:
        if not force_rebuild and self.cache_path.exists():
            logger.info("从缓存加载知识图谱...")
            return self._load_from_cache()
        
        logger.info(f"开始构建知识图谱（共 {len(questions)} 道题）...")
        
        if not self.llm_model.is_loaded:
            logger.info("加载盘古7B模型...")
            self.llm_model.load_model()
        
        self._add_basic_nodes(questions)
        
        self._extract_relations_with_llm(questions)
        
        self._save_to_cache()
        
        logger.info(f"知识图谱构建完成: {self.graph.number_of_nodes()} 个节点, "
                   f"{self.graph.number_of_edges()} 条边")
        
        return self.graph
    
    def _add_basic_nodes(self, questions: List[Dict[str, Any]]):
        logger.info("添加基础节点...")
        
        for q in questions:
            q_id = q.get('题号')
            major = q.get('知识点大类', q.get('knowledge_point_major', '未知'))
            minor = q.get('知识点小类', q.get('knowledge_point_minor', '未知'))
            difficulty = q.get('难度', 0.5)
            
            self.graph.add_node(
                f"Q{q_id}",
                type='question',
                data=q,
                major_point=major,
                minor_point=minor,
                difficulty=difficulty
            )
            
            major_node = f"KP_Major:{major}"
            minor_node = f"KP_Minor:{major}/{minor}"
            
            if not self.graph.has_node(major_node):
                self.graph.add_node(major_node, type='major_point', name=major)
            
            if not self.graph.has_node(minor_node):
                self.graph.add_node(minor_node, type='minor_point', 
                                  major=major, minor=minor, name=f"{major}/{minor}")
            
            self.graph.add_edge(minor_node, major_node, relation='belongs_to')
            self.graph.add_edge(f"Q{q_id}", minor_node, relation='tests', 
                              difficulty=difficulty)
        
        logger.info(f"基础节点添加完成: {self.graph.number_of_nodes()} 个节点")
    
    def _extract_relations_with_llm(self, questions: List[Dict[str, Any]]):
        logger.info("使用盘古7B分析题目，提取知识关系...")
        
        kp_groups = {}
        for q in questions:
            major = q.get('知识点大类', q.get('knowledge_point_major', '未知'))
            minor = q.get('知识点小类', q.get('knowledge_point_minor', '未知'))
            key = f"{major}/{minor}"
            if key not in kp_groups:
                kp_groups[key] = []
            kp_groups[key].append(q)
        
        total_groups = len(kp_groups)
        for idx, (kp_key, kp_questions) in enumerate(kp_groups.items(), 1):
            logger.info(f"分析知识点 {idx}/{total_groups}: {kp_key}")
            
            sample_questions = kp_questions[:3]
            context = self._build_context(sample_questions)
            
            relations = self._call_llm_for_relations(kp_key, context)
            
            self._add_relations_to_graph(kp_key, relations)
        
        logger.info(f"知识关系提取完成: {self.graph.number_of_edges()} 条边")
    
    def _build_context(self, questions: List[Dict[str, Any]]) -> str:
        context_parts = []
        for q in questions:
            context_parts.append(f"""
题目: {q.get('问题', '')[:100]}
答案: {q.get('答案', '')[:50]}
解析: {q.get('解析', '')[:100]}
""")
        return "\n".join(context_parts)
    
    def _call_llm_for_relations(self, kp_key: str, context: str) -> Dict[str, Any]:
        """调用盘古7B提取知识关系"""
        prompt = f"""分析以下数学知识点的题目，识别关键概念和它们之间的关系。

知识点: {kp_key}

题目示例:
{context}

请提取：
1. 核心概念（3-5个关键词）
2. 前置知识（学习此知识点前需要掌握的）
3. 后续知识（掌握此知识点后可以学习的）
4. 解题方法（常用的方法或技巧）

严格按照以下JSON格式输出：
{{
  "concepts": ["概念1", "概念2", "概念3"],
  "prerequisites": ["前置知识1", "前置知识2"],
  "next_topics": ["后续知识1", "后续知识2"],
  "methods": ["方法1", "方法2"]
}}

只输出JSON，不要有任何额外文字。"""
        
        try:
            response = self.llm_model.generate(
                prompt,
                temperature=0.1,
                max_length=512,
                enable_thinking=False
            )

            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.warning(f"LLM分析失败: {e}")
            return {'concepts': [], 'prerequisites': [], 'next_topics': [], 'methods': []}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析盘古7B的JSON响应"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}")
        
        # 降级：正则提取
        result = {
            'concepts': self._extract_list(response, r'概念[:：]([^\n]+)'),
            'prerequisites': self._extract_list(response, r'前置[:：]([^\n]+)'),
            'next_topics': self._extract_list(response, r'后续[:：]([^\n]+)'),
            'methods': self._extract_list(response, r'方法[:：]([^\n]+)')
        }
        return result
    
    def _extract_list(self, text: str, pattern: str) -> List[str]:
        """从文本中提取列表"""
        matches = re.findall(pattern, text)
        if matches:
            items = matches[0].split('、')
            return [item.strip() for item in items if item.strip()]
        return []
    
    def _add_relations_to_graph(self, kp_key: str, relations: Dict[str, Any]):
        """将提取的关系添加到图谱"""
        minor_node = f"KP_Minor:{kp_key}"
        
        # 添加概念节点
        for concept in relations.get('concepts', []):
            concept_node = f"Concept:{concept}"
            if not self.graph.has_node(concept_node):
                self.graph.add_node(concept_node, type='concept', name=concept)
            self.graph.add_edge(minor_node, concept_node, relation='involves')
        
        # 添加前置知识关系
        for prereq in relations.get('prerequisites', []):
            prereq_node = f"KP_Minor:{prereq}"
            if self.graph.has_node(prereq_node):
                self.graph.add_edge(prereq_node, minor_node, relation='prerequisite')
        
        # 添加后续知识关系
        for next_topic in relations.get('next_topics', []):
            next_node = f"KP_Minor:{next_topic}"
            if self.graph.has_node(next_node):
                self.graph.add_edge(minor_node, next_node, relation='leads_to')
        
        # 添加方法节点
        for method in relations.get('methods', []):
            method_node = f"Method:{method}"
            if not self.graph.has_node(method_node):
                self.graph.add_node(method_node, type='method', name=method)
            self.graph.add_edge(minor_node, method_node, relation='uses')
    
    def _save_to_cache(self):
        """保存图谱到缓存"""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"知识图谱已缓存到: {self.cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def _load_from_cache(self) -> nx.DiGraph:
        """从缓存加载图谱"""
        try:
            with open(self.cache_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"从缓存加载知识图谱: {self.graph.number_of_nodes()} 个节点, "
                       f"{self.graph.number_of_edges()} 条边")
            return self.graph
        except Exception as e:
            logger.error(f"缓存加载失败: {e}")
            raise
    
    def get_graph(self) -> nx.DiGraph:
        """获取知识图谱"""
        return self.graph


def create_kg_builder(llm_model, cache_path: str = "./data/knowledge_graph.pkl"):
    """创建知识图谱构建器"""
    return KnowledgeGraphBuilder(llm_model, cache_path)