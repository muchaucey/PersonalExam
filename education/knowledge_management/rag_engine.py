# -*- coding: utf-8 -*-
"""
基于知识图谱的RAG引擎
使用知识图谱进行智能题目检索
"""

import logging
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random

logger = logging.getLogger(__name__)


class KnowledgeGraphRAG:

    
    def __init__(self, knowledge_graph: nx.DiGraph, embedding_model):
        """
        Args:
            knowledge_graph: 知识图谱（NetworkX图）
            embedding_model: 嵌入模型（用于相似度计算）
        """
        self.graph = knowledge_graph
        self.embedding_model = embedding_model
        
        logger.info("✅ 知识图谱RAG引擎初始化完成")
    
    def search_questions_for_student(self,
                                    student_id: str,
                                    major_point: str,
                                    minor_point: str,
                                    student_mastery: float,
                                    used_question_ids: set,
                                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        为学生检索最适合的题目
        
        Args:
            student_id: 学生ID
            major_point: 目标知识点大类
            minor_point: 目标知识点小类
            student_mastery: 学生掌握度
            used_question_ids: 已使用的题目ID
            top_k: 返回题目数量
            
        Returns:
            题目列表，按适合度排序
        """
        logger.info(f"🔍 为学生 {student_id} 检索题目: {major_point}/{minor_point}, "
                   f"掌握度 {student_mastery:.3f}")
        
        target_kp = f"KP_Minor:{major_point}/{minor_point}"
        
        if not self.graph.has_node(target_kp):
            logger.warning(f"⚠️  知识点 {target_kp} 不在图谱中")
            return []
        
        candidates = self._get_candidate_questions(
            target_kp, student_mastery, used_question_ids
        )
        
        if not candidates:
            logger.warning("⚠️  未找到候选题目")
            return []
        
        scored_questions = []
        for q_node, q_data in candidates:
            score = self._calculate_question_score(
                q_node, q_data, target_kp, student_mastery
            )
            scored_questions.append({
                'question': q_data,
                'node': q_node,
                'score': score
            })
        
        # 3. 排序并返回top_k
        scored_questions.sort(key=lambda x: x['score'], reverse=True)
        results = scored_questions[:top_k]
        
        logger.info(f"检索到 {len(results)} 道题目")
        return results
    
    def _get_candidate_questions(self,
                                target_kp: str,
                                student_mastery: float,
                                used_question_ids: set) -> List[Tuple[str, Dict]]:
        candidates = []
        
        # 策略1：直接相关的题目
        if self.graph.has_node(target_kp):
            for neighbor in self.graph.predecessors(target_kp):
                if neighbor.startswith('Q') and self.graph.nodes[neighbor]['type'] == 'question':
                    q_data = self.graph.nodes[neighbor]['data']
                    if q_data.get('题号') not in used_question_ids:
                        candidates.append((neighbor, q_data))
        
        # 策略2：如果直接题目不够，扩展到相关知识点
        if len(candidates) < 5:
            related_kps = self._get_related_knowledge_points(target_kp, student_mastery)
            for related_kp in related_kps:
                if self.graph.has_node(related_kp):
                    for neighbor in self.graph.predecessors(related_kp):
                        if neighbor.startswith('Q') and self.graph.nodes[neighbor]['type'] == 'question':
                            q_data = self.graph.nodes[neighbor]['data']
                            if q_data.get('题号') not in used_question_ids:
                                if (neighbor, q_data) not in candidates:
                                    candidates.append((neighbor, q_data))
        
        logger.info(f"📊 找到 {len(candidates)} 个候选题目")
        return candidates
    
    def _get_related_knowledge_points(self, target_kp: str, 
                                     student_mastery: float) -> List[str]:
        """获取相关知识点"""
        related = []
        
        if not self.graph.has_node(target_kp):
            return related
        
        # 根据掌握度决定扩展方向
        if student_mastery < 0.4:
            # 掌握度低，寻找前置知识点
            for pred in self.graph.predecessors(target_kp):
                if self.graph[pred][target_kp].get('relation') == 'prerequisite':
                    related.append(pred)
        elif student_mastery > 0.7:
            # 掌握度高，寻找后续知识点
            for succ in self.graph.successors(target_kp):
                if self.graph[target_kp][succ].get('relation') == 'leads_to':
                    related.append(succ)
        else:
            # 中等掌握度，寻找同层级知识点
            major_node = None
            for succ in self.graph.successors(target_kp):
                if self.graph.nodes[succ]['type'] == 'major_point':
                    major_node = succ
                    break
            
            if major_node:
                for pred in self.graph.predecessors(major_node):
                    if self.graph.nodes[pred]['type'] == 'minor_point' and pred != target_kp:
                        related.append(pred)
        
        return related[:3]  # 最多3个相关知识点
    
    def _calculate_question_score(self,
                                  q_node: str,
                                  q_data: Dict,
                                  target_kp: str,
                                  student_mastery: float) -> float:
        """
        计算题目得分（综合多个因素）
        
        评分维度：
        1. 难度匹配度 (40%)
        2. 知识点相关度 (30%)
        3. 学习路径契合度 (20%)
        4. 多样性 (10%)
        """
        # 1. 难度匹配度
        q_difficulty = q_data.get('难度', 0.5)
        target_difficulty = self._get_target_difficulty(student_mastery)
        difficulty_score = 1.0 - abs(q_difficulty - target_difficulty)
        
        # 2. 知识点相关度（基于图结构）
        relevance_score = self._calculate_relevance(q_node, target_kp)
        
        # 3. 学习路径契合度
        path_score = self._calculate_path_fitness(q_node, target_kp, student_mastery)
        
        # 4. 多样性（随机因子）
        diversity_score = random.random()
        
        # 综合得分
        total_score = (
            0.4 * difficulty_score +
            0.3 * relevance_score +
            0.2 * path_score +
            0.1 * diversity_score
        )
        
        return total_score
    
    def _get_target_difficulty(self, student_mastery: float) -> float:
        """根据掌握度确定目标难度"""
        if student_mastery < 0.3:
            return 0.25  # 简单
        elif student_mastery < 0.7:
            return 0.50  # 中等
        else:
            return 0.75  # 困难
    
    def _calculate_relevance(self, q_node: str, target_kp: str) -> float:
        """计算题目与目标知识点的相关度"""
        try:
            # 使用最短路径长度作为相关度
            if nx.has_path(self.graph, q_node, target_kp):
                path_length = nx.shortest_path_length(self.graph, q_node, target_kp)
                # 路径越短越相关
                return 1.0 / (path_length + 1)
            else:
                return 0.5  # 无直接路径，给中等分
        except:
            return 0.5
    
    def _calculate_path_fitness(self, q_node: str, target_kp: str, 
                               student_mastery: float) -> float:
        """计算题目在学习路径上的契合度"""
        # 简化版：基于题目涉及的概念数量
        concept_count = 0
        for neighbor in self.graph.neighbors(q_node):
            if self.graph.nodes.get(neighbor, {}).get('type') == 'concept':
                concept_count += 1
        
        # 概念数量适中得分高
        if 2 <= concept_count <= 4:
            return 1.0
        elif concept_count == 1 or concept_count == 5:
            return 0.7
        else:
            return 0.4
    
    def get_knowledge_subgraph(self, 
                              major_point: str,
                              minor_point: str,
                              depth: int = 2) -> Dict[str, Any]:
        """
        获取知识子图（用于可视化或分析）
        
        Args:
            major_point: 知识点大类
            minor_point: 知识点小类
            depth: 扩展深度
            
        Returns:
            子图数据
        """
        target_kp = f"KP_Minor:{major_point}/{minor_point}"
        
        if not self.graph.has_node(target_kp):
            return {'nodes': [], 'edges': []}
        
        # 使用BFS获取子图
        subgraph_nodes = set([target_kp])
        current_level = {target_kp}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # 添加后继节点
                for succ in self.graph.successors(node):
                    subgraph_nodes.add(succ)
                    next_level.add(succ)
                # 添加前驱节点
                for pred in self.graph.predecessors(node):
                    subgraph_nodes.add(pred)
                    next_level.add(pred)
            current_level = next_level
        
        # 提取子图
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # 转换为可序列化的格式
        nodes_data = []
        for node in subgraph.nodes():
            node_data = {'id': node, **self.graph.nodes[node]}
            # 移除不可序列化的data字段
            if 'data' in node_data:
                del node_data['data']
            nodes_data.append(node_data)
        
        edges_data = []
        for u, v in subgraph.edges():
            edges_data.append({
                'source': u,
                'target': v,
                **self.graph[u][v]
            })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'node_count': len(nodes_data),
            'edge_count': len(edges_data)
        }


def create_kg_rag(knowledge_graph: nx.DiGraph, embedding_model):
    """创建基于知识图谱的RAG引擎"""
    return KnowledgeGraphRAG(knowledge_graph, embedding_model)