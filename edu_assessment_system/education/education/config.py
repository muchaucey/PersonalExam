"""
教育评估系统配置文件
包含所有模型路径、系统参数和配置信息
"""
"""
教育评估系统配置文件
包含所有模型路径、系统参数和配置信息
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
WORKING_DIR = PROJECT_ROOT / "rag_storage"
QUESTION_DB = DATA_DIR / "question_database.json"
KG_GRAPH_PATH = DATA_DIR / "knowledge_graph.html"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
WORKING_DIR.mkdir(exist_ok=True)

# ==================== 模型路径配置 ====================
# Windows本地模型路径 (使用原始路径字符串)
PANGU_MODEL_PATH = "/opt/pangu/openPangu-Embedded-7B-V1.1"  
BGE_M3_MODEL_PATH = "/home/weitianyu/bgem3"  

# ==================== LightRAG配置 ====================
LIGHTRAG_CONFIG = {
    "working_dir": str(WORKING_DIR),
    "embedding_cache_config": {
        "enabled": True,
        "similarity_threshold": 0.95
    }
}

# ==================== 盘古7B模型配置 ====================
# 盘古7B统一配置（用于出题和评估）
PANGU_MODEL_CONFIG = {
    "model_path": PANGU_MODEL_PATH,
    "max_new_tokens": 32768,  # 盘古7B支持最大32768
    "temperature": 0.7,
    "top_p": 0.9,
    "device": "npu",  # 使用NPU加速
    "torch_dtype": "float16",  # NPU推荐使用float16
    "trust_remote_code": True,
    # 盘古7B特殊配置
    "eos_token_id": 45892,  # 盘古7B的结束token
    "system_prompt": "你必须严格遵守法律法规和社会道德规范。生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。",
}

# 出题模型配置 (盘古7B)
QUESTION_MODEL_CONFIG = PANGU_MODEL_CONFIG.copy()
QUESTION_MODEL_CONFIG.update({
    "temperature": 0.8,  # 出题时需要更高的创造性
    "top_p": 0.95,
})

# 评估模型配置 (盘古7B)
EVALUATION_MODEL_CONFIG = PANGU_MODEL_CONFIG.copy()
EVALUATION_MODEL_CONFIG.update({
    "temperature": 0.3,  # 评估时需要更稳定的输出
    "top_p": 0.85,
})

# 嵌入模型配置 (优先使用盘古7B，备用BGE-M3)
EMBEDDING_MODEL_CONFIG = {
    "model_path": BGE_M3_MODEL_PATH,
    "device": "cpu",  # 嵌入模型使用CPU以释放NPU资源
    "batch_size": 32,
    "max_length": 512,
    "use_pangu_embedding": True,  # 优先尝试使用盘古7B的嵌入能力
    "pangu_model_path": PANGU_MODEL_PATH,
}

# ==================== 题型配置 ====================
QUESTION_TYPES = {
    "代数": ["方程", "不等式", "数列", "函数"],
    "几何": ["平面几何", "立体几何", "解析几何"],
    "微积分": ["极限", "导数", "积分"],
    "概率论": ["概率", "统计", "随机变量"]
}

DIFFICULTY_LEVELS = ["简单", "中等", "困难"]

# ==================== 出题配置 ====================
QUESTION_CONFIG = {
    "questions_per_type": 5,  # 每个知识点出题数量
    "min_difficulty": "简单",
    "max_difficulty": "困难",
    "time_limit_per_question": 300,  # 秒
    "enable_thinking": False,  # 是否启用思维链（出题时通常不需要）
}

# ==================== 评估配置 ====================
EVALUATION_CONFIG = {
    "pass_score": 0.6,  # 及格分数
    "excellent_score": 0.85,  # 优秀分数
    "weight_difficulty": {
        "简单": 1.0,
        "中等": 1.5,
        "困难": 2.0
    },
    "enable_thinking": False,  # 评估时不需要显示思维过程
}

# ==================== 可视化配置 ====================
VISUALIZATION_CONFIG = {
    "node_size": 3000,
    "node_color": "lightblue",
    "edge_color": "gray",
    "font_size": 10,
    "figure_size": (15, 10),
    "layout": "spring"  # spring, circular, kamada_kawai
}

# ==================== UI配置 ====================
UI_CONFIG = {
    "title": "智能教育评估对话系统 - 盘古7B驱动（题库出题模式）",
    "theme": "default",
    "port": 7860,
    "share": False,  # 改为 False，不尝试创建公共链接
    "server_name": "0.0.0.0"  # 添加这一行，允许外部访问
}

# ==================== Prompt模板 ====================
PROMPTS = {
    "question_generation": """你是一个专业的数学出题专家。请根据以下信息生成一道高质量的题目：

知识点: {knowledge_point}
难度: {difficulty}
题型: {question_type}

参考示例:
{reference_examples}

要求:
1. 题目应该清晰明确，符合指定的知识点和难度
2. 答案必须准确无误
3. 解析应该详细、易懂，包含完整的解题步骤
4. 题目应该具有一定的教育意义和启发性
5. 格式必须严格按照JSON格式

请严格按照以下JSON格式输出，不要添加任何额外内容或说明：
{{
    "问题": "题目描述",
    "答案": "标准答案",
    "解析": "详细解析步骤",
    "难度": "{difficulty}",
    "知识点": "{knowledge_point}"
}}

注意：
- 请确保JSON格式完全正确，所有字段名使用双引号
- 不要使用单引号
- 不要添加任何JSON之外的文字说明
""",

    "evaluation": """你是一个专业的教育评估专家。请根据学生的答题情况给出全面、专业的综合评价。

学生信息:
- 选择知识点: {knowledge_point}
- 答题数量: {total_questions}
- 正确数量: {correct_count}
- 准确率: {accuracy:.1f}%

答题详情:
{answer_details}

请从以下几个方面进行专业评估:

1. **知识掌握程度评级**
   根据学生表现，给出明确的等级评定（优秀/良好/及格/不及格）

2. **答题表现分析**
   - 分析学生在不同难度题目上的表现
   - 指出答题中的优势和不足

3. **薄弱知识点识别**
   - 明确指出需要加强的具体知识点
   - 分析错误的可能原因

4. **个性化学习建议**
   - 提供具体、可操作的学习建议
   - 推荐适合的学习资源或练习方向
   - 给出下一步学习计划

请用专业、友好的语气给出详细评价，帮助学生明确改进方向。
""",

    "answer_check": """你是一名严谨、专业的数学老师，需要对学生答案进行严格评判。

题目: {question}
标准答案: {correct_answer}
学生答案: {student_answer}
解析: {explanation}

【重要评判标准】:
1. **完整性检查**：学生答案必须包含标准答案的所有关键信息点
   - 如果标准答案有多个部分，学生必须回答所有部分
   - 如果标准答案指明了多个区间、多个值，学生必须全部答出
   - 遗漏任何关键信息都应判定为错误

2. **准确性检查**：学生答案的每个信息点都必须正确
   - 数值必须精确匹配（或在合理误差范围内）
   - 符号、方向、区间必须正确
   - 数学表述必须严谨

3. **等价性判断**：只有当学生答案在数学上完全等价于标准答案时才判正确
   - 不同的表述形式（如分数与小数）是可以的
   - 但信息量不能减少

【评判示例】:
- 标准答案："在 (0,1) 上单调减少，在 (1,+∞) 上单调增加"
  学生答案："单调递增" → 错误（遗漏了单调减少的部分和区间信息）
  学生答案："在 (1,+∞) 上单调增加" → 错误（遗漏了单调减少的部分）
  学生答案："(0,1) 减，(1,+∞) 增" → 正确（信息完整，表述简洁）

- 标准答案："x = 2 或 x = 3"
  学生答案："x = 2" → 错误（遗漏了 x = 3）
  学生答案："2, 3" → 正确（信息完整）

请按照以下格式严格回答：
判定结果: [正确/错误]
理由: [详细说明判定理由，特别要指出学生答案是否完整、准确，如果错误请明确指出遗漏或错误之处]

不要添加任何其他内容。
"""
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": str(PROJECT_ROOT / "logs" / "system.log")
}

# 创建日志目录
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# ==================== 系统信息 ====================
SYSTEM_INFO = {
    "version": "2.0.0",
    "author": "AI Education Team",
    "description": "基于盘古7B、LightRAG、知识图谱和BKT算法的智能教育评估系统（依赖题库出题）",
    "model": "openPanGu-Embedded-7B-V1.1",
    "device": "Ascend 910B NPU"
}

if __name__ == "__main__":
    print("配置文件加载成功!")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"RAG存储目录: {WORKING_DIR}")
    print(f"模型: {SYSTEM_INFO['model']}")
    print(f"设备: {SYSTEM_INFO['device']}")
