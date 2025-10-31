import json
import re
from typing import List, Dict, Any, Tuple

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据并验证格式，处理空白替换"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                item = json.loads(line)
                required_fields = ["question", "answer"]
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    raise ValueError(f"缺少必填字段: {missing_fields}")
                
                # 统一空白符格式（使用4个下划线，确保一致性）
                processed_question = re.sub(r"_+", "____", item["question"])
                
                data.append({
                    "id": item.get("id", f"line_{line_num}"),
                    "question": processed_question,
                    "answer": str(item["answer"]).strip(),
                    "subject": item.get("subject", "unknown").lower()  # 统一学科名称为小写
                })
            except json.JSONDecodeError:
                print(f"警告：第{line_num}行不是有效的JSON，已跳过")
            except Exception as e:
                print(f"警告：第{line_num}行处理失败 - {str(e)}，已跳过")
    return data

def process_docs(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理文档，将原问题转换为填空题（核心逻辑）"""
    processed = []
    for doc in dataset:
        question = doc["question"]
        answer = doc["answer"]
        
        # 处理包含选项的情况（MMLU原始数据格式）
        if "choices" in doc:
            try:
                # 支持答案为索引（int）或直接文本（str）
                if answer.isdigit():
                    answer_idx = int(answer)
                    if 0 <= answer_idx < len(doc["choices"]):
                        correct_answer = doc["choices"][answer_idx]
                    else:
                        raise IndexError("答案索引超出选项范围")
                else:
                    correct_answer = answer  # 直接使用答案文本
                
                # 替换问题中的正确答案为空白（避免部分匹配，使用精确匹配）
                question_with_blank = question.replace(correct_answer, "____")
                # 若未替换成功（可能答案不在问题中），在句尾添加空白
                if question_with_blank == question:
                    question_with_blank = f"{question} ____"
            except (IndexError, ValueError, TypeError):
                # 容错处理：保留原问题并添加空白
                question_with_blank = f"{question} ____"
        else:
            # 无选项时直接在句尾添加空白
            question_with_blank = f"{question} ____"
        
        processed.append({
            "id": doc.get("id", f"doc_{len(processed)}"),
            "question": question_with_blank,
            "answer": answer.strip(),
            "subject": doc.get("subject", "unknown").lower()
        })
    return processed

def build_prompt(item: Dict[str, Any], template: List[str]) -> str:
    """构建提示词，支持学科特定模板"""
    if not template or len(template) < 2:
        raise ValueError("提示模板格式错误，需包含至少两个元素")
    prompt = template[0].format(question=item["question"])
    prompt += "\n" + template[1]
    return prompt

def extract_answer(output: str) -> str:
    """提取答案，支持多种常见格式"""
    if not output:
        return ""
    
    # 优先匹配明确的答案标记
    patterns = [
        r"Answer:\s*(.*?)(?=\n|$)",
        r"The answer is\s*(.*?)(?=\n|$)",
        r"Final answer:\s*(.*?)(?=\n|$)",
        r"Fill in the blank:\s*(.*?)(?=\n|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    
    # 若未找到标记，提取最后非空行
    lines = [line.strip() for line in output.split("\n") if line.strip()]
    return lines[-1] if lines else ""

def normalize_answer(answer: str, rules: List[str] = None) -> str:
    """标准化答案，默认处理标点、大小写和空白"""
    if rules is None:
        # 默认规则：移除标点、小写、去空白
        rules = [
            "replace:.,;!?()[]{}'",
            "trim",
            "lowercase",
            "remove_articles"
        ]
    
    normalized = answer
    for rule in rules:
        if rule.startswith("replace:"):
            chars = rule.split(":", 1)[1].strip()
            normalized = re.sub(f"[{re.escape(chars)}]", "", normalized)
        elif rule == "trim":
            normalized = normalized.strip()
        elif rule == "lowercase":
            normalized = normalized.lower()
        elif rule == "remove_articles":
            normalized = re.sub(r"\b(a|an|the)\b", " ", normalized).strip()
            normalized = re.sub(r"\s+", " ", normalized)  # 合并空格
    return normalized

def process_results(doc: Dict[str, Any], results: List[str], normalization_rules: List[str] = None) -> Dict[str, Any]:
    """处理推理结果，计算正确性（支持精确匹配和部分匹配）"""
    if not results:
        pred = ""
    else:
        pred = extract_answer(results[0])  # 单模型推理结果
    
    # 标准化答案
    pred_norm = normalize_answer(pred, normalization_rules)
    ref_norm = normalize_answer(doc["answer"], normalization_rules)
    
    # 计算匹配结果
    exact_match = (pred_norm == ref_norm)
    partial_match = (ref_norm in pred_norm) if ref_norm else False
    
    return {
        "id": doc["id"],
        "subject": doc["subject"],
        "prediction": pred,
        "reference": doc["answer"],
        "exact_match": exact_match,
        "partial_match": partial_match
    }

def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """评估结果聚合，按学科和总体计算准确率"""
    # 按学科分组
    subject_results = {}
    for res in results:
        subject = res["subject"]
        if subject not in subject_results:
            subject_results[subject] = {
                "total": 0,
                "exact_match": 0,
                "partial_match": 0
            }
        subject_results[subject]["total"] += 1
        if res["exact_match"]:
            subject_results[subject]["exact_match"] += 1
        if res["partial_match"]:
            subject_results[subject]["partial_match"] += 1
    
    # 计算学科级指标
    for subject in subject_results:
        total = subject_results[subject]["total"]
        subject_results[subject]["exact_accuracy"] = round(
            (subject_results[subject]["exact_match"] / total) * 100, 2
        ) if total > 0 else 0.0
        subject_results[subject]["partial_accuracy"] = round(
            (subject_results[subject]["partial_match"] / total) * 100, 2
        ) if total > 0 else 0.0
    
    # 计算总体指标
    total = len(results)
    total_exact = sum(1 for res in results if res["exact_match"])
    total_partial = sum(1 for res in results if res["partial_match"])
    
    overall = {
        "total": total,
        "exact_match": total_exact,
        "partial_match": total_partial,
        "exact_accuracy": round((total_exact / total) * 100, 2) if total > 0 else 0.0,
        "partial_accuracy": round((total_partial / total) * 100, 2) if total > 0 else 0.0
    }
    
    return {
        "overall": overall,
        "per_subject": subject_results
    }

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """保存评估结果到JSON文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存至: {output_path}")
    except IOError as e:
        print(f"保存结果失败: {str(e)}")
