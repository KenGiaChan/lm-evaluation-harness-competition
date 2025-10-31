import json
import re
from typing import List, Dict, Any, Tuple

def load_data(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                required_fields = ["question", "answer"]
                if not all(field in item for field in required_fields):
                    raise ValueError(f"缺少必填字段: {required_fields}")
                
                processed_question = item["question"].replace("___", "____").replace("_", "____")
                
                data.append({
                    "id": item.get("id", f"line_{line_num}"),
                    "question": processed_question,
                    "answer": str(item["answer"]).strip(),
                    "subject": item.get("subject", "unknown")  # 保留MMLU的学科信息
                })
            except json.JSONDecodeError:
                print(f"警告：第{line_num}行不是有效的JSON，已跳过")
            except Exception as e:
                print(f"警告：第{line_num}行处理失败 - {str(e)}，已跳过")
    return data

def process_docs(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    for doc in dataset:
        # MMLU原数据可能包含选项，这里将正确答案替换为空白（核心转换逻辑）
        if "choices" in doc and "answer" in doc:
            try:
                # 获取正确答案文本（MMLU的answer通常是选项索引）
                correct_answer = doc["choices"][int(doc["answer"])]
                # 将问题中的正确答案替换为空白符，构建填空题
                question_with_blank = doc["question"].replace(correct_answer, "____")
            except (IndexError, ValueError):
                # 容错：若转换失败，直接使用原问题+空白
                question_with_blank = f"{doc['question']} ____"
        
        processed.append({
            "id": doc.get("id", f"doc_{len(processed)}"),
            "question": question_with_blank,
            "answer": str(doc["answer"]).strip(),
            "subject": doc.get("subject", "unknown")
        })
    return processed

def build_prompt(item: Dict[str, Any], template: List[str]) -> str:
    """构建Prompt（兼容MMLU的subject字段，可用于学科特定提示）"""
    prompt = template[0].format(question=item["question"])
    prompt += "\n" + template[1]
    return prompt

def extract_answer(output: str, pattern: str = r"Answer:\s*(.*?)$") -> str:
    """提取答案（参考minerva_math的答案提取逻辑）"""
    if not output:
        return ""
    
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
    else:
        # 从推理过程中提取最后出现的数字/短语（适配MMLU常见答案类型）
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        extracted = lines[-1] if lines else ""
    
    return extracted

def normalize_answer(answer: str, rules: List[str]) -> str:
    """标准化答案（参考minerva_math的标准化逻辑）"""
    normalized = answer
    for rule in rules:
        if rule.startswith("replace:"):
            chars = rule.split(":")[1].strip()
            normalized = re.sub(f"[{re.escape(chars)}]", "", normalized)
        elif rule == "trim":
            normalized = normalized.strip()
        elif rule == "lowercase":
            normalized = normalized.lower()
        elif rule == "remove_articles":
            normalized = re.sub(r"\b(a|an|the)\b", " ", normalized).strip()
    return normalized

def process_results(doc: Dict[str, Any], results: List[str], normalization_rules: List[str]) -> Dict[str, Any]:
    """
    处理结果（参考minerva_math的process_results）
    返回单样本评估结果和学科信息
    """
    pred = extract_answer(results[0])  # 取第一个结果（单模型推理）
    pred_norm = normalize_answer(pred, normalization_rules)
    ref_norm = normalize_answer(doc["answer"], normalization_rules)
    
    # 支持部分匹配（MMLU答案可能包含解释性文本）
    is_correct = pred_norm == ref_norm or ref_norm in pred_norm
    
    return {
        "id": doc["id"],
        "subject": doc["subject"],
        "prediction": pred,
        "reference": doc["answer"],
        "correct": is_correct
    }

def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    评估总结果（按学科聚合，参考minerva_math）
    """
    # 按学科分组
    subject_results = {}
    for res in results:
        subject = res["subject"]
        if subject not in subject_results:
            subject_results[subject] = {"total": 0, "correct": 0}
        subject_results[subject]["total"] += 1
        if res["correct"]:
            subject_results[subject]["correct"] += 1
    
    # 计算每个学科的准确率
    for subject in subject_results:
        total = subject_results[subject]["total"]
        subject_results[subject]["accuracy"] = round(
            (subject_results[subject]["correct"] / total) * 100, 2
        ) if total > 0 else 0.0
    
    # 计算总体准确率
    total_correct = sum(res["correct"] for res in results)
    total = len(results)
    overall_accuracy = round((total_correct / total) * 100, 2) if total > 0 else 0.0
    
    return {
        "overall": {"accuracy": overall_accuracy, "total": total, "correct": total_correct},
        "per_subject": subject_results
    }

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """保存评估结果（包含学科细分）"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评估结果已保存至: {output_path}")
