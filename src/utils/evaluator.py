import ast
import os
import pandas as pd
import json
from pandas.testing import assert_frame_equal
import numpy as np
from typing import Dict, Tuple
import random  # 新增：用于随机选择路径

global_stats = {
    "total_samples": 0,
    "acc_em_correct": 0,
    "acc_pip_correct": 0,
    "cs_sum": 0.0,
}

def read_csv_files(folder_path, folder_name):
    table_dict = {}
    if folder_name == "auto_pipeline":
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.csv') and not file_name.startswith('training'):
                key = os.path.splitext(file_name)[0]
                file_path = os.path.join(folder_path, file_name)
                table_dict[key] = pd.read_csv(file_path).iloc[:, 1:]
    elif folder_name == "buildings":
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.csv') and not file_name.startswith('target'):
                file_path = os.path.join(folder_path, file_name)
                table_dict['test_0'] = pd.read_csv(file_path)
    return table_dict
            
def calculate_acc_pip(result, target, rtol=1e-5, atol=1e-8):
    """Pipeline-level Accuracy (Acc_Pip), following the Auto-Pipeline benchmark's
    num-succ-synthesized / P: 1.0 iff the program produced a non-empty table whose
    output schema covers the target schema (a 'successfully synthesized' pipeline)."""
    # if result.empty or target.empty:
    #     return 0.0
    # return 1.0 if set(target.columns).issubset(set(result.columns)) else 0.0
    
    if result.empty or target.empty:
        return 0.0
    common_cols = list(set(result.columns) & set(target.columns))
    if not common_cols:
        return 0.0
    col_ratio = len(common_cols) / len(target.columns)
    
    result_common = result[common_cols].reset_index(drop=True)
    target_common = target[common_cols].reset_index(drop=True)
    sort_col = common_cols[0]
    result_sorted = result_common.sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
    target_sorted = target_common.sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
    min_len = min(len(result_sorted), len(target_sorted))
    result_sorted = result_sorted.iloc[:min_len].reset_index(drop=True)
    result_sorted = target_sorted.iloc[:min_len].reset_index(drop=True)
    col_ratio = col_ratio * min_len / max(len(result_sorted), len(target_sorted))
    try:
        assert_frame_equal(result_sorted, target_sorted, 
                            check_exact=False, 
                            rtol=rtol, atol=atol,
                            check_dtype=False)
        return col_ratio
    except AssertionError:
        numeric_cols = result_sorted.select_dtypes(include=[np.number]).columns.tolist()
        str_cols = result_sorted.select_dtypes(include=['object']).columns.tolist()
        numeric_mask = np.isclose(
            result_sorted[numeric_cols], 
            target_sorted[numeric_cols], 
            rtol=rtol, 
            atol=atol, 
            equal_nan=True
        )
        def compare_str_cols(a, b):
            return (a.isna() & b.isna()) | (a == b)
        
        str_mask = compare_str_cols(
            result_sorted[str_cols],
            target_sorted[str_cols]
        ).to_numpy()
        combined_mask = np.concatenate([numeric_mask, str_mask], axis=1)
        similarity = np.mean(combined_mask)
        return similarity


def calculate_acc_em(result, target, rtol=1e-5, atol=1e-8):
    """Exact Match Accuracy (Acc_EM): 1.0 iff result is identical to target in both
    schema and data instances. Row order normalized by sorting on the first column;
    numerics use float tolerance."""
    if result.empty or target.empty:
        return 0.0
    if set(result.columns) != set(target.columns):
        return 0.0
    if len(result) != len(target):
        return 0.0
    cols = list(target.columns)
    sort_col = cols[0]
    r = result[cols].sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
    t = target[cols].sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
    try:
        assert_frame_equal(r, t, check_exact=False, rtol=rtol, atol=atol, check_dtype=False)
        return 1.0
    except AssertionError:
        return 0.0


def calculate_column_similarity(result, target):
    """Column Similarity (CS): proportion of target column names that appear in result."""
    if result.empty or target.empty:
        return 0.0
    common_cols = list(set(result.columns) & set(target.columns))
    return len(common_cols) / len(target.columns)
        
def extract_last_variable(code_str):
    tree = ast.parse(code_str)
    last_var = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # 遍历赋值目标，取最后一个有效的变量名
            for target in reversed(node.targets):
                if isinstance(target, ast.Name):
                    last_var = target.id
                    break  # 取最右侧的变量名（如多赋值时）
    return last_var  

def get_output_var(code_lines):
    """获取最后一个赋值语句的变量名"""
    if not code_lines:
        return None
    last_line = code_lines[-1]
    if '=' in last_line:
        var_part = last_line.split('=', 1)[0].strip()
        return var_part
    else:
        return None
def process_json_files(
    json_folder: str, 
    folder_name: str,
    data_folder: str, 
    output_base: str, 
    length_type: int,
    start_num: int,
    end_num: int
) -> Dict:
    """处理指定length_type的所有JSON文件"""
    global global_stats
    total_samples = 0
    acc_em_correct = 0
    acc_pip_correct = 0
    cs_sum = 0.0
    error_files = []

    for num in range(start_num, end_num):
        if folder_name == "auto_pipeline":
            file_id = f"length{length_type}_{num}"  # 新增：记录文件标识符
            target_file = os.path.join(data_folder, file_id, "target.csv")
            folder_path = os.path.join(data_folder, file_id)
            output_path = os.path.join(output_base, f"length{length_type}", "tables")
            json_file = f"{file_id}.json"
            os.makedirs(output_path, exist_ok=True)
        elif folder_name == "buildings":
            file_id = f"group{length_type}_{num}"  # 新增：记录文件标识符
            target_file = os.path.join(data_folder, file_id, f"target{length_type}_{num}.csv")
            folder_path = os.path.join(data_folder, file_id)
            output_path = os.path.join(output_base, f"group{length_type}", "tables")
            json_file = f"{file_id}.json"
            os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(target_file):
            continue
        json_path = os.path.join(json_folder, json_file)
        
        table_dict = read_csv_files(folder_path, folder_name)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            total_samples += 1
            error_files.append({"file_id": file_id, "error": str(e)})  # 新增：记录错误信息
            continue
        paths = data 
        
        path = paths[0]
        acc_em = 0.0
        acc_pip = 0.0
        cs = 0.0
        exec_env = {'pd': pd, **table_dict}
        try:
            last_var = None
            code_str = None
            for code_line in path:
                code_str = code_line
                exec(code_str, exec_env)
                last_var = extract_last_variable(code_str)

            result = exec_env.get(last_var, pd.DataFrame())
            if folder_name == "auto_pipeline":
                target = pd.read_csv(target_file).iloc[:, 1:]
            else:
                target = pd.read_csv(target_file)
            acc_em  = calculate_acc_em(result, target)
            acc_pip = calculate_acc_pip(result, target)
            cs      = calculate_column_similarity(result, target)
        except Exception as e:
            error_files.append({"file_id": file_id, "error": str(e)})

        acc_em_correct  += int(acc_em == 1.0)
        acc_pip_correct += int(acc_pip == 1.0)
        cs_sum          += cs
        total_samples   += 1

    global_stats["total_samples"]    += total_samples
    global_stats["acc_em_correct"]   += acc_em_correct
    global_stats["acc_pip_correct"]  += acc_pip_correct
    global_stats["cs_sum"]           += cs_sum

    if error_files:
        error_log_path = os.path.join(output_base, f"errors_length{length_type}.json")
        with open(error_log_path, 'w') as f:
            json.dump(error_files, f, indent=4)

    return {
        "metrics": {
            "Acc_EM":            acc_em_correct  / total_samples if total_samples else 0.0,
            "Acc_Pip":           acc_pip_correct / total_samples if total_samples else 0.0,
            "column_similarity": cs_sum          / total_samples if total_samples else 0.0,
            "total_samples":     total_samples,
            "acc_em_correct":    acc_em_correct,
            "acc_pip_correct":   acc_pip_correct,
        }
    }

def main(json_folder, data_folder, output_base, length_types, start_num, end_num):
    global global_stats
    folder_name = os.path.basename(data_folder)
    if folder_name not in ["auto_pipeline", "buildings"]:
            raise ValueError(f"Unsupported folder name: {folder_name}")
        
    for length_type in length_types:
        if folder_name == "auto_pipeline":
            json_dir = os.path.join(json_folder, f"length{length_type}")
            output_dir = os.path.join(output_base, f"length{length_type}")
        elif folder_name == "buildings":
            json_dir = os.path.join(json_folder, f"group{length_type}")
            output_dir = os.path.join(output_base, f"group{length_type}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行处理
        result_data = process_json_files(
            json_folder=json_dir,
            folder_name=folder_name,
            data_folder=data_folder,
            output_base=output_base,
            length_type=length_type,
            start_num=start_num,
            end_num=end_num
        )
        
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(result_data["metrics"], f, indent=4)
        m = result_data["metrics"]
        print(f"Length Type {length_type}  "
              f"Acc_EM: {m['Acc_EM']:.4f}  "
              f"Acc_Pip: {m['Acc_Pip']:.4f}  "
              f"CS: {m['column_similarity']:.4f}")

    n = global_stats["total_samples"]
    global_metrics = {
        "Acc_EM":            global_stats["acc_em_correct"]  / n if n else 0.0,
        "Acc_Pip":           global_stats["acc_pip_correct"] / n if n else 0.0,
        "column_similarity": global_stats["cs_sum"]          / n if n else 0.0,
        "total_samples":     n,
        "acc_em_correct":    global_stats["acc_em_correct"],
        "acc_pip_correct":   global_stats["acc_pip_correct"],
    }
    with open(os.path.join(output_base, 'global_metrics.json'), 'w') as f:
        json.dump(global_metrics, f, indent=4)
    print(f"Global  "
          f"Acc_EM: {global_metrics['Acc_EM']:.4f}  "
          f"Acc_Pip: {global_metrics['Acc_Pip']:.4f}  "
          f"CS: {global_metrics['column_similarity']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_folder', type=str, default='./result/auto_pipeline')
    parser.add_argument('--data_folder', type=str, default='./data/auto_pipeline')
    parser.add_argument('--output_base', type=str, default='./predict/auto_pipeline')
    parser.add_argument('--length_types', type=int, nargs='+', default=[6])
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=100)
    args = parser.parse_args()
    main(
        json_folder=args.json_folder,
        data_folder=args.data_folder,
        output_base=args.output_base,
        length_types=args.length_types,
        start_num=args.start_num,
        end_num=args.end_num
    )