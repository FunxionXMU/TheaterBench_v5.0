import os
import json
import base64
import re
import time
import subprocess
from tqdm import tqdm
from openai import OpenAI
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置部分 =================

# API配置，按模型提供商分类
API_CONFIGS = {
    "aliyun": {
        "api_key": "sk-7e6b3a62b1f64945a5a4a9347afa5c72",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "siliconflow": {
        "api_key": "sk-izrxbwrnxotwsvcngnnmwivxmyqukrivnjcoszzpscmfasjz",
        "base_url": "https://api.siliconflow.cn/v1"  # 修正：去掉了末尾的 /chat/completions
    }
}

# 配置要测试的模型列表
TEST_MODELS = [
    # 阿里云模型
    "qwen-vl-max-latest",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    # SiliconFlow模型
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "zai-org/GLM-4.6V"
]

# 模型到API提供商的映射
MODEL_TO_API = {
    "qwen3-vl-plus": "aliyun",
    "qwen3-vl-flash": "aliyun",
    "qwen-vl-max-latest": "aliyun",
    "qwen3-vl-30b-a3b-thinking": "aliyun",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "siliconflow",
    "zai-org/GLM-4.6V": "siliconflow",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "siliconflow"
}

VIDEO_DIR = "t2v_videos"

# 最大并发数设置（根据API限制和系统资源调整）
# 设置为模型总数+1，确保每个视频能并发调用所有模型
MAX_CONCURRENT_MODELS = len(TEST_MODELS) + 1

# ================= 辅助函数 =================

def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

def extract_video_segment(video_path, start_time=3, end_time=7):
    """
    从视频中提取指定时间段（3-7秒）的片段
    返回: (提取的视频片段路径, 是否是临时文件)
    """
    if not os.path.exists(video_path):
        return video_path, False
    
    segment_path = video_path.replace(".mp4", f"_segment_{start_time}s_{end_time}s.mp4")
    
    cmd = [
        "ffmpeg", "-y",                # 覆盖输出
        "-i", video_path,              # 输入
        "-ss", str(start_time),        # 开始时间
        "-to", str(end_time),          # 结束时间
        "-c:v", "libx264",             # 视频编码器
        "-pix_fmt", "yuv420p",         # 像素格式
        "-an",                         # 移除音频
        segment_path
    ]
    
    try:
        # 执行截取，静默输出
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Extracted video segment from {start_time}s to {end_time}s: {segment_path}")
        return segment_path, True
    except Exception as e:
        print(f"❌ Video segment extraction failed: {e}. Using original video.")
        return video_path, False

def compress_video_smart(video_path, target_size_mb=7.0):
    """
    智能压缩视频：如果视频超过目标大小，则通过降低帧率来压缩，
    同时保持画质（CRF 18）不变。
    
    关键调整：
    我们将 target_size_mb 默认值从 9.5MB 降低到 7.0MB。
    原因：Base64 编码会使文件大小增加约 33%。
    - 7.0MB * 1.33 ≈ 9.3MB (安全，小于 API 的 10MB 限制)
    - 9.0MB * 1.33 ≈ 12.0MB (会报错 Exceeded limit)
    
    返回: (文件路径, 是否是临时文件)
    """
    if not os.path.exists(video_path):
        return video_path, False

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    
    # 不管文件大小，都进行压缩
    print(f"📦 Video {os.path.basename(video_path)} is {file_size:.2f}MB. Compressing to optimize for API...")
    
    temp_path = video_path.replace(".mp4", "_compressed_temp.mp4")
    
    # 策略：降低帧率，但保持高质量
    # 调整为 5fps (1秒视频也有5帧，满足 >4帧 的要求)
    target_fps = "5" 
    
    # 如果文件特别大 (>30MB)，尝试降低到 3fps (对于长视频也足够)
    if file_size > 30:
        target_fps = "3"

    cmd = [
        "ffmpeg", "-y",                # 覆盖输出
        "-i", video_path,              # 输入
        "-r", target_fps,              # 目标帧率
        "-c:v", "libx264",             # 编码器
        "-pix_fmt", "yuv420p",         # 强制使用 yuv420p 像素格式
        "-crf", "18",                  # 恒定质量因子 (18=高画质/视觉无损)
        "-preset", "veryfast",         # 编码速度
        "-an",                         # 移除音频
        temp_path
    ]
    
    try:
        # 执行压缩，静默输出
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 检查压缩后的大小
        new_size = os.path.getsize(temp_path) / (1024 * 1024)
        
        # 如果还是太大，尝试进一步压缩 (CRF 23 仍然很清晰)
        if new_size > target_size_mb:
            print(f"⚠️ Still too large ({new_size:.2f}MB). Re-compressing aggressively...")
            cmd_aggressive = [
                "ffmpeg", "-y", "-i", video_path,
                "-r", "2", "-c:v", "libx264", # 只有在大文件(通常较长)才敢用2fps
                "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast", "-an",
                temp_path
            ]
            subprocess.run(cmd_aggressive, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            new_size = os.path.getsize(temp_path) / (1024 * 1024)

        print(f"✅ Compressed to {new_size:.2f}MB (FPS: {target_fps})")
        return temp_path, True
        
    except Exception as e:
        print(f"❌ Compression failed (ffmpeg might be missing): {e}. Using original.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return video_path, False

def call_transit_model(raw_text):
    """调用中转模型转换格式"""
    transit_system_prompt = """
    You are a professional text format converter. Your job is to convert the given text into a strict JSON format.
    The JSON should have two fields:
    1. "answer": The selected option letter (e.g., "A", "B", "C", "D")
    2. "reasoning": "<Short explanation>"
    OUTPUT FORMAT: Return a STRICT JSON object with no additional text.
    """
    
    transit_user_prompt = f"Please convert the following text into JSON format:\n{raw_text}"
    
    transit_messages = [
        {"role": "system", "content": transit_system_prompt},
        {"role": "user", "content": transit_user_prompt}
    ]
    
    api_config = API_CONFIGS["siliconflow"]
    transit_client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
    
    for retry in range(3):
        try:
            response = transit_client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=transit_messages,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
                timeout=60
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            if retry == 2: return {"answer": "N/A", "reasoning": f"Transit Error: {e}"}
            time.sleep(2)
    return {"answer": "N/A", "reasoning": "Transit failed"}

def call_vlm_model(video_path, question, options, model_name):
    """调用 VLM 模型"""
    SUPPORT_JSON_MODELS = ["Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3-vl-plus", "qwen3-vl-flash", "qwen-vl-max-latest"]
    api_provider = MODEL_TO_API.get(model_name, "siliconflow")
    api_config = API_CONFIGS[api_provider]
    
    client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
    
    system_prompt = """
    You are an expert video understanding AI. 
    Analyze the video and return a JSON object: {"answer": "<option>", "reasoning": "<text>"}
    """

    user_prompt = f"""
    # VIDEO QUESTION:
    **Question:** {question}
    **OPTIONS:**
    {chr(10).join([f"{key}. {value}" for key, value in options.items()])}
    Select the correct answer.
    """

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    except FileNotFoundError:
        return {"answer": "N/A", "reasoning": "Video file not found during read"}

    # 构建视频内容部分
    video_url = f"data:video/mp4;base64,{video_base64}"
    
    if api_provider == "aliyun":
        # 阿里云 OpenAI 兼容模式格式 (type: video_url)
        video_content = {
            "type": "video_url",
            "video_url": {"url": video_url}
        }
    else:
        # SiliconFlow 格式 (带 fps 等参数)
        video_content = {
            "type": "video_url",
            "video_url": {
                "url": video_url,
                "detail": "auto",
                "max_frames": 12,
                "fps": 1
            }
        }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}, video_content]}
    ]

    max_retries = 3
    retry_delay = 5
    
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"} if model_name in SUPPORT_JSON_MODELS else None,
                timeout=60
            )
            
            content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            
            if model_name in SUPPORT_JSON_MODELS:
                return json.loads(content)
            else:
                return call_transit_model(content)
                
        except Exception as e:
            print(f"❌ {model_name} Request Failed: {e}")
            if retry < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return {"answer": "N/A", "reasoning": f"Exception: {e}"}
    
    return {"answer": "N/A", "reasoning": "Max retries exceeded"}

# ================= 主程序 =================

def process_video(eval_entry, prompt_index, test_results, lock, processed_videos):
    """处理单个视频的函数，用于并行执行"""
    object_name = eval_entry['object_name']
    scenario_type = eval_entry['scenario_type']
    
    key = f"{object_name}_{scenario_type}"
    if key not in prompt_index:
        return
    
    if eval_entry['status'] != 'evaluated' or not eval_entry.get('best_video'):
        return
    
    prompt_entry = prompt_index[key]
    mcq = prompt_entry['mcq']
    
    best_video_file = eval_entry['best_video']
    original_video_path = os.path.join(VIDEO_DIR, best_video_file)
    
    # 使用object_name_scenario_type组合作为唯一标识，避免同一视频被重复处理
    unique_key = f"{object_name}_{scenario_type}"
    
    # 检查该组合是否已经处理过，如果是则跳过
    if unique_key in processed_videos:
        print(f"⏭️  Skipping {best_video_file} ({object_name} - {scenario_type})... already processed")
        return
    
    # 提前将视频标记为已处理，避免后续可能的重复处理
    with lock:
        processed_videos.add(unique_key)
    
    print(f"🔄 Processing {best_video_file} ({object_name} - {scenario_type})...")
    
    # === 视频处理 ===
    # 1. 首先提取3-7秒的视频片段
    segment_path, is_segment_temp = extract_video_segment(original_video_path, start_time=3, end_time=7)
    
    # 2. 然后压缩视频片段
    current_video_path, is_compressed_temp = compress_video_smart(segment_path)
    
    # 使用固定问题
    question = "What most likely happened according to the video?"
    
    try:
        # 并行调用所有模型
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_MODELS, len(TEST_MODELS))) as executor:
            # 提交所有模型调用任务
            future_to_model = {
                executor.submit(call_vlm_model, current_video_path, question, mcq['options'], model_name): model_name
                for model_name in TEST_MODELS
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_response = future.result()
                    
                    correct_answer = mcq['correct_answer']
                    model_answer = model_response['answer']
                    is_correct = str(model_answer).upper() == str(correct_answer).upper()
                    
                    # 使用锁确保线程安全地更新结果列表
                    with lock:
                        test_results.append({
                            "object_name": object_name,
                            "scenario_type": scenario_type,
                            "video_file": best_video_file,
                            "mcq_question": question,  # 使用实际使用的问题
                            "mcq_options": mcq['options'],
                            "correct_answer": correct_answer,
                            "model_name": model_name,
                            "model_answer": model_answer,
                            "is_correct": is_correct,
                            "model_reasoning": model_response['reasoning'],
                            "original_score": eval_entry['best_score']
                        })
                        
                    print(f"   [{model_name}] Ans: {model_answer} ({'✅' if is_correct else '❌'})")
                except Exception as e:
                    print(f"   [{model_name}] Error: {e}")
                    # 使用锁确保线程安全地更新结果列表
                    with lock:
                        test_results.append({
                            "object_name": object_name,
                            "scenario_type": scenario_type,
                            "video_file": best_video_file,
                            "mcq_question": question,  # 使用实际使用的问题
                            "mcq_options": mcq['options'],
                            "correct_answer": mcq['correct_answer'],
                            "model_name": model_name,
                            "model_answer": "N/A",
                            "is_correct": False,
                            "model_reasoning": f"Error: {e}",
                            "original_score": eval_entry['best_score']
                        })
    
    finally:
        # === 清理临时文件 ===
        temp_files = []
        if is_compressed_temp and current_video_path != segment_path:
            temp_files.append(current_video_path)
        if is_segment_temp and segment_path != original_video_path:
            temp_files.append(segment_path)
            
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

def main():
    import threading
    
    print(f"🚀 Starting Video Understanding Test using {', '.join(TEST_MODELS)}...")
    print("🔍 测试配置: 截取视频3-7秒片段，使用固定问题")
    
    eval_files = [f for f in os.listdir('.') if f.startswith('physibench_evaluated_v') and f.endswith('.json')]
    eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not eval_files:
        print("❌ No evaluation files found.")
        return
    
    eval_file = eval_files[0]
    version = re.search(r'v(\d+)', eval_file).group(1)
    
    # 寻找对应的MCQ结果文件，支持不同的命名格式（如mcq_blind_test_results_v76_unique.json）
    mcq_files = [f for f in os.listdir('.') if f.startswith('mcq_blind_test_results_') and f.endswith('.json')]
    matched_mcq_files = []
    for f in mcq_files:
        match = re.search(r'v(\d+)', f)
        if match and match.group(1) == version:
            matched_mcq_files.append(f)
    
    if not matched_mcq_files:
        print(f"❌ No MCQ result files found for version v{version}")
        return
    
    # 选择最新的MCQ结果文件
    mcq_file = max(matched_mcq_files, key=os.path.getmtime)
    
    with open(eval_file, "r", encoding='utf-8') as f: eval_results = json.load(f)
    with open(mcq_file, "r", encoding='utf-8') as f: mcq_results = json.load(f)
    
    # 创建索引，映射object_name_scenario_type到mcq结果
    prompt_index = {f"{result['director_data']['object_name']}_{result['director_data']['scenario_type']}": result for result in mcq_results if result.get('blind_test_passed', False)}
    
    test_results = []
    lock = threading.Lock()  # 用于线程安全地更新结果列表
    
    # 设置输出文件名
    output_file = f"mid_understanding_test_v{version}.json"
    processed_videos = set()
    
    # 检查是否有已存在的测试结果文件，用于断点续传
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
            test_results = existing_results
            
            # 提取已处理的object_name_scenario_type组合
            for result in existing_results:
                unique_key = f"{result['object_name']}_{result['scenario_type']}"
                processed_videos.add(unique_key)
            
            print(f"📋 Found existing results file: {output_file}")
            print(f"   - {len(existing_results)} results loaded")
            print(f"   - {len(processed_videos)} unique object-scenario combinations already processed")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Failed to load existing results: {e}")
            print("   Starting fresh...")
    
    # 计算总的并发数
    # 每个视频需要处理len(TEST_MODELS)个模型，所以总的并发数应该是MAX_CONCURRENT_MODELS * 视频并发数
    # 降低并发数以避免API速率限制 (TPM limit reached)
    video_concurrency = 2
    total_concurrent_workers = video_concurrency
    
    print(f"⚙️  Using {total_concurrent_workers} concurrent video workers, each with up to {MAX_CONCURRENT_MODELS} model workers...")
    
    # 并行处理所有视频
    with ThreadPoolExecutor(max_workers=total_concurrent_workers) as executor:
        # 提交所有视频处理任务
        futures = []
        valid_entries = 0
        for eval_entry in eval_results:
            object_name = eval_entry['object_name']
            scenario_type = eval_entry['scenario_type']
            key = f"{object_name}_{scenario_type}"
            
            # 只提交有效的评估条目
            if key in prompt_index and eval_entry['status'] == 'evaluated' and eval_entry.get('best_video'):
                future = executor.submit(process_video, eval_entry, prompt_index, test_results, lock, processed_videos)
                futures.append(future)
                valid_entries += 1
        
        print(f"📋 Found {valid_entries} valid evaluation entries matching MCQ blind test results...")
        
        # 等待所有任务完成
        print(f"📋 Submitted {len(futures)} video processing tasks...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Video processing error: {e}")

    # 计算统计信息
    if test_results:
        correct_count = sum(1 for r in test_results if r['is_correct'])
        print(f"\n📊 Overall Accuracy: {correct_count / len(test_results) * 100:.1f}%")
        
    # 保存
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # 统计每个模型在每个场景下的正确率
    if not test_results:
        print("\n❌ 没有测试结果可统计")
        return
    
    print(f"\n📊 开始统计每个模型在每个场景下的正确率...")
    
    # 收集所有唯一的场景类型和模型名称
    scenario_types = sorted(set(r['scenario_type'] for r in test_results))
    model_names = sorted(set(r['model_name'] for r in test_results))
    
    # 统计数据结构: {model: {scenario: (correct_count, total_count)}}
    stats = {}
    for model in model_names:
        stats[model] = {}
        for scenario in scenario_types:
            # 筛选该模型在该场景下的所有结果
            model_scenario_results = [
                r for r in test_results 
                if r['model_name'] == model and r['scenario_type'] == scenario
            ]
            if not model_scenario_results:
                stats[model][scenario] = (0, 0)
                continue
            
            total = len(model_scenario_results)
            correct = sum(1 for r in model_scenario_results if r['is_correct'])
            stats[model][scenario] = (correct, total)
    
    # 使用 PrettyTable 打印统计表格
    print(f"\n{'='*100}")
    print("📈 各模型在各场景下的正确率统计")
    print(f"{'='*100}")
    
    # 创建表格
    table = PrettyTable()
    
    # 设置表头
    field_names = ["场景类型"]
    for model in model_names:
        field_names.append(model)
    field_names.append("情景总计")  # 添加情景总计列
    table.field_names = field_names
    
    # 填充表格内容
    for scenario in scenario_types:
        row = [scenario]
        scenario_total_correct = 0
        scenario_total_count = 0
        
        for model in model_names:
            correct, total = stats[model][scenario]
            if total == 0:
                accuracy = "N/A"
            else:
                accuracy = f"{correct/total*100:.1f}% ({correct}/{total})"
            row.append(accuracy)
            
            # 累计场景的总正确数和总测试数
            scenario_total_correct += correct
            scenario_total_count += total
        
        # 计算场景总计
        if scenario_total_count == 0:
            scenario_accuracy = "N/A"
        else:
            scenario_accuracy = f"{scenario_total_correct/scenario_total_count*100:.1f}% ({scenario_total_correct}/{scenario_total_count})"
        row.append(scenario_accuracy)
        
        table.add_row(row)
    
    # 添加总计行
    total_row = ["总计"]
    overall_total_correct = 0
    overall_total_count = 0
    
    for model in model_names:
        all_correct = sum(correct for correct, total in stats[model].values())
        all_total = sum(total for correct, total in stats[model].values())
        if all_total == 0:
            total_accuracy = "N/A"
        else:
            total_accuracy = f"{all_correct/all_total*100:.1f}% ({all_correct}/{all_total})"
        total_row.append(total_accuracy)
        
        # 累计所有场景的总正确数和总测试数
        overall_total_correct += all_correct
        overall_total_count += all_total
    
    # 计算所有场景总计的总计
    if overall_total_count == 0:
        overall_accuracy = "N/A"
    else:
        overall_accuracy = f"{overall_total_correct/overall_total_count*100:.1f}% ({overall_total_correct}/{overall_total_count})"
    total_row.append(overall_accuracy)
    
    table.add_row(total_row)
    
    # 设置表格样式
    table.align = "l"  # 左对齐
    table.padding_width = 1  # 单元格内边距
    
    # 打印表格
    print(table)
    print(f"{'='*100}")

if __name__ == "__main__":
    main()