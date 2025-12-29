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

def compress_video_smart(video_path, target_size_mb=7.0):
    """
    智能压缩视频：只保留视频的3-7秒片段，并在需要时进行压缩以确保文件大小合适。
    
    主要功能：
    1. 提取视频的3-7秒片段（共4秒）作为分析目标
    2. 进行必要的压缩以确保最终文件大小不超过目标大小
    3. 保持视频质量（使用合适的CRF值）同时优化文件大小
    
    参数说明：
    - video_path: 输入视频文件路径
    - target_size_mb: 目标文件大小（MB），默认为7.0MB
    
    关键调整：
    1. 我们将 target_size_mb 默认值设置为 7.0MB，因为 Base64 编码会使文件大小增加约 33%。
       - 7.0MB * 1.33 ≈ 9.3MB (安全，小于 API 的 10MB 限制)
       - 9.0MB * 1.33 ≈ 12.0MB (会报错 Exceeded limit)
    2. 只保留视频的3-7秒片段进行处理和分析，这样可以：
       - 显著减小文件大小
       - 保留视频中最可能包含关键信息的部分
       - 加快处理速度
    
    处理流程：
    1. 首先提取3-7秒的视频片段
    2. 如果提取后的片段仍然超过目标大小，尝试通过降低帧率进一步压缩
    3. 保持良好的视频质量（使用适当的CRF值）
    
    返回: (处理后的文件路径, 是否是临时文件)
    """
    if not os.path.exists(video_path):
        return video_path, False

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    
    # 不管文件大小，都进行压缩和剪辑
    print(f"📦 Video {os.path.basename(video_path)} is {file_size:.2f}MB. Processing...")
    print(f"✂️  Extracting 3-7 second segment as requested...")
    
    temp_path = video_path.replace(".mp4", "_segmented_temp.mp4")
    
    # 首先剪辑视频，只保留3-7秒部分
    clip_cmd = [
        "ffmpeg", "-y",                # 覆盖输出
        "-i", video_path,              # 输入
        "-ss", "3",                   # 开始时间（秒）
        "-t", "4",                    # 持续时间（秒）- 3-7秒，共4秒
        "-c:v", "libx264",             # 编码器
        "-pix_fmt", "yuv420p",         # 强制使用 yuv420p 像素格式
        "-crf", "18",                  # 恒定质量因子 (18=高画质/视觉无损)
        "-preset", "veryfast",         # 编码速度
        "-an",                         # 移除音频
        temp_path
    ]
    
    try:
        # 执行剪辑，静默输出
        subprocess.run(clip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 检查剪辑后的大小
        new_size = os.path.getsize(temp_path) / (1024 * 1024)
        
        # 如果还是太大，尝试进一步压缩 (CRF 23 仍然很清晰)
        if new_size > target_size_mb:
            print(f"⚠️ Segment is still too large ({new_size:.2f}MB). Further compressing...")
            # 策略：降低帧率，但保持高质量
            target_fps = "5"
            
            # 如果文件特别大，尝试降低到 3fps
            if new_size > 30:
                target_fps = "3"
                
            compress_cmd = [
                "ffmpeg", "-y", "-i", temp_path,
                "-r", target_fps, "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast", "-an",
                temp_path + "_compressed.mp4"
            ]
            
            subprocess.run(compress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # 替换为压缩后的文件
            os.remove(temp_path)
            os.rename(temp_path + "_compressed.mp4", temp_path)
            new_size = os.path.getsize(temp_path) / (1024 * 1024)

        print(f"✅ Extracted and processed to {new_size:.2f}MB (3-7 second segment)")
        return temp_path, True
        
    except Exception as e:
        print(f"❌ Processing failed (ffmpeg might be missing): {e}. Using original video.")
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

def call_vlm_for_description(video_path, model_name):
    """调用 VLM 模型查看视频并生成详细描述"""
    api_provider = MODEL_TO_API.get(model_name, "siliconflow")
    api_config = API_CONFIGS[api_provider]
    
    client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
    
    # 描述任务的系统提示
    system_prompt = """
    You are an expert video understanding AI. 
    Please carefully analyze the video and provide a detailed description of everything that happens in the video.
    Your description should be comprehensive, sequential, and capture all important visual elements, movements, and interactions.
    """

    # 请求详细描述的用户提示
    user_prompt = """
    Please provide a detailed description of everything that happens in this video. Describe the scenes, objects, actions, and any changes that occur over time.
    """

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    except FileNotFoundError:
        return "Video file not found during read"

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
                max_tokens=1024,  # 增加tokens以获取更详细的描述
                timeout=60
            )
            
            content = response.choices[0].message.content.strip()
            return content
                
        except Exception as e:
            print(f"❌ {model_name} Description Request Failed: {e}")
            if retry < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return f"Description Error: {e}"
    
    return "Max retries exceeded for description"

def call_vlm_model(video_path=None, question=None, options=None, model_name=None, video_description=None):
    """调用 VLM 模型回答问题
    
    Args:
        video_path: 视频文件路径（第一次问答使用）
        question: 问题文本
        options: 选项字典
        model_name: 模型名称
        video_description: 视频描述文本（第二次问答使用）
        
    Returns:
        包含答案和推理的字典
    """
    SUPPORT_JSON_MODELS = ["Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3-vl-plus", "qwen3-vl-flash", "qwen-vl-max-latest"]
    api_provider = MODEL_TO_API.get(model_name, "siliconflow")
    api_config = API_CONFIGS[api_provider]
    
    client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
    
    system_prompt = """
    You are an expert video understanding AI. 
    Analyze the information provided and return a JSON object: {"answer": "<option>", "reasoning": "<text>"}
    """

    if video_description is not None:
        # 第二次问答：使用视频描述
        user_prompt = f"""
        # VIDEO DESCRIPTION:
        {video_description}
        
        # QUESTION:
        **Question:** {question}
        **OPTIONS:**
        {chr(10).join([f"{key}. {value}" for key, value in options.items()])}
        Select the correct answer based on the video description.
        """
        
        # 构建消息，只有文本，没有视频
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        # 第一次问答（保留原有功能）：使用视频
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

def process_video(eval_entry, prompt_index, test_results, lock, processed_videos, test_mode):
    """处理单个视频的函数，用于并行执行
    实现两次问答流程：
    1. 第一次：让VLM看视频并生成详细描述
    2. 第二次：基于第一次生成的描述回答问题
    """
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
    
    # === 视频压缩处理 ===
    current_video_path, is_temp_file = compress_video_smart(original_video_path)
    
    # 根据测试方式选择题干
    if test_mode == '1':
        question = mcq['question']
    else:
        question = "What most likely happened in the [3-7s] interval according to the video?"
    
    try:
        # ===== 第一次问答：生成视频描述 =====
        print(f"   🔍 First pass: Generating video descriptions...")
        video_descriptions = {}
        
        # 并行调用所有模型生成描述
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_MODELS, len(TEST_MODELS))) as executor:
            future_to_model = {
                executor.submit(call_vlm_for_description, current_video_path, model_name): model_name
                for model_name in TEST_MODELS
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    description = future.result()
                    video_descriptions[model_name] = description
                    print(f"   [{model_name}] Description generated successfully")
                except Exception as e:
                    print(f"   [{model_name}] Error generating description: {e}")
                    video_descriptions[model_name] = f"Error: {e}"
        
        # ===== 第二次问答：基于描述回答问题 =====
        print(f"   📝 Second pass: Answering questions based on descriptions...")
        
        # 并行调用所有模型基于描述回答问题
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_MODELS, len(TEST_MODELS))) as executor:
            future_to_model = {}
            # 为每个模型提交回答任务，使用其自己生成的描述
            for model_name in TEST_MODELS:
                if model_name in video_descriptions:
                    future_to_model[executor.submit(
                        call_vlm_model, 
                        video_path=None,  # 不使用视频，只使用描述
                        question=question, 
                        options=mcq['options'], 
                        model_name=model_name, 
                        video_description=video_descriptions[model_name]
                    )] = model_name
            
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
                            "video_description": video_descriptions.get(model_name, "No description"),  # 保存生成的描述
                            "original_score": eval_entry['best_score'],
                            "test_mode": test_mode  # 记录测试方式
                        })
                        
                    print(f"   [{model_name}] Ans: {model_answer} ({'✅' if is_correct else '❌'})")
                except Exception as e:
                    print(f"   [{model_name}] Error answering question: {e}")
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
                            "video_description": video_descriptions.get(model_name, "No description"),
                            "original_score": eval_entry['best_score'],
                            "test_mode": test_mode  # 记录测试方式
                        })
    
    finally:
        # === 清理临时文件 ===
        if is_temp_file and os.path.exists(current_video_path):
            try:
                os.remove(current_video_path)
            except OSError:
                pass

def main():
    import threading
    
    print(f"🚀 Starting Video Understanding Test using {', '.join(TEST_MODELS)}...")
    
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
    
    # 添加测试方式选择
    print("\n🔍 请选择测试方式:")
    print("1. 使用JSON中的原始题干")
    print("2. 使用通用题干: 'What most likely happened in the [3-7s] interval according to the video?'")
    
    test_mode = input("请输入选择 (1 或 2): ").strip()
    while test_mode not in ['1', '2']:
        print("❌ 无效选择，请重新输入")
        test_mode = input("请输入选择 (1 或 2): ").strip()
    
    test_results = []
    lock = threading.Lock()  # 用于线程安全地更新结果列表
    
    # 检查是否有已存在的测试结果文件，用于断点续传
    if test_mode == '1':
        output_file = f"video_our_method_v{version}.json"
    else:
        output_file = f"video_our_method_generic_v{version}.json"
    processed_videos = set()
    
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
                future = executor.submit(process_video, eval_entry, prompt_index, test_results, lock, processed_videos, test_mode)
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