import os
import json
import cv2
import base64
import requests
import re
import numpy as np
import time
import threading
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from phase0_config_data import client_gemini, DIRECTOR_MODEL

# ================= 配置部分 =================

# Remove the old API configuration since we're using gemini
# API_KEY = "sk-izrxbwrnxotwsvcngnnmwivxmyqukrivnjcoszzpscmfasjz"
# BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"

# Use the gemini model from config
MODEL_NAME = DIRECTOR_MODEL

# Support for JSON format output
SUPPORT_JSON_MODELS = [DIRECTOR_MODEL]  # Use the model from config

# 🟢 修改：不再硬编码输入文件，设为 None 以便自动查找
INPUT_JSON_FILE = None 
VIDEO_DIR = "t2v_videos"
OUTPUT_JSON_FILE = None  # 将在main函数中根据版本号动态设置

# ================= 辅助函数 =================

def sanitize_filename(name):
    """
    与生成视频时的命名逻辑保持一致
    """
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

def call_transit_model(raw_text, is_coherence=False):
    """调用中转模型转换格式为JSON
    is_coherence: 是否为连贯性评估，决定返回的JSON字段结构
    """
    if is_coherence:
        transit_system_prompt = """
        You are a professional text format converter. Your job is to convert the given text into a strict JSON format.
        The JSON should have three fields:
        1. "expected_end_frame": "<Description of what the end frame should look like>"
        2. "reasoning": "<Explanation of the evaluation>"
        3. "score": <0-10, allow decimals>
        OUTPUT FORMAT: Return a STRICT JSON object with no additional text.
        """
        
        transit_user_prompt = f"Please convert the following text into JSON format with expected_end_frame, reasoning, and score fields:\n{raw_text}"
    else:
        transit_system_prompt = """
        You are a professional text format converter. Your job is to convert the given text into a strict JSON format.
        The JSON should have two fields:
        1. "reasoning": "<Short explanation of why>"
        2. "score": <0-10, allow decimals>
        OUTPUT FORMAT: Return a STRICT JSON object with no additional text.
        """
        
        transit_user_prompt = f"Please convert the following text into JSON format:\n{raw_text}"
    
    transit_messages = [
        {"role": "system", "content": transit_system_prompt},
        {"role": "user", "content": transit_user_prompt}
    ]
    
    max_retries = 3
    retry_delay = 5
    
    for retry in range(max_retries):
        try:
            response = client_gemini.chat.completions.create(
                model=DIRECTOR_MODEL,
                messages=transit_messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            # 确保所有必需字段都存在
            if is_coherence and 'expected_end_frame' not in result:
                result['expected_end_frame'] = 'No expected end frame description provided'
            return result
        except Exception as e:
            print(f"❌ Transit Error: {e}")
            
        if retry < max_retries - 1:
            print(f"🔄 Transit retrying in {retry_delay} seconds... (Attempt {retry + 2}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
    
    if is_coherence:
        return {"expected_end_frame": "Transit failed to convert to JSON", "score": 0, "reasoning": "Transit failed to convert to JSON"}
    else:
        return {"score": 0, "reasoning": "Transit failed to convert to JSON"}

def extract_frames(video_path):
    """
    从视频中提取 3 帧 (Start, Mid, End)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    timestamps = [0.1, duration / 2, duration-0.1]
    
    extracted_frames = []
    
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            new_width = 512
            new_height = int(height * (new_width / width))
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            _, buffer = cv2.imencode('.jpg', frame_resized)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            extracted_frames.append(b64_str)
        else:
            print(f"⚠️ 无法读取时间点 {t}s 的帧")
    
    cap.release()
    
    if len(extracted_frames) == 3:
        return extracted_frames
    return None

def call_vlm_evaluator(frame, description, segment_name):
    """
    调用 Qwen-VL 对单个视频片段进行评分，包含重试逻辑
    """
    system_prompt = """
    You are an expert Video Quality Assurance AI. 
    Your job is to evaluate if a specific frame from a generated video matches the required description.
    You will receive one frame from the video and the expected textual description for that specific moment.
    
    YOUR FOCUS: ONLY evaluate whether this specific frame matches its specific description.
    Do NOT consider other frames or descriptions.
    
    CRITERIA:
    1. **Adherence:** Does the visual content match the expected state of the described action at this specific phase?
    2. **Accuracy:** Is the object, scene, or action described clearly visible in this frame?
    
    OUTPUT FORMAT:
    Return a STRICT JSON object:
    {
        "reasoning": "<Short explanation of why>",
        "score": <0-10, allow decimals>
    }
    """

    # 根据片段类型添加特殊提示
    segment_special_note = ""
    if segment_name == "Start":
        segment_special_note = "⚠️  SPECIAL NOTE FOR START FRAME: If the description contains phrases like 'suddenly appears' or similar, it is reasonable that the object is not yet fully visible or present in this frame."
    elif segment_name == "End":
        segment_special_note = "⚠️  SPECIAL NOTE FOR END FRAME: If the description contains phrases like 'disappears' or 'merges with something else', it is reasonable that the object is no longer visible or distinguishable in this frame."
    
    user_prompt = f"""
    # 🖼️ CURRENT SEGMENT TO EVALUATE:
    **Segment:** {segment_name}
    **Expected Description:** {description}
    {segment_special_note}

    # 🖼️ PROVIDED IMAGE:
    This is the frame from the {segment_name} segment.

    Evaluate if this frame matches the expected description for this specific moment, considering any special notes provided.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
            ]
        }
    ]

    # 根据模型是否支持JSON格式来决定是否设置response_format
    max_retries = 5  # 增加最大重试次数
    retry_delay = 10  # 增加初始延迟到10秒
    
    for retry in range(max_retries):
        try:
            response = client_gemini.chat.completions.create(
                model=DIRECTOR_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"} if DIRECTOR_MODEL in SUPPORT_JSON_MODELS else None
            )
            
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            
            # 如果模型支持JSON格式，直接解析
            if DIRECTOR_MODEL in SUPPORT_JSON_MODELS:
                return json.loads(content)
            else:
                # 否则调用中转模型转换格式
                return call_transit_model(content)
        except Exception as e:
            print(f"❌ Request Failed: {e}")
            if retry < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds... (Attempt {retry + 2}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                return {"score": 0, "reasoning": f"Exception: {e}"}
    
    return {"score": 0, "reasoning": "Max retries exceeded"}


def evaluate_coherence(start_frame, end_frame, synopsis, start_description, end_description):
    """
    评估首尾帧与梗概的连贯性
    首先基于首帧和梗概推理尾帧应该是什么样的画面，
    然后评估上传的尾帧与期望是否相符
    """
    system_prompt = """
    You are an expert Video Quality Assurance AI specializing in narrative coherence evaluation.
    
    Your task is to analyze video frames and evaluate story coherence. You will receive:
    1. A start frame showing the beginning of a story
    2. An end frame showing the conclusion
    3. A synopsis describing the events that should occur
    4. Descriptions of what should be in each frame
    
    YOUR EVALUATION PROCESS:
    1. First, examine the start frame and understand what it shows
    2. Read the synopsis to understand what events should happen
    3. Determine what the end frame SHOULD look like after these events
    4. Examine the actual end frame provided
    5. Compare the expected end frame with the actual end frame
    6. Score how well they match (0-10 scale)
    
    EVALUATION CRITERIA:
    - Logical progression: Does the story make visual sense from start to end?
    - Event completion: Are the synopsis events properly reflected in the frames?
    - Visual consistency: Do the visual elements transform appropriately?
    
    IMPORTANT: You must provide a detailed description of what the end frame should look like based on your analysis of the start frame and synopsis events.
    
    OUTPUT FORMAT - YOU MUST RETURN EXACTLY THIS JSON STRUCTURE:
    {
        "expected_end_frame": "Detailed description of what the end frame should show based on start frame + synopsis",
        "reasoning": "Explanation of your analysis and comparison",
        "score": "Number from 0-10 with 1 decimal place"
    }
    """
    
    user_prompt = f"""
    # 🎬 STORY ANALYSIS TASK
    
    ## STARTING POINT
    **Start Frame Description:** {start_description}
    This is what the video begins with.
    
    ## STORY EVENTS  
    **Synopsis:** {synopsis}
    These events should occur between start and end.
    
    ## EXPECTED CONCLUSION
    **End Frame Description:** {end_description}
    This is what should be shown in the end frame.
    
    ## YOUR TASK
    1. Look at the START frame first - what does it show?
    2. Read the SYNOPSIS - what should happen next?
    3. Determine what the END frame SHOULD look like after these events
    4. Look at the ACTUAL end frame provided
    5. Compare: Does the actual end frame match your expectation?
    6. Score the match (0-10) and explain your reasoning
    
    ## FRAMES TO ANALYZE
    Below are the start and end frames. Analyze them carefully and provide your evaluation in the required JSON format.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{start_frame}"}},
                {"type": "text", "text": "This is the START frame."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{end_frame}"}},
                {"type": "text", "text": "This is the END frame to evaluate."}
            ]
        }
    ]

    max_retries = 5
    retry_delay = 10
    
    for retry in range(max_retries):
        try:
            response = client_gemini.chat.completions.create(
                model=DIRECTOR_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            
            if DIRECTOR_MODEL in SUPPORT_JSON_MODELS:
                result = json.loads(content)
                return result
            else:
                return call_transit_model(content, is_coherence=True)
        except Exception as e:
            print(f"❌ Coherence Request Failed: {e}")
            if retry < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds... (Attempt {retry + 2}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return {"expected_end_frame": "Evaluation failed", "reasoning": f"Exception: {e}", "score": 0}
    
    return {"expected_end_frame": "Evaluation failed", "reasoning": "Max retries exceeded", "score": 0}

# ================= 辅助函数 - 并行处理 =================

def process_entry(entry):
    """处理单个条目，返回评估结果"""
    constraints = entry.get('constraints', {})
    keyword = constraints.get('keyword', '')
    s_type = constraints.get('type', '')
    
    safe_keyword = sanitize_filename(keyword)
    # 确保正确处理新的场景名，与 Phase 2 保持一致
    safe_type = sanitize_filename(s_type.replace(" Scenario", ""))
    
    timeline = entry.get('timeline', {})
    synopsis = entry.get('synopsis', '')
    
    # 获取三个片段的描述
    segment_descriptions = {
        "Start": timeline.get("start_caption", "N/A"),
        "Middle": timeline.get("middle_caption", "N/A"),
        "End": timeline.get("end_caption", "N/A")
    }

    print(f"🔄 Processing {keyword} - {s_type}...")
    
    best_score = -1
    best_video_file = None
    best_reasoning = ""
    candidates = []
    
    found_videos = []
    for i in range(1, 6): 
        filename = f"{safe_keyword}_{safe_type}_{i}.mp4"
        filepath = os.path.join(VIDEO_DIR, filename)
        
        if os.path.exists(filepath):
            found_videos.append((filepath, filename))
    
    if not found_videos:
        eval_entry = {
            "object_name": keyword,
            "scenario_type": s_type,
            "status": "missing_videos"
        }
        return eval_entry
        
    for filepath, filename in found_videos:
        frames = extract_frames(filepath)
        
        if frames and len(frames) == 3:
            # 对每个片段单独评分
            segment_scores = []
            valid_video = True  # 标记视频是否有效（所有片段都及格）
            
            for i, (segment_name, frame) in enumerate(zip(["Start", "Middle", "End"], frames)):
                description = segment_descriptions[segment_name]
                eval_result = call_vlm_evaluator(frame, description, segment_name)
                
                segment_score = {
                    "segment": segment_name,
                    "score": float(eval_result.get('score', 0)),
                    "reasoning": eval_result.get('reasoning', '')
                }
                segment_scores.append(segment_score)
                
                # 检查是否有片段不及格（<6分）
                if segment_score["score"] < 6:
                    valid_video = False
            
            # 如果视频帧匹配通过，再进行连贯性评估
            if valid_video:
                # 计算三个片段的平均分作为视频帧匹配分数
                frame_match_score = sum(seg["score"] for seg in segment_scores) / len(segment_scores)
                
                # 进行连贯性评估
                start_frame, _, end_frame = frames
                coherence_result = evaluate_coherence(
                    start_frame, 
                    end_frame, 
                    synopsis, 
                    segment_descriptions["Start"], 
                    segment_descriptions["End"]
                )
                
                coherence_score = float(coherence_result.get('score', 0))
                
                # 检查连贯性是否及格
                if coherence_score >= 6:
                    # 综合分数：帧匹配分数（占70%） + 连贯性分数（占30%）
                    final_score = (frame_match_score * 0.7) + (coherence_score * 0.3)
                    final_reasoning = f"Frame match score: {frame_match_score:.2f}, Coherence score: {coherence_score:.2f}"
                else:
                    # 连贯性不及格，视频无效
                    valid_video = False
                    final_score = 0
                    final_reasoning = "Video invalid due to failed coherence evaluation."
            else:
                # 有片段不及格，视频作废
                final_score = 0
                final_reasoning = "Video invalid due to failed segment(s)."
                coherence_result = None
            
            # 准备连贯性评估结果
            coherence_evaluation = None
            if coherence_result:
                coherence_evaluation = {
                    "expected_end_frame": coherence_result.get('expected_end_frame', ''),
                    "reasoning": coherence_result.get('reasoning', ''),
                    "score": coherence_result.get('score', 0)
                }
            
            candidates.append({
                "video_file": filename,
                "final_score": final_score,
                "segment_scores": segment_scores,
                "coherence_evaluation": coherence_evaluation,
                "reasoning": final_reasoning,
                "valid": valid_video
            })
            
            # 更新最佳视频（只考虑有效视频）
            if valid_video and final_score > best_score:
                best_score = final_score
                best_video_file = filename
                best_reasoning = final_reasoning
        else:
            candidates.append({
                "video_file": filename,
                "final_score": 0,
                "segment_scores": [],
                "coherence_evaluation": None,
                "reasoning": "Frame extraction failed (must get exactly 3 frames)",
                "valid": False
            })

    eval_entry = {
        "object_name": keyword,
        "scenario_type": s_type,
        "status": "evaluated",
        "best_video": best_video_file,
        "best_score": best_score,
        "best_reasoning": best_reasoning,
        "all_candidates": candidates,
        "has_valid_video": best_video_file is not None  # 标记是否有有效视频
    }
    
    return eval_entry

# ================= 主程序 =================

def main():
    print(f"� Starting Video Evaluation using {MODEL_NAME}...")
    
    # 🟢 1. 自动查找最新的 surprise JSON 文件 (如果没有指定)
    target_file = INPUT_JSON_FILE
    if not target_file:
        json_files = [f for f in os.listdir('.') if f.startswith('physibench_surprise_') and f.endswith('.json')]
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if json_files:
            target_file = json_files[0]
            print(f"📂 Auto-selected latest file: {target_file}")
        else:
            print(f"❌ No physibench_surprise_*.json files found.")
            return

    if not os.path.exists(target_file):
        print(f"❌ Input file not found: {target_file}")
        return
    
    # 🟢 2. 从输入文件名中提取版本号
    version_match = re.search(r'v(\d+)', target_file)
    if version_match:
        version = version_match.group(1)
        print(f"📌 Extracted version: v{version}")
    else:
        print(f"⚠️  Could not extract version from filename: {target_file}")
        version = "unknown"
    
    # 🟢 3. 设置输出文件名，使用提取的版本号
    global OUTPUT_JSON_FILE
    OUTPUT_JSON_FILE = f"physibench_evaluated_v{version}.json"
    print(f"📁 Output file will be: {OUTPUT_JSON_FILE}")

    with open(target_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 Loaded {len(data)} entries")
    
    # 🔄 检查是否存在已完成的结果文件（断点重续）
    existing_results = []
    processed_entries = set()
    no_valid_video_count = 0  # 统计没有最终视频的prompt数量
    missing_videos_count = 0  # 统计没有对应视频的prompt数量
    
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
            
            if existing_results:
                # 创建已处理条目的唯一标识集合
                processed_entries = {f"{entry['object_name']}_{entry['scenario_type']}" for entry in existing_results}
                print(f"📁 Found existing results: {len(existing_results)} entries already processed")
                
                # 统计已存在结果中的各类数量
                for entry in existing_results:
                    if entry.get('status') == 'missing_videos':
                        missing_videos_count += 1
                        no_valid_video_count += 1
                    elif not entry.get('has_valid_video', False):
                        no_valid_video_count += 1
        except json.JSONDecodeError:
            print(f"⚠️  Existing results file {OUTPUT_JSON_FILE} is invalid, will start fresh")
    
    # 筛选未处理的条目
    unprocessed_data = []
    for entry in data:
        constraints = entry.get('constraints', {})
        keyword = constraints.get('keyword', '')
        s_type = constraints.get('type', '')
        entry_id = f"{keyword}_{s_type}"
        
        if entry_id not in processed_entries:
            unprocessed_data.append(entry)
    
    print(f"🔄 Will process {len(unprocessed_data)} new entries")
    
    # 合并现有结果
    results = existing_results.copy()
    
    # 如果有未处理的条目，开始处理
    if unprocessed_data:
        # 2. 并行处理未完成的条目
        # 设置最大并发数，根据API限制和系统资源调整
        max_concurrent = 2  # 减少并发数以避免API速率限制
        print(f"⚙️  Using {max_concurrent} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 提交所有未处理的任务
            futures = [executor.submit(process_entry, entry) for entry in unprocessed_data]
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Scenarios"):
                try:
                    eval_entry = future.result()
                    results.append(eval_entry)
                    
                    # 检查是否没有最终视频
                    if eval_entry.get('status') == 'missing_videos':
                        missing_videos_count += 1
                        no_valid_video_count += 1
                    elif not eval_entry.get('has_valid_video', False):
                        no_valid_video_count += 1
                    
                    # 保存进度，每次有新结果就保存
                    with open(OUTPUT_JSON_FILE, "w", encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"❌ Error processing entry: {e}")
    else:
        print("✅ All entries have already been processed!")
        # 保存现有结果（确保格式正确）
        with open(OUTPUT_JSON_FILE, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 计算各类prompt数量
    total_prompts = len(results)
    passed_count = total_prompts - no_valid_video_count
    failed_with_videos_count = no_valid_video_count - missing_videos_count  # 有视频但没有有效视频的数量
    
    print(f"\n✅ Evaluation Complete! Results saved to {OUTPUT_JSON_FILE}")
    print(f"📊 Statistics:")
    print(f"   - Total prompts: {total_prompts}")
    print(f"   - Passed prompts: {passed_count} (have valid videos)")
    print(f"   - Failed prompts: {failed_with_videos_count} (have videos but no valid ones)")
    print(f"   - Missing videos: {missing_videos_count} (no corresponding videos)")

if __name__ == "__main__":
    main()