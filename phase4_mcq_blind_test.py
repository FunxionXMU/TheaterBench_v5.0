import os
import json
import time
import random
from phase0_config_data import (
    client_gemini, client_deepseek,
    DIRECTOR_MODEL, HELPER_MODEL, VERSION
)

# ================= Helper Functions =================

def clean_and_parse_json(content):
    """
    JSON cleaning function to handle extra data and Markdown issues
    """
    if not content: return None
    content = content.strip()
    
    try:
        # 1. Remove Markdown code block markers
        if "```" in content:
            # Find the first and last code block markers
            first_marker = content.find("```")
            last_marker = content.rfind("```")
            if first_marker != -1 and last_marker != -1 and first_marker != last_marker:
                # Extract content between markers
                content = content[first_marker + 3 : last_marker].strip()
                # Remove any "json" prefix
                content = content.replace("json", "", 1).strip()
        
        # 2. More robust handling for "Extra data" - find the complete JSON object
        # This handles cases where there might be multiple { or } in the content
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx + 1]
        
        # 3. Remove any trailing commas inside the JSON (common issue)
        import re
        content = re.sub(r',\s*([}])', r'\1', content)
        
        # 4. Attempt to parse the cleaned content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON Parse Error - {e}")
        print(f"DEBUG: Content being parsed: {content[:500]}...")  # Show first 500 chars for debugging
        return None
    except Exception as e:
        print(f"DEBUG: Unexpected error in JSON parsing - {e}")
        return None


def shuffle_mcq_options(mcq_data):
    # 检查必要字段是否存在
    if not all(key in mcq_data for key in ['question', 'options', 'correct_answer']):
        print(f"DEBUG: Missing required fields in mcq_data: {mcq_data}")
        return None
    
    correct_key_old = mcq_data['correct_answer']
    correct_text = mcq_data['options'].get(correct_key_old)
    if not correct_text:
        print(f"DEBUG: Correct answer key {correct_key_old} not found in options: {mcq_data['options']}")
        return None 
    
    option_texts = list(mcq_data['options'].values())
    random.shuffle(option_texts)
    new_options = {}
    new_correct_key = None
    for i, text in enumerate(option_texts):
        new_key = chr(65 + i) 
        new_options[new_key] = text
        if text == correct_text: new_correct_key = new_key
    
    return {"question": mcq_data['question'], "options": new_options, "correct_answer": new_correct_key}


# ================= Agent 3: Quizmaster =================

def quizmaster_agent(final_prompt, director_data, obj, s_type, feedback=None):
    
    # 从director_data中提取timeline信息和normal_timeline信息
    timeline = director_data.get('timeline', {})
    normal_timeline = director_data.get('normal_timeline', {})
    start_caption = timeline.get('start_caption', '')
    end_caption = timeline.get('end_caption', '')
    normal_middle_caption = normal_timeline.get('middle_caption', '')
    correct_middle_caption = timeline.get('middle_caption', '')
    
    # 处理caption格式，移除时间戳前缀
    if normal_middle_caption.startswith('[3-7s]'):
        normal_middle_caption = normal_middle_caption[6:].strip()
    if correct_middle_caption.startswith('[3-7s]'):
        correct_middle_caption = correct_middle_caption[6:].strip()
    if start_caption.startswith('[0-3s]'):
        start_caption = start_caption[6:].strip()
    if end_caption.startswith('[7-10s]'):
        end_caption = end_caption[6:].strip()

    # 为caption添加默认值，确保题目不为空
    display_start_caption = start_caption if start_caption else "(No start information available)"
    display_end_caption = end_caption if end_caption else "(No end information available)"
    
    # 自己生成题目
    question = f"[0-3s]: {display_start_caption}; [3-7s]: ???; [7-10s]: {display_end_caption}. What most likely happened in the [3-7s] interval according to the video?"
    
    # 添加反馈信息（如果有）
    feedback_block = ""
    if feedback:
        feedback_block = f"""
        # 🚨 BLIND TESTER FEEDBACK:
        A student guessed the correct answer WITHOUT seeing the video.
        Their reasoning: "{feedback}"
        👉 **CORRECTION TASK:** Regenerate options to defeat this reasoning.
        """
    
    # 构建让AI只生成两个干扰选项的system_prompt
    system_prompt = f"""
    You are an Exam Creator's assistant.
    Task: Generate 2 additional distractor options for a 4-option MCQ.
    GOAL: Anti-Blind-Guessing.
    
    # EXISTING INFORMATION:
    - Question prompt: {question}
    - One existing option (distractor): {normal_middle_caption}
    - One existing option (correct answer): {correct_middle_caption}
    - Object: {obj}
    - Scenario type: {s_type}
    
    # DISTRACTOR GENERATION STRATEGY (CRITICAL):
    - Generate 2 more highly misleading distractors that make it impossible for Video VLMs relying solely on text priors to answer correctly.
    - Your generated options should NOT be identical to the existing distractor or correct answer.
    - Options should be plausible but incorrect based on normal expectations.
    
    # 🎨 STYLE & TONE MATCHING (CRITICAL):
    1. **MATCH LENGTH:** Your options must have roughly the same word count as the existing options.
    2. **MATCH DETAIL LEVEL:** Maintain similar level of detail in your descriptions.
    
    {feedback_block}

    OUTPUT JSON (RAW ONLY, NO MARKDOWN): {{ "distractor1": "...", "distractor2": "..." }}
    """
    
    user_content = f"Video Prompt: {final_prompt}"

    try:
        print(f"      ⏳ Quizmaster API Requesting...") # Debug Log
        response = client_gemini.chat.completions.create(
            model=DIRECTOR_MODEL, 
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_content}
            ], 
            temperature=1.0, 
            response_format={"type": "json_object"}
        )
        
        # 解析AI生成的两个干扰选项
        ai_distractors = clean_and_parse_json(response.choices[0].message.content)
        
        # 构建所有选项列表
        all_options = [
            normal_middle_caption,  # 已有的干扰选项
            correct_middle_caption, # 已有的正确选项
            ai_distractors.get("distractor1", ""), # AI生成的干扰选项1
            ai_distractors.get("distractor2", "")  # AI生成的干扰选项2
        ]
        
        # 过滤空选项并确保有足够的选项
        all_options = [opt for opt in all_options if opt.strip()]
        if len(all_options) < 4:
            # 如果选项不足，补充一些基本选项
            for i in range(4 - len(all_options)):
                all_options.append(f"Something else happened involving {obj}.")
        
        # 随机打乱选项顺序并为正确选项分配字母
        shuffled_options = all_options.copy()
        random.shuffle(shuffled_options)
        
        # 构建选项字典
        options = {}
        correct_answer = None
        for i, option in enumerate(shuffled_options):
            option_letter = chr(65 + i)  # A, B, C, D
            options[option_letter] = option
            # 找出正确答案的字母
            if option == correct_middle_caption:
                correct_answer = option_letter
        
        # 确保正确答案被正确标记
        if not correct_answer:
            # 如果找不到，默认为第一个选项
            correct_answer = "A"
        
        # 返回完整的MCQ数据
        mcq_data = {
            "question": question,
            "options": options,
            "correct_answer": correct_answer
        }
        
        return mcq_data
        
    except Exception as e:
        print(f"      ❌ Quizmaster Error: {e}")
        return {"is_server_error": True}


# ================= Agent 4: Blind Tester =================

def blind_tester_agent(mcq_data):
    options_text_block = "\n".join([f"{k}: {v}" for k, v in mcq_data['options'].items()])
    system_prompt = 'You are a student guessing the answer without seeing the video. Output JSON (RAW ONLY, NO MARKDOWN): { "guessed_option": "A", "confidence": "High/Low", "reasoning": "..." }'
    user_input = f"Question: {mcq_data['question']}\nOptions:\n{options_text_block}"
    
    try:
        print(f"      ⏳ Blind Tester API Requesting...") # Debug Log
        response = client_deepseek.chat.completions.create(
            model=HELPER_MODEL, 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], 
            temperature=0.6, 
            response_format={"type": "json_object"},
            timeout=60 
        )
        return clean_and_parse_json(response.choices[0].message.content) # 🟢 使用清洗函数
    except Exception as e:
        print(f"      ❌ Blind Tester Error: {e}")
        return {"is_server_error": True}


def synthesize_final_prompt(director_data, s_type):
    timeline = director_data.get('timeline', {})
    # 确保timeline中包含必要的字段
    if not timeline.get('start_caption'):
        timeline['start_caption'] = "(No start caption available)"
    if not timeline.get('middle_caption'):
        timeline['middle_caption'] = "(No middle caption available)"
    if not timeline.get('end_caption'):
        timeline['end_caption'] = "(No end caption available)"
    
    # For Practical and surprise, we still want a "Realistic Style" prefix so the model generates photorealistic video
    prefix = "Realistic Style: " if s_type in ["Practical Scenario", "surprise Scenario"] else "Surreal Style: "
    return (
        f"{prefix}- Start (0s): {timeline.get('start_caption')}\n"
        f"- Action (Mid): {timeline.get('middle_caption')}\n"
        f"- Result (End): {timeline.get('end_caption')}"
    )

# ================= Data Merging Function =================

def merge_timeline_data(evaluated_data, surprise_data):
    """
    Merge timeline data from surprise_data into evaluated_data based on object_name
    """
    # Create a dictionary for quick lookup of surprise data by object_name
    surprise_dict = {}
    for item in surprise_data:
        object_name = item.get('constraints', {}).get('keyword')
        if object_name:
            surprise_dict[object_name] = item
    
    # Merge timeline data into evaluated items
    merged_data = []
    for evaluated_item in evaluated_data:
        object_name = evaluated_item.get('object_name')
        if object_name and object_name in surprise_dict:
            # Add timeline and normal_timeline from surprise data
            evaluated_item['timeline'] = surprise_dict[object_name].get('timeline', {})
            evaluated_item['normal_timeline'] = surprise_dict[object_name].get('normal_timeline', {})
            # Add constraints for consistency with existing code
            evaluated_item['constraints'] = surprise_dict[object_name].get('constraints', {})
            print(f"📊 成功合并 {object_name} 的时间线数据")
        merged_data.append(evaluated_item)
    
    return merged_data

# ================= Main Function =================

def main(evaluated_file, surprise_file, output_file=None):
    # Set default output file if not provided - no timestamp
    if not output_file:
        output_file = f"mcq_blind_test_results_{VERSION}.json"
    
    print(f"🚀 启动 MCQ 出题与盲测工具 {VERSION}...")
    
    # Read evaluated JSON file
    if not os.path.exists(evaluated_file):
        print(f"❌ 评估文件不存在: {evaluated_file}")
        return
    
    # Read surprise JSON file
    if not os.path.exists(surprise_file):
        print(f"❌ 时间线文件不存在: {surprise_file}")
        return
    
    try:
        with open(evaluated_file, "r", encoding='utf-8') as f:
            evaluated_data = json.load(f)
        print(f"📊 读取评估文件: {evaluated_file}")
        print(f"   共包含 {len(evaluated_data)} 个条目")
    except json.JSONDecodeError as e:
        print(f"❌ 解析评估文件失败: {e}")
        return
    
    try:
        with open(surprise_file, "r", encoding='utf-8') as f:
            surprise_data = json.load(f)
        print(f"📊 读取时间线文件: {surprise_file}")
        print(f"   共包含 {len(surprise_data)} 个条目")
    except json.JSONDecodeError as e:
        print(f"❌ 解析时间线文件失败: {e}")
        return
    
    # Merge the two datasets
    director_data_list = merge_timeline_data(evaluated_data, surprise_data)
    print(f"🔄 合并后的数据条目数: {len(director_data_list)}")
    
    # Load existing results if output file exists
    processed_entries = set()
    results = []
    blind_test_passed = 0
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding='utf-8') as f:
                results = json.load(f)
            print(f"📈 加载现有结果文件: {output_file}")
            print(f"   已处理 {len(results)} 个条目")
            
            # Extract processed entries using constraints as unique identifier
            for result in results:
                constraints = result.get('constraints', {})
                if constraints:
                    # Create a unique identifier for the entry
                    entry_id = f"{constraints.get('keyword')}_{constraints.get('type')}_{constraints.get('mode')}"
                    processed_entries.add(entry_id)
            print(f"   跳过已处理的条目...")
        except json.JSONDecodeError:
            print(f"⚠️  现有结果文件格式错误，将创建新文件")
            results = []
    
    for idx, entry in enumerate(director_data_list):
        print(f"\n🔄 处理条目 {idx + 1}/{len(director_data_list)}")
        
        # Extract necessary data from entry
        obj = entry.get('object_name')
        s_type = entry.get('scenario_type')
        constraints = entry.get('constraints', {})
        mode = constraints.get('mode', '')
        director_data = entry  # 直接使用整个条目作为director_data
        
        if not obj or not s_type or not director_data:
            print(f"   ⚠️  条目缺少必要信息，跳过")
            continue
        
        # Check if this entry has already been processed
        entry_id = f"{obj}_{s_type}_{mode}"
        if entry_id in processed_entries:
            print(f"   ⏭️  条目已处理，跳过: {obj} ({s_type})")
            continue
        
        # Filter based on has_valid_video field
        if not entry.get('has_valid_video', False):
            print(f"   ⏭️  条目未通过评估(无有效视频)，跳过: {obj} ({s_type})")
            continue
        
        print(f"   📦 对象: {obj}")
        print(f"   🎬 场景类型: {s_type}")
        
        # Synthesize final prompt
        final_prompt = synthesize_final_prompt(director_data, s_type)
        
        # Generate MCQ and perform blind test with retries
        blind_success = False
        max_total_retries = 3
        total_retry_count = 0
        feedback = None
        
        while total_retry_count < max_total_retries and not blind_success:
            # Generate MCQ
            mcq = None
            mcq_retry_count = 0
            max_mcq_retries = 3
            
            while mcq_retry_count < max_mcq_retries and not mcq:
                raw_mcq = quizmaster_agent(final_prompt, director_data, obj, s_type, feedback=feedback)
                mcq_retry_count += 1
                
                if raw_mcq and raw_mcq.get('is_server_error'):
                    print(f"   ⚠️  Quizmaster API错误，重试中... ({mcq_retry_count}/{max_mcq_retries})")
                    time.sleep(2)
                    continue
                
                mcq = shuffle_mcq_options(raw_mcq)
                if not mcq:
                    print(f"   ⚠️  MCQ格式错误，重试中... ({mcq_retry_count}/{max_mcq_retries})")
                    time.sleep(2)
            
            if not mcq:
                print(f"   ❌ 无法生成有效的MCQ，跳过")
                break
            
            print(f"   ✅ MCQ生成成功")
            
            # Perform blind test
            blind_result = blind_tester_agent(mcq)
            
            if blind_result and blind_result.get('is_server_error'):
                print(f"   ⚠️  Blind Tester API错误，重试中... ({total_retry_count + 1}/{max_total_retries})")
                time.sleep(2)
                total_retry_count += 1
                continue
            
            guess = blind_result.get('guessed_option')
            truth = mcq.get('correct_answer')
            
            if guess and truth and guess != truth:
                print(f"   ✅ 盲测通过: 猜测={guess}, 正确={truth}")
                blind_test_passed += 1
                blind_success = True
            else:
                print(f"   ❌ 盲测失败: 猜测={guess}, 正确={truth}")
                feedback = blind_result.get('reasoning')
                total_retry_count += 1
                print(f"   ⚠️  盲测失败，尝试重新生成MCQ... ({total_retry_count}/{max_total_retries})")
        
        if not blind_success:
            print(f"   ❌ 盲测多次失败，跳过此条目")
            continue
        
        # Add result
        result_entry = {
            "constraints": constraints,
            "director_data": director_data,
            "final_prompt": final_prompt,
            "mcq": mcq,
            "blind_test_log": blind_result,
            "blind_test_passed": True
        }
        
        results.append(result_entry)
        
        # Save intermediate results
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"   💾 结果已保存到: {output_file}")
    
    # Final statistics
    print(f"\n📊 处理完成")
    print(f"   总条目数: {len(director_data_list)}")
    print(f"   成功生成MCQ并通过盲测: {len(results)}")
    print(f"   盲测通过率: {blind_test_passed / len(director_data_list) * 100:.1f}%" if director_data_list else 0)
    print(f"   结果文件: {output_file}")


if __name__ == "__main__":
    import argparse
    import os
    import glob
    
    parser = argparse.ArgumentParser(description="MCQ出题与盲测工具")
    parser.add_argument("-e", "--evaluated", help="评估结果JSON文件路径，默认为physibench_evaluated_v79.json")
    parser.add_argument("-s", "--surprise", help="时间线JSON文件路径，默认为physibench_surprise_v79_unique.json")
    parser.add_argument("-o", "--output", help="输出JSON文件路径(MCQ和盲测结果，默认自动命名)")
    
    args = parser.parse_args()
    
    # 设置默认文件路径
    default_evaluated_file = "physibench_evaluated_v79.json"
    default_surprise_file = "physibench_surprise_v79_unique.json"
    
    # 使用提供的参数或默认值
    evaluated_file = args.evaluated if args.evaluated else default_evaluated_file
    surprise_file = args.surprise if args.surprise else default_surprise_file
    
    # 确保文件存在
    if not os.path.exists(evaluated_file):
        print(f"❌ 默认评估文件不存在: {evaluated_file}")
        print("请使用 -e 参数指定正确的评估文件路径")
        exit(1)
    
    if not os.path.exists(surprise_file):
        print(f"❌ 默认时间线文件不存在: {surprise_file}")
        print("请使用 -s 参数指定正确的时间线文件路径")
        exit(1)
    
    print(f"📄 使用评估文件: {evaluated_file}")
    print(f"📄 使用时间线文件: {surprise_file}")
    
    main(evaluated_file, surprise_file, args.output)