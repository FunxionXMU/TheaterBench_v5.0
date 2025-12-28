import os
import json
import time
import random
from openai import OpenAI
from tqdm import tqdm
from phase0_config_data import (
    OBJECT_DICT, client_gemini, client_deepseek,
    DIRECTOR_MODEL, HELPER_MODEL, VERSION,
    TEST_MODE
)

# ================= 核心：单一情景定义 =================

SCENARIO_TYPES = ["Surprise Scenario"]

SCENARIO_POLICIES = {
    "Surprise Scenario": (
        "✅ GOAL: Create a scene with a SURREAL TWIST in the Middle, while Start and End remain PASSIVE.\n"
        "✅ STRUCTURE: Setup (Static) -> Twist (Surreal Action) -> Reaction (Static).\n"
        "❌ AVOID: Action in Start/End phases. The only action must happen in the Middle."
    )
}

COMMON_CHECKLIST = """
    1. **Is Start Passive?** No touching or complex interaction allowed in the first 3 seconds.
    2. **Is Middle Surreal?** The core interaction must be physically impossible or highly unexpected.
    3. **Is End Passive?** Just looking/reacting. No further action.
    4. **Timeline Consistency:** "start_caption" and "end_caption" must be identical between normal_timeline and timeline.
    5. **Only Middle Can Change:** Only "middle_caption" can have differences between normal_timeline and timeline.
"""

# ================= Helper Functions =================

def clean_and_parse_json(content):
    """
    JSON cleaning function to handle extra data and Markdown issues
    """
    if not content: return None
    content = content.strip()
    
    try:
        if "```" in content:
            first_marker = content.find("```")
            last_marker = content.rfind("```")
            if first_marker != -1 and last_marker != -1 and first_marker != last_marker:
                content = content[first_marker + 3 : last_marker].strip()
                content = content.replace("json", "", 1).strip()
        
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
        
        import re
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        return json.loads(content)
    except Exception as e:
        print(f"DEBUG: JSON Parse Error - {e}")
        return None

def synthesize_final_prompt(director_data, s_type):
    timeline = director_data.get('timeline', {})
    prefix = "Realistic Style: "
    return (
        f"{prefix}- {timeline.get('start_caption')}\n"
        f"- {timeline.get('middle_caption')}\n"
        f"- {timeline.get('end_caption')}"
    )

# ================= Agent 1: Director (Strict Phase Control) =================

def director_agent_normal(obj, feedback=None): 
    """
    第一步导演调用：只负责生成normal_timeline
    """
    
    system_prompt = f"""
    You are a Director specializing in normal, realistic scenes.
    Target Object: {obj}
    
    # 🎬 THE "NORMAL SCENE" PROTOCOL
    
    You must construct a realistic, normal interaction scene that is **ENGAGING AND NOT BORING**.
    - The scene should be realistic and plausible
    - The scene should be engaging and not mundane or trivial (e.g., "peeling and eating a banana" is too boring for a banana peel)
    - Focus on interactions that create a natural narrative or have inherent visual interest
    
    **STEP 1: The Normal Draft**
    - Imagine an engaging, normal interaction with the object.
    - *Example (Banana Peel): Man walks on street -> Steps on banana peel and falls down -> Upset expression.*
    - *AVOID: Simply peeling a banana and eating it (too boring).*
    
    # 📝 STRICT TIMELINE RULES (10 Seconds):
    
    1. **[0-3s] START (The Setup - NO CORE INTERACTION):**
       - Show the object and the subject (human/animal).
       - The subject can look at or approach the object, but **MUST NOT TOUCH** or interact with it yet.
       - *Goal: Establish the scene.*
    
    2. **[3-7s] MIDDLE (The Interaction - NORMAL ACTION):**
       - This is where the normal interaction happens.
       - The object should behave in a physically possible, expected way.
       - The interaction should be interesting and not trivial.
    
    3. **[7-10s] END (The Aftermath - REACTION ONLY):**
       - The core action stops.
       - Subject reacts to the normal interaction.
    
    # 💡 ONE-SHOT EXAMPLE:
    **Object:** Banana Peel
    **Normal Thought:** Man walks on street -> Steps on banana peel -> Falls down.
    - *Why this is good:* Creates a classic, visually interesting interaction that everyone recognizes
    - *Why "peeling and eating" is bad:* Too mundane, lacks visual interest and narrative potential
    
    **JSON Output:**
    {{
        "reasoning_trace": "Idea 1: A person peels a banana and starts eating; Idea 2: A person walks on the street, steps on a banana peel, and slips. Comparing the two ideas, Idea 1 is very boring; Idea 2, while ensuring the scene is realistic, is more visually appealing.",
        "synopsis": "A man walking on the street steps on a banana peel and slips, falling down.",
        "normal_timeline": {{
            "start_caption": "[0-3s] A man walks on the street; there is a banana peel lying on the road ahead of him.",
            "middle_caption": "[3-7s] The man doesn't notice the banana peel and steps on it, slipping and falling.",
            "end_caption": "[7-10s] The man gets up, looking confused and staring at the banana peel."
        }}
    }}
    
    # OUTPUT JSON (RAW ONLY):
    """

    user_content = f"Object: {obj}"

    try:
        print(f"      ⏳ Director Normal API Requesting... (Timeout: 60s)") 
        response = client_gemini.chat.completions.create(
            model=DIRECTOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], 
            temperature=0.7,
            response_format={"type": "json_object"},
            timeout=60
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Director Normal Error: {e}")
        return {"is_server_error": True}

def director_agent_surreal(obj, normal_timeline, feedback=None):
    """
    第二步导演调用：负责基于normal_timeline修改middle_caption为超现实内容
    """
    feedback_block = ""
    if feedback:
        feedback_block = f"""
        # 🚨 REVIEWER FEEDBACK (PREVIOUS REJECTION):
        Issue: "{feedback}"
        👉 **CORRECTION:** Ensure Start/End remain identical to normal_timeline, only modify middle_caption to be surreal.
        """

    # 从normal_timeline中提取caption
    start_caption = normal_timeline.get('start_caption', '')
    middle_caption = normal_timeline.get('middle_caption', '')
    end_caption = normal_timeline.get('end_caption', '')

    system_prompt = f"""
    You are a Director specializing in Surreal/Absurd Short Films.
    Target Object: {obj}
    
    # 🎬 THE "SURREAL TWIST" PROTOCOL
    
    You must modify ONLY the MIDDLE phase of an existing timeline to introduce a surreal/impossible element.
    
    **CRITICAL REQUIREMENTS:**
    1. Keep start_caption and end_caption **IDENTICAL** to the original
    2. Only modify middle_caption to be surreal/impossible
    3. The surreal twist should relate to the original normal interaction
    
    **EXISTING NORMAL TIMELINE:**
    {{"start_caption": "{start_caption}", "middle_caption": "{middle_caption}", "end_caption": "{end_caption}"}}
    
    # 💡 TRANSFORMATION EXAMPLE:
    **Normal Middle:** The man doesn't notice the banana peel and steps on it, slipping and falling.
    **Surreal Middle:** The man doesn't notice the banana peel, but as he approaches, the banana peel suddenly moves and actively wraps around his foot, tripping him.
    
    **JSON Output:**
    {{
        "reasoning_trace": "The beginning shows a person and a banana peel, and the ending shows the person getting up, indicating they fell down in the middle. To ensure coherence with the beginning and end while making the scene incredible, I chose to have the banana peel actively wrap around his foot and trip him.",
        "synopsis": "A man walking on the street steps on a banana peel, but instead of a normal slip, the banana peel actively wraps around his foot and trips him.",
        "timeline": {{
            "middle_caption": "[3-7s] The man doesn't notice the banana peel, but as he approaches, the banana peel suddenly moves and actively wraps around his foot, tripping him."
        }}
    }}

    {feedback_block}
    
    # OUTPUT JSON (RAW ONLY):
    """

    user_content = f"Transform only the middle_caption of the timeline for object: {obj}"

    try:
        print(f"      ⏳ Director Surreal API Requesting... (Timeout: 60s)") 
        response = client_gemini.chat.completions.create(
            model=DIRECTOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], 
            temperature=1.3,
            response_format={"type": "json_object"},
            timeout=60
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Director Surreal Error: {e}")
        return {"is_server_error": True}

def director_agent(obj, s_type="Surprise Scenario", feedback=None): 
    
    feedback_block = ""
    if feedback:
        feedback_block = f"""
        # 🚨 REVIEWER FEEDBACK (PREVIOUS REJECTION):
        Issue: "{feedback}"
        👉 **CORRECTION:** Ensure Start/End are PASSIVE and Middle is SURREAL.
        """

    system_prompt = f"""
    You are a Director specializing in Surreal/Absurd Short Films.
    Target Object: {obj}
    
    # 🎬 THE "ISOLATED TWIST" PROTOCOL
    
    You must construct a scene where the "Weirdness" is isolated entirely in the MIDDLE phase.
    
    **STEP 1: The Normal Draft (Mental Scratchpad)**
    - Imagine a normal interaction.
    - *Example (Banana Peel): Man walks on street -> Steps on banana peel and falls down -> Upset expression.*
    
    **STEP 2: The Surreal Injection (MODIFY ONLY THE MIDDLE)**
    - **CRITICAL REQUIREMENT:** ONLY modify the middle_caption (3-7s development stage)
    - Keep Start: Man walks on street (No interaction yet).
    - MODIFY Middle: Instead of normal slipping, the banana peel actively trips the man.
    - Keep End: Man looks upset/confused (No more action).
    
    # 📝 STRICT TIMELINE RULES (10 Seconds):
    
    1. **[0-3s] START (The Setup - NO CORE INTERACTION):**
       - Show the object and the subject (human/animal).
       - The subject can look at or approach the object, but **MUST NOT TOUCH** or interact with it yet.
       - *Goal: Establish the scene.*
    
    2. **[3-7s] MIDDLE (The Twist - CORE INTERACTION):**
       - This is where the interaction happens.
       - **CRITICAL:** Replace the "Normal Interaction" with a **SURREAL/IMPOSSIBLE** event.
       - The object should behave unexpectedly (e.g., explodes into flowers, melts, talks, attacks).
    
    3. **[7-10s] END (The Aftermath - REACTION ONLY):**
       - The core action stops.
    
    # 💡 ONE-SHOT EXAMPLE:
    **Object:** Banana Peel
    **Normal Thought:** Man walks on street -> Steps on banana peel -> Falls down.
    **Surreal Twist:** Man walks on street -> Banana peel actively trips man -> Man looks confused.
    **Important:** ONLY modify the middle_caption (3-7s development stage)
    
    **JSON Output:**
    {{
        "reasoning_trace": "Normal: Man steps on banana peel. Twist: Banana peel actively trips man.",
        "synopsis": "A man walking on the street steps on a banana peel, but instead of a normal slip, the banana peel actively wraps around his foot and trips him.",
        "normal_timeline": {{
            "start_caption": "[0-3s] A man walks on the street; there is a banana peel lying on the road ahead of him.",
            "middle_caption": "[3-7s] The man doesn't notice the banana peel and steps on it, slipping and falling.",
            "end_caption": "[7-10s] The man gets up, looking confused and staring at the banana peel."
        }},
        "timeline": {{
            "start_caption": "[0-3s] A man walks on the street; there is a banana peel lying on the road ahead of him.",
            "middle_caption": "[3-7s] The man doesn't notice the banana peel, but as he approaches, the banana peel suddenly moves and actively wraps around his foot, tripping him.",
            "end_caption": "[7-10s] The man gets up, looking confused and staring at the banana peel."
        }}
    }}

    {feedback_block}
    
    # OUTPUT JSON (RAW ONLY):
    """

    user_content = f"Object: {obj}"

    try:
        print(f"      ⏳ Director API Requesting... (Timeout: 60s)") 
        response = client_gemini.chat.completions.create(
            model=DIRECTOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], 
            temperature=1.3,
            response_format={"type": "json_object"},
            timeout=60
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Director Error: {e}")
        return {"is_server_error": True}

# ================= Agent 2: Reviewer (Structure Enforcer) =================

def reviewer_agent_normal(director_data, obj):
    """
    专门用于审核normal_timeline的审核员
    """
    if not director_data: return {"passed": False, "reason": "No Output"}

    audit_cot = """
    # 🧠 NORMAL TIMELINE STRUCTURE AUDIT:
    
    1. **Check START [0-3s]:** - Does the interaction (touching/action) start here? 
       - **IF YES -> REJECT.** (Must be setup only).
       
    2. **Check MIDDLE [3-7s]:**
       - Is there a NORMAL INTERACTION? (Physically possible, expected behavior)
       - If it's surreal or no interaction happens -> **REJECT**.        
    3. **Check END [7-10s]:**
       - Is there complex action? (e.g., he runs away and calls mom).
       - **IF YES -> REJECT.** (Must be simple reaction/staring).
    
    4. **Check SCENE QUALITY:**
       - Is the scene engaging and not boring?
       - If it's too trivial or mundane -> **REJECT**. (e.g., simply peeling a banana)
    """

    system_prompt = f"""
    You are a Timeline Structure Auditor specialized in NORMAL, REALISTIC scenes.
    
    {audit_cot}
    
    Target Object: {obj}
    OUTPUT JSON (RAW ONLY): {{ "passed": true/false, "reason": "...", "score": 0-10 }}
    """
    
    user_content = f"Script: {json.dumps(director_data.get('normal_timeline', {}))}\nSynopsis: {director_data.get('synopsis')}"

    try:
        print(f"      ⏳ Normal Reviewer API Requesting...")
        response = client_deepseek.chat.completions.create(
            model=HELPER_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], temperature=0.1, response_format={"type": "json_object"},
            timeout=60
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Normal Reviewer API Failed: {e}") 
        return {"passed": False, "reason": "API Error", "is_server_error": True}


def reviewer_agent(director_data, obj, s_type="Surprise Scenario"):
    if not director_data: return {"passed": False, "reason": "No Output"}

    audit_cot = """
    # 🧠 STRUCTURE AUDIT:
    
    1. **Check START [0-3s]:** - Does the interaction (touching/action) start here? 
       - **IF YES -> REJECT.** (Must be setup only).
       
    2. **Check MIDDLE [3-7s]:**
       - Is there a TWIST? (Surreal/Unexpected).
       - If it's just normal action (e.g., he sits on the chair normally) -> **REJECT**.        
    3. **Check END [7-10s]:**
       - Is there complex action? (e.g., he runs away and calls mom).
       - **IF YES -> REJECT.** (Must be simple reaction/staring).
    
    4. **Check TRANSITIONS BETWEEN SEGMENTS:**
       - Is there continuity between START and MIDDLE? (Should follow logically)
       - Is there continuity between MIDDLE and END? (Should follow logically)
       - If transitions are disjointed or unrelated -> **REJECT**.
    """

    system_prompt = f"""
    You are a Timeline Structure Auditor. Enforce the [Setup -> Twist -> Reaction] format.
    
    {audit_cot}
    
    Target Object: {obj}
    OUTPUT JSON (RAW ONLY): {{ "passed": true/false, "reason": "...", "score": 0-10 }}
    """
    
    user_content = f"Script: {json.dumps(director_data.get('timeline', {}))}\nSynopsis: {director_data.get('synopsis')}"

    try:
        print(f"      ⏳ Reviewer API Requesting...")
        response = client_deepseek.chat.completions.create(
            model=HELPER_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], temperature=0.1, response_format={"type": "json_object"},
            timeout=60
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"      ❌ Reviewer API Failed: {e}") 
        return {"passed": False, "reason": "API Error", "is_server_error": True}

# ================= 主程序 =================

def combine_director_data(normal_data, surreal_data):
    """
    合成函数：合并normal_data和surreal_data生成最终的director_data
    """
    if not normal_data or not surreal_data:
        return None
    
    # 合并两个调用的结果
    combined_data = {}
    
    # 从normal_data中获取内容
    combined_data['reasoning_trace'] = normal_data.get('reasoning_trace', '')
    combined_data['synopsis'] = normal_data.get('synopsis', '')
    combined_data['normal_timeline'] = normal_data.get('normal_timeline', {})
    
    # 从surreal_data中获取内容
    if 'reasoning_trace' in surreal_data and surreal_data['reasoning_trace']:
        # 合并reasoning信息
        normal_reasoning = combined_data['reasoning_trace']
        surreal_reasoning = surreal_data['reasoning_trace']
        # 如果normal_reasoning不为空，则添加换行，否则直接使用surreal_reasoning
        if normal_reasoning:
            combined_data['reasoning_trace'] = f"{normal_reasoning}\n{surreal_reasoning}"
        else:
            combined_data['reasoning_trace'] = surreal_reasoning
    
    # 如果surreal_data提供了synopsis，则使用它覆盖normal_data的synopsis
    if 'synopsis' in surreal_data and surreal_data['synopsis']:
        combined_data['synopsis'] = surreal_data['synopsis']
    
    # 创建timeline，使用normal_timeline的start_caption和end_caption，以及surreal_data的middle_caption
    normal_timeline = combined_data['normal_timeline']
    surreal_timeline = surreal_data.get('timeline', {})
    
    combined_data['timeline'] = {
        'start_caption': normal_timeline.get('start_caption', ''),
        'middle_caption': surreal_timeline.get('middle_caption', ''),
        'end_caption': normal_timeline.get('end_caption', '')
    }
    
    return combined_data

def main():
    final_dataset = []
    output_filename = f"physibench_surprise_{VERSION}.json"
    
    print(f"🚀 启动 PIPELINE {VERSION} (Surreal Twist Mode)...")
    
    # --- 断点续传 ---
    processed_objects = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r", encoding='utf-8') as f:
                final_dataset = json.load(f)
                for entry in final_dataset:
                    k = entry['constraints']['keyword']
                    processed_objects.add(k)
            print(f"🔄 从文件加载了 {len(final_dataset)} 条已有数据。")
        except:
            final_dataset = []

    s_type = "Surprise Scenario"

    # 选择物体
    while True:
        try:
            num_objects = input(f"🔢 处理多少个物体？ (1-{len(OBJECT_DICT)}, Enter=全部): ").strip()
            if not num_objects:
                num_objects = len(OBJECT_DICT)
            else:
                num_objects = int(num_objects)
                num_objects = max(1, min(num_objects, len(OBJECT_DICT)))
            break
        except ValueError:
            print("❌ 无效输入")

    remaining_objects = [obj for obj in OBJECT_DICT if obj not in processed_objects]
    random.shuffle(remaining_objects)
    objects_to_process = remaining_objects[:num_objects]
    
    print(f"📋 目标: {len(objects_to_process)} 个物体 -> {output_filename}")
    
    progress_bar = tqdm(total=len(objects_to_process))
    
    for obj in objects_to_process:
        progress_bar.set_postfix({'Obj': obj})
        print(f"\n📦 Processing: {obj}")
        
        # 定义变量，避免UnboundLocalError
        normal_feedback = None
        dir_feedback = None
        
        # --- Normal导演-审核循环 ---
        normal_data = None
        normal_success = False
        MAX_NORMAL_RETRIES = 2
        
        for normal_attempt in range(MAX_NORMAL_RETRIES + 1):
            if normal_attempt > 0:
                print(f"   🔄 Normal Retry #{normal_attempt}...")
            
            # 生成normal_timeline
            normal_data = director_agent_normal(obj, feedback=normal_feedback)
            if not normal_data or normal_data.get('is_server_error'):
                # 如果是服务器错误，继续重试
                if normal_attempt < MAX_NORMAL_RETRIES:
                    print(f"   ⏳ 等待重试...")
                    time.sleep(1)
                    continue
                else:
                    print("   ❌ Normal导演多次尝试失败，跳过此物体")
                    break
            
            # 审核normal_timeline
            normal_audit = reviewer_agent_normal(normal_data, obj)
            if normal_audit.get('is_server_error'):
                # 简单的重试逻辑
                time.sleep(1)
                normal_audit = reviewer_agent_normal(normal_data, obj)
            
            if normal_audit.get('passed'):
                print(f"   ✅ Normal Passed (Score: {normal_audit.get('score')})")
                normal_success = True
                break
            else:
                print(f"   ❌ Normal Rejected: {normal_audit.get('reason')}")
                normal_feedback = normal_audit.get('reason')
        
        # 如果normal导演失败，跳过此物体
        if not normal_success:
            continue
        
        # --- Surreal导演-审核循环 ---
        success = False
        MAX_SURREAL_RETRIES = 2
        
        for attempt in range(MAX_SURREAL_RETRIES + 1):
            if attempt > 0: print(f"   🔄 Surreal Retry #{attempt}...")
            
            # 修改middle_caption为超现实内容
            surreal_data = director_agent_surreal(obj, normal_data.get('normal_timeline'), feedback=dir_feedback)
            if not surreal_data: break
            if surreal_data.get('is_server_error'): continue
            
            # 合并两个结果
            director_data = combine_director_data(normal_data, surreal_data)
            
            final_prompt = synthesize_final_prompt(director_data, s_type)
            
            # 审核surreal timeline
            audit = reviewer_agent(director_data, obj, s_type)
            if audit.get('is_server_error'):
                 # Simple retry logic for reviewer error
                 time.sleep(1)
                 audit = reviewer_agent(director_data, obj, s_type)
            
            if audit.get('passed'):
                print(f"   ✅ Surreal Passed (Score: {audit.get('score')})")
                
                entry = {
                    "constraints": {'keyword': obj, 'type': s_type},
                    "reasoning_trace": director_data.get('reasoning_trace'),
                    "synopsis": director_data.get('synopsis'),
                    "normal_timeline": director_data.get('normal_timeline'),
                    "timeline": director_data.get('timeline'), 
                    "final_t2v_prompt": final_prompt,
                    "reviewer_audit": audit
                }
                final_dataset.append(entry)
                success = True
                
                # 立即保存
                with open(output_filename, "w", encoding='utf-8') as f:
                    json.dump(final_dataset, f, indent=2, ensure_ascii=False)
                break
            else:
                print(f"   ❌ Surreal Rejected: {audit.get('reason')}")
                dir_feedback = audit.get('reason')
        
        if not success:
            print(f"      💀 Surreal Failed after retries.")
        
        progress_bar.update(1)
        time.sleep(1)
    
    progress_bar.close()
    print(f"\n✅ Done.")

if __name__ == "__main__":
    main()