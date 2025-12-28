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
    prefix = "Surreal Style: "
    return (
        f"{prefix}- [0-3.0 s]: {timeline.get('start_caption')}\n"
        f"- [3.0-7.0 s]: {timeline.get('middle_caption')}\n"
        f"- [7.0-10.0 s]: {timeline.get('end_caption')}"
    )

# ================= Agent 1: Director (Strict Phase Control) =================

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
        "reasoning_trace": "Normal: Man steps on banana peel. Twist: Banana peel actively trips man. Start: Walking on street. Middle: Active tripping. End: Confused expression.",
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
        
        # --- Loop ---
        MAX_RETRIES = 2
        dir_feedback = None
        success = False
        
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0: print(f"   🔄 Retry #{attempt}...")
            
            # Director
            director_data = director_agent(obj, s_type, feedback=dir_feedback)
            if not director_data: break
            if director_data.get('is_server_error'): continue
            
            final_prompt = synthesize_final_prompt(director_data, s_type)
            
            # Reviewer
            audit = reviewer_agent(director_data, obj, s_type)
            if audit.get('is_server_error'):
                 # Simple retry logic for reviewer error
                 time.sleep(1)
                 audit = reviewer_agent(director_data, obj, s_type)
            
            if audit.get('passed'):
                print(f"   ✅ Passed (Score: {audit.get('score')})")
                
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
                
                # Save immediately
                with open(output_filename, "w", encoding='utf-8') as f:
                    json.dump(final_dataset, f, indent=2, ensure_ascii=False)
                break
            else:
                print(f"   ❌ Rejected: {audit.get('reason')}")
                dir_feedback = audit.get('reason')
        
        if not success:
            print(f"      💀 Failed after retries.")
        
        progress_bar.update(1)
        time.sleep(1)
    
    progress_bar.close()
    print(f"\n✅ Done.")

if __name__ == "__main__":
    main()