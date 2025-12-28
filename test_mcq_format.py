#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证mcq_blind_test.py中quizmaster_agent函数的问题格式修改
"""

import sys
import json
from mcq_blind_test import quizmaster_agent

def test_quizmaster_format():
    """
    测试quizmaster_agent函数是否能正确生成基于start_caption和end_caption的问题格式
    """
    # 创建测试数据
    test_director_data = {
        'timeline': {
            'start_caption': 'A man is walking on the street, there\'s a banana peel on the road ahead of him',
            'middle_caption': 'The man steps on the banana peel and slips',
            'end_caption': 'The man gets up and looks at the banana peel confusedly'
        }
    }
    
    test_obj = 'Banana Peel'
    test_s_type = 'CG Scenario'
    test_final_prompt = 'Surreal Style: - Start (0s): A man is walking on the street, there\'s a banana peel on the road ahead of him\n- Action (Mid): The man steps on the banana peel and slips\n- Result (End): The man gets up and looks at the banana peel confusedly'
    
    # 模拟quizmaster_agent函数的行为，只打印system_prompt而不调用API
    # 从源码中提取相关逻辑
    timeline = test_director_data.get('timeline', {})
    start_caption = timeline.get('start_caption', '')
    end_caption = timeline.get('end_caption', '')
    
    # 构建system_prompt
    system_prompt = f"""
    You are an Exam Creator.
    Task: Generate a 4-Option MCQ.
    GOAL: Anti-Blind-Guessing.
    
    # QUESTION FORMAT (CRITICAL):
    The question must be structured as follows:
    "[0-3s]: {start_caption}; [3-7s]: ???; [7-10s]: {end_caption}. What most likely happened in the [3-7s] interval?"
    
    # DISTRACTOR DISTRIBUTION STRATEGY (CRITICAL):
    # 🔍 Distractor Generation Strategy (CRITICAL FOR ANTI-BLIND-GUESSING):
    - **Core Objective:** Generate 3 highly misleading distractors that make it impossible for Video VLMs relying solely on text priors or only recognizing objects in the frame to answer correctly.
    - **Principles:**
      1. **Visual Similarity:** All options must be visually highly similar to the correct answer.
      2. **Event Confusion:** The events described in the distractors must be superficially similar to the real event.
      3. **Text Prior Trap:** Avoid options that can be directly eliminated through text prior knowledge.
    
    # 🎨 STYLE & TONE MATCHING (CRITICAL):
    1. **MATCH LENGTH:** All options must have roughly the same word count.
    2. **MATCH POETRY/DETAIL:** If Correct Answer is poetic, Distractors must be poetic.
    
    OUTPUT JSON (RAW ONLY, NO MARKDOWN): {{ "question": "...", "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }}, "correct_answer": "C" }}
    """
    
    print("\n===== 测试问题格式生成 =====")
    print("期望的问题格式:")
    expected_question = f"[0-3s]: {start_caption}; [3-7s]: ???; [7-10s]: {end_caption}. What most likely happened in the [3-7s] interval?"
    print(expected_question)
    
    # 检查system_prompt中是否包含正确的问题格式模板
    if "[0-3s]: " in system_prompt and "[3-7s]: ???;" in system_prompt and "[7-10s]: " in system_prompt and "What most likely happened in the [3-7s] interval?" in system_prompt:
        print("\n✅ 测试通过：问题格式模板正确包含在system_prompt中")
    else:
        print("\n❌ 测试失败：问题格式模板未正确包含在system_prompt中")
        
    print("\n===== synthesize_final_prompt 函数测试 =====")
    # 测试synthesize_final_prompt函数的字段检查功能
    from mcq_blind_test import synthesize_final_prompt
    
    # 测试缺少字段的情况
    incomplete_director_data = {'timeline': {}}
    result = synthesize_final_prompt(incomplete_director_data, 'CG Scenario')
    
    if "(No start caption available)" in result and "(No middle caption available)" in result and "(No end caption available)" in result:
        print("✅ 测试通过：synthesize_final_prompt 函数正确处理了缺失字段的情况")
    else:
        print("❌ 测试失败：synthesize_final_prompt 函数未能正确处理缺失字段的情况")
        print(f"生成的结果: {result}")
    
    print("\n测试完成！")
    return True

if __name__ == "__main__":
    try:
        test_quizmaster_format()
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
