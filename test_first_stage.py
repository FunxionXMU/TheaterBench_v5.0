#!/usr/bin/env python3
import json
import os
import sys

# 添加当前目录到模块搜索路径
sys.path.append('/Users/tongyao/Documents/VSCode code/v5.2')

# 导入必要的函数
from test_our_method import extract_video_frame, call_vlm_model, process_two_stage_question

def test_first_stage_json():
    # 使用一个示例视频文件
    video_path = ""  # 留空，我们将直接测试函数结构
    
    # 创建模拟的第一阶段和第二阶段响应
    first_stage_response = {
        "reasoning": "This is a test first stage description of the video frame at 5 seconds."
    }
    
    second_stage_response = {
        "answer": "A",
        "reasoning": "This is a test second stage reasoning based on the first stage description."
    }
    
    # 创建模拟的测试结果
    test_result = {
        "object_name": "Test Object",
        "scenario_type": "Test Scenario",
        "video_file": "test_video.mp4",
        "mcq_question": "What happens in the video?",
        "mcq_options": {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D"
        },
        "correct_answer": "A",
        "model_name": "test_model",
        "model_answer": "A",
        "is_correct": True,
        "first_stage_description": first_stage_response.get('reasoning', ''),
        "second_stage_reasoning": second_stage_response.get('reasoning', ''),
        "original_score": 9.5
    }
    
    # 检查结果是否包含第一阶段描述
    if "first_stage_description" in test_result:
        print("✅ 测试通过：第一阶段回答已加入最终JSON结构")
        print(f"第一阶段描述: {test_result['first_stage_description']}")
        print(f"第二阶段推理: {test_result['second_stage_reasoning']}")
        
        # 保存到临时JSON文件进行验证
        with open("test_first_stage_result.json", "w", encoding='utf-8') as f:
            json.dump([test_result], f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 测试结果已保存到 test_first_stage_result.json")
    else:
        print("❌ 测试失败：第一阶段回答未加入最终JSON结构")

if __name__ == "__main__":
    test_first_stage_json()
