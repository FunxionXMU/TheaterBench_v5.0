#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的quizmaster_agent函数，验证题目生成和选项组合逻辑
"""

import os
import sys
import json
import random
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要测试的函数
from mcq_blind_test import quizmaster_agent, clean_and_parse_json

def test_quizmaster_agent():
    """测试quizmaster_agent函数的核心功能"""
    print("开始测试quizmaster_agent函数...")
    
    # 模拟测试数据
    final_prompt = "A man encounters a banana peel on the street"
    obj = "banana peel"
    s_type = "anti_text_prior"
    
    # 模拟director_data，包含timeline和normal_timeline
    director_data = {
        "timeline": {
            "start_caption": "A man is walking on the street, there's a banana peel on the road ahead of him",
            "middle_caption": "[3-7s]: The man didn't notice the banana peel, but the banana peel actively tripped him",
            "end_caption": "The man gets up and looks at the banana peel confusedly"
        },
        "normal_timeline": {
            "middle_caption": "[3-7s]: The man didn't notice the banana peel under his feet, stepped on it, and slipped"
        }
    }
    
    # 模拟AI返回的两个干扰选项
    mock_ai_response = {
        "distractor1": "The man noticed the banana peel under his feet, went around it, but slipped",
        "distractor2": "The man wanted to pick up the banana peel, but the banana peel ran away on its own"
    }
    
    # 使用patch模拟API调用
    with patch('mcq_blind_test.client_gemini.chat.completions.create') as mock_create:
        # 配置mock返回值
        mock_message = MagicMock()
        mock_message.content = json.dumps(mock_ai_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        # 固定随机数种子，确保测试结果可重复
        random.seed(42)
        
        # 调用被测试函数
        result = quizmaster_agent(final_prompt, director_data, obj, s_type)
        
        # 验证结果
        print("\n测试结果验证:")
        print(f"1. 是否返回有效数据: {'是' if isinstance(result, dict) and 'is_server_error' not in result else '否'}")
        
        # 检查必要的字段是否存在
        assert 'question' in result, "缺少'question'字段"
        assert 'options' in result, "缺少'options'字段"
        assert 'correct_answer' in result, "缺少'correct_answer'字段"
        print("2. 必要字段验证: 通过")
        
        # 检查题目格式
        expected_question_start = "[0-3s]: A man is walking on the street, there's a banana peel on the road ahead of him; "
        expected_question_end = " What most likely happened in the [3-7s] interval?"
        assert result['question'].startswith(expected_question_start), "题目格式不正确"
        assert result['question'].endswith(expected_question_end), "题目格式不正确"
        print("3. 题目格式验证: 通过")
        
        # 检查选项数量
        assert len(result['options']) == 4, f"选项数量应该是4个，实际是{len(result['options'])}个"
        print("4. 选项数量验证: 通过")
        
        # 检查正确选项是否被正确包含和标记
        correct_answer_text = "The man didn't notice the banana peel, but the banana peel actively tripped him"
        correct_answer_letter = result['correct_answer']
        assert correct_answer_letter in result['options'], f"正确答案字母 {correct_answer_letter} 不在选项中"
        assert result['options'][correct_answer_letter] == correct_answer_text, "正确选项内容不匹配"
        print("5. 正确选项验证: 通过")
        
        # 检查normal_timeline的middle_caption是否作为干扰选项包含在其中
        normal_distractor_text = "The man didn't notice the banana peel under his feet, stepped on it, and slipped"
        assert normal_distractor_text in result['options'].values(), "normal_timeline的middle_caption未作为干扰选项包含"
        print("6. Normal timeline干扰选项验证: 通过")
        
        # 检查AI生成的两个干扰选项是否包含
        ai_distractor1_found = mock_ai_response["distractor1"] in result['options'].values()
        ai_distractor2_found = mock_ai_response["distractor2"] in result['options'].values()
        assert ai_distractor1_found, "AI生成的干扰选项1未包含"
        assert ai_distractor2_found, "AI生成的干扰选项2未包含"
        print("7. AI生成干扰选项验证: 通过")
        
        # 打印最终生成的MCQ内容用于检查
        print("\n生成的MCQ内容:")
        print(f"题目: {result['question']}")
        print("选项:")
        for letter, option in result['options'].items():
            is_correct = " (正确答案)" if letter == correct_answer_letter else ""
            print(f"  {letter}) {option}{is_correct}")
        
        print("\n🎉 所有测试通过！quizmaster_agent函数工作正常")
        return True

def test_edge_cases():
    """测试边缘情况"""
    print("\n开始测试边缘情况...")
    
    # 测试1: 缺少某些字段的情况
    final_prompt = "Test prompt"
    obj = "test object"
    s_type = "test type"
    
    # 缺少middle_caption的情况
    director_data_minimal = {
        "timeline": {
            "start_caption": "Start",
            "end_caption": "End"
        },
        "normal_timeline": {}
    }
    
    with patch('mcq_blind_test.client_gemini.chat.completions.create') as mock_create:
        # 配置mock返回空的干扰选项
        mock_message = MagicMock()
        mock_message.content = json.dumps({"distractor1": "", "distractor2": ""})
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        # 调用函数
        result = quizmaster_agent(final_prompt, director_data_minimal, obj, s_type)
        
        # 验证结果是否包含基本结构
        assert 'question' in result
        assert 'options' in result
        assert 'correct_answer' in result
        assert len(result['options']) == 4
        print("✅ 边缘情况1: 缺少字段时能正常处理")
    
    print("🎉 所有边缘情况测试通过！")
    return True

if __name__ == "__main__":
    try:
        print("==================================================")
        print("🎯 quizmaster_agent函数测试")
        print("==================================================")
        
        # 运行主要测试
        main_test_passed = test_quizmaster_agent()
        
        # 运行边缘情况测试
        edge_test_passed = test_edge_cases()
        
        if main_test_passed and edge_test_passed:
            print("\n==================================================")
            print("✅ 所有测试成功通过！")
            print("✅ quizmaster_agent函数现在可以自己生成题目并组合所有选项")
            print("✅ 正确选项和normal_timeline的干扰选项都直接从数据中提取")
            print("✅ AI只负责生成两个额外的干扰选项")
            print("==================================================")
            sys.exit(0)
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
