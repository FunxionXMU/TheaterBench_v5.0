import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase4_mcq_blind_test import quizmaster_agent, clean_and_parse_json

class TestModifiedQuizmasterAgent(unittest.TestCase):
    
    def setUp(self):
        # 模拟director_data数据
        self.director_data = {
            'timeline': {
                'start_caption': 'A ball is placed on a table',
                'middle_caption': '[3-7s] The ball rolls off the table',
                'end_caption': 'The ball hits the floor'
            },
            'normal_timeline': {
                'middle_caption': '[3-7s] The ball stays on the table'
            }
        }
        self.final_prompt = "Realistic Style: - Start (0s): A ball is placed on a table\n- Action (Mid): The ball rolls off the table\n- Result (End): The ball hits the floor"
        self.obj = "ball"
        self.s_type = "Practical Scenario"
    
    @patch('mcq_blind_test.client_gemini.chat.completions.create')
    def test_quizmaster_agent_with_mock_api(self, mock_create):
        # 模拟AI响应
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"distractor1": "The ball bounces on the table", "distractor2": "The ball changes color"}'
        mock_create.return_value = mock_response
        
        # 调用函数
        result = quizmaster_agent(self.final_prompt, self.director_data, self.obj, self.s_type)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertFalse(result.get('is_server_error', False))
        
        # 验证题目格式
        self.assertIn('question', result)
        self.assertIn('options', result)
        self.assertIn('correct_answer', result)
        
        # 验证选项数量
        self.assertEqual(len(result['options']), 4)
        
        # 验证正确选项被正确标记
        correct_answer = result['correct_answer']
        self.assertIn(correct_answer, ['A', 'B', 'C', 'D'])
        self.assertEqual(result['options'][correct_answer], 'The ball rolls off the table')
        
        # 验证所有选项都存在
        options_texts = list(result['options'].values())
        self.assertIn('The ball stays on the table', options_texts)  # 已有的干扰选项
        self.assertIn('The ball bounces on the table', options_texts)  # AI生成的干扰选项1
        self.assertIn('The ball changes color', options_texts)  # AI生成的干扰选项2
    
    def test_caption_processing(self):
        # 验证时间戳前缀移除功能
        from phase4_mcq_blind_test import quizmaster_agent
        
        # 使用patch避免实际API调用
        with patch('mcq_blind_test.client_gemini.chat.completions.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"distractor1": "", "distractor2": ""}'
            mock_create.return_value = mock_response
            
            result = quizmaster_agent(self.final_prompt, self.director_data, self.obj, self.s_type)
            
            # 验证时间戳已被移除
            options_texts = list(result['options'].values())
            self.assertNotIn('[3-7s]', ' '.join(options_texts))
    
    @patch('mcq_blind_test.client_gemini.chat.completions.create')
    def test_with_feedback(self, mock_create):
        # 测试带有反馈信息的情况
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"distractor1": "The ball is picked up by a hand", "distractor2": "The ball floats in mid-air"}'
        mock_create.return_value = mock_response
        
        feedback = "The ball must roll off because of gravity"
        result = quizmaster_agent(self.final_prompt, self.director_data, self.obj, self.s_type, feedback=feedback)
        
        # 验证仍然生成了有效的MCQ
        self.assertIsNotNone(result)
        self.assertEqual(len(result['options']), 4)
        
    def test_edge_case_empty_options(self):
        # 测试边缘情况：AI返回空选项
        with patch('mcq_blind_test.client_gemini.chat.completions.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"distractor1": "", "distractor2": ""}'
            mock_create.return_value = mock_response
            
            result = quizmaster_agent(self.final_prompt, self.director_data, self.obj, self.s_type)
            
            # 验证补充了默认选项
            self.assertEqual(len(result['options']), 4)
            
if __name__ == '__main__':
    unittest.main()
