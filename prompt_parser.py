import json

# JSONL文件路径
jsonl_file = "c:/Users/21498/OneDrive/homewrok/test_sora/batch_TyCxmOOhfVgfpzQAhfHFWaZolrAOiFnw.jsonl"

def get_prompts(file_path, count=3):
    """
    从JSONL文件中提取指定数量的prompt
    
    Args:
        file_path: JSONL文件路径
        count: 要提取的prompt数量，默认为3
        
    Returns:
        包含指定数量prompt的列表，如果出错返回空列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取第一行
            first_line = f.readline().strip()
            
            if not first_line:
                print("文件为空或第一行为空")
                return []
            
            # 解析第一行的JSON
            data = json.loads(first_line)
            
            # 提取prompt数组
            content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # 移除代码块标记
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # 解析prompt数组
            prompts = json.loads(content)
            
            # 返回指定数量的prompt
            return prompts[:count]
                
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    
    return []

# 保持向后兼容性
def get_first_three_prompts(file_path):
    """
    从JSONL文件中提取前三个prompt（向后兼容函数）
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含前三个prompt的列表，如果出错返回空列表
    """
    return get_prompts(file_path, count=3)

def parse_and_print_prompts(file_path, count=3):
    """
    解析并打印指定数量的prompt
    
    Args:
        file_path: JSONL文件路径
        count: 要打印的prompt数量，默认为3
    """
    prompts = get_prompts(file_path, count)
    
    # 打印prompt
    if prompts:
        print(f"前{len(prompts)}个prompt:")
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt}")

# 保持向后兼容性
def parse_and_print_first_three_prompts(file_path):
    """
    解析并打印前三个prompt（向后兼容函数）
    
    Args:
        file_path: JSONL文件路径
    """
    parse_and_print_prompts(file_path, count=3)

if __name__ == "__main__":
    parse_and_print_first_three_prompts(jsonl_file)
