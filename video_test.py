import os
import json
import base64
import time
from openai import OpenAI

# ================= 配置部分 =================

# API配置，按模型提供商分类
API_CONFIGS = {
    "aliyun": {
        "api_key": "sk-7e6b3a62b1f64945a5a4a9347afa5c72",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "siliconflow": {
        "api_key": "sk-izrxbwrnxotwsvcngnnmwivxmyqukrivnjcoszzpscmfasjz",
        "base_url": "https://api.siliconflow.cn/v1"
    }
}

# 配置要测试的模型列表
TEST_MODELS = [
    # 阿里云模型
    "qwen-vl-max-latest",
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
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "siliconflow",
    "zai-org/GLM-4.6V": "siliconflow",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "siliconflow"
}

# 默认配置
DEFAULT_VIDEO_PATH = "t2v_videos\\Projector_(投影仪)_SFX_1.mp4"
DEFAULT_QUESTION = "Summarize the video."

# ================= 辅助函数 =================

def compress_video_smart(video_path, target_size_mb=7.0):
    """
    智能压缩视频：如果视频超过目标大小，则通过降低帧率来压缩，
    同时保持画质（CRF 18）不变。
    
    返回: (文件路径, 是否是临时文件)
    """
    if not os.path.exists(video_path):
        return video_path, False

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    
    # 如果小于目标大小，直接返回
    if file_size <= target_size_mb:
        return video_path, False

    print(f"📦 Video {os.path.basename(video_path)} is {file_size:.2f}MB. Compressing to < {target_size_mb}MB...")
    
    temp_path = video_path.replace(".mp4", "_compressed_temp.mp4")
    
    # 策略：降低帧率，但保持高质量
    target_fps = "5" 
    
    # 如果文件特别大 (>30MB)，尝试降低到 3fps
    if file_size > 30:
        target_fps = "3"

    cmd = [
        "ffmpeg", "-y",                # 覆盖输出
        "-i", video_path,              # 输入
        "-r", target_fps,              # 目标帧率
        "-c:v", "libx264",             # 编码器
        "-pix_fmt", "yuv420p",         # 强制使用 yuv420p 像素格式
        "-crf", "18",                  # 恒定质量因子 (18=高画质/视觉无损)
        "-preset", "veryfast",         # 编码速度
        "-an",                         # 移除音频
        temp_path
    ]
    
    try:
        # 执行压缩，静默输出
        import subprocess
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 检查压缩后的大小
        new_size = os.path.getsize(temp_path) / (1024 * 1024)
        
        # 如果还是太大，尝试进一步压缩
        if new_size > target_size_mb:
            print(f"⚠️ Still too large ({new_size:.2f}MB). Re-compressing aggressively...")
            cmd_aggressive = [
                "ffmpeg", "-y", "-i", video_path,
                "-r", "2", "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast", "-an",
                temp_path
            ]
            subprocess.run(cmd_aggressive, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            new_size = os.path.getsize(temp_path) / (1024 * 1024)

        print(f"✅ Compressed to {new_size:.2f}MB (FPS: {target_fps})")
        return temp_path, True
        
    except Exception as e:
        print(f"❌ Compression failed (ffmpeg might be missing): {e}. Using original.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return video_path, False

def encode_video_to_base64(video_path):
    """
    将视频文件编码为base64格式
    """
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        return video_base64
    except FileNotFoundError:
        print(f"❌ Video file not found: {video_path}")
        return None

def call_vlm_model(video_path, question, model_name="qwen3-vl-plus"):
    """
    调用VLM模型来理解视频内容并回答问题
    """
    SUPPORT_JSON_MODELS = ["Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3-vl-plus", "qwen3-vl-flash", "qwen-vl-max-latest"]
    api_provider = MODEL_TO_API.get(model_name, "siliconflow")
    api_config = API_CONFIGS[api_provider]
    
    client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
    
    system_prompt = """
    You are an expert video understanding AI. 
    Analyze the video and answer the following question in detail.
    Include reasoning about what you see in the video.
    """

    user_prompt = question

    # 压缩视频
    compressed_video_path, is_temp_file = compress_video_smart(video_path)
    
    try:
        # 编码视频为base64
        video_base64 = encode_video_to_base64(compressed_video_path)
        if video_base64 is None:
            return {"error": "Failed to encode video"}

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
                print(f"📞 Calling {model_name} ({api_provider})...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"} if model_name in SUPPORT_JSON_MODELS else None,
                    timeout=120
                )
                
                content = response.choices[0].message.content.strip()
                # 移除可能的JSON标记
                content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    # 尝试解析为JSON
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 如果不是JSON，返回纯文本
                    return {"answer": content, "reasoning": "No structured reasoning available"}
                    
            except Exception as e:
                print(f"❌ {model_name} Request Failed: {e}")
                if retry < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {"error": f"Max retries exceeded: {e}"}
                    
    finally:
        # 清理临时文件
        if is_temp_file and os.path.exists(compressed_video_path):
            try:
                os.remove(compressed_video_path)
                print(f"🗑️ Cleaned up temporary file: {compressed_video_path}")
            except OSError:
                pass

# ================= 主程序 =================

def main():
    """
    主程序：测试VLM模型对视频的理解
    """
    print("🎬 视频理解测试工具")
    print("=" * 50)
    
    # 检查默认视频是否存在
    if not os.path.exists(DEFAULT_VIDEO_PATH):
        print(f"❌ 默认视频不存在: {DEFAULT_VIDEO_PATH}")
        print("请确保视频文件在指定路径下。")
        return
    
    print(f"📹 使用视频: {DEFAULT_VIDEO_PATH}")
    print(f"❓ 默认问题: {DEFAULT_QUESTION}")
    print("=" * 50)
    
    # 获取用户输入的问题
    user_question = input("请输入您的问题（直接回车使用默认问题）: ").strip()
    if not user_question:
        user_question = DEFAULT_QUESTION
    
    print(f"\n💬 问题: {user_question}")
    
    # 选择模型
    print("\n🤖 可用模型:")
    for i, model in enumerate(TEST_MODELS, 1):
        provider = MODEL_TO_API.get(model, "siliconflow")
        print(f"   {i}. {model} ({provider})")
    
    try:
        model_choice = input("请选择模型编号（直接回车使用默认模型）: ").strip()
        if model_choice:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(TEST_MODELS):
                selected_model = TEST_MODELS[model_index]
            else:
                print("⚠️ 无效的选择，使用默认模型")
                selected_model = "qwen3-vl-plus"
        else:
            selected_model = "qwen3-vl-plus"
    except ValueError:
        print("⚠️ 无效的输入，使用默认模型")
        selected_model = "qwen3-vl-plus"
    
    print(f"\n✅ 选择的模型: {selected_model}")
    
    # 调用模型
    print("\n" + "=" * 50)
    print("🔄 正在分析视频...")
    start_time = time.time()
    
    result = call_vlm_model(DEFAULT_VIDEO_PATH, user_question, selected_model)
    
    end_time = time.time()
    print(f"⏱️ 分析完成，耗时: {end_time - start_time:.2f}秒")
    print("=" * 50)
    
    # 显示结果
    print("\n📊 分析结果:")
    print("=" * 50)
    
    if "error" in result:
        print(f"❌ 错误: {result['error']}")
    else:
        # 显示所有可用的结果字段
        if "answer" in result:
            print(f"💡 回答: {result['answer']}")
        if "reasoning" in result:
            print(f"🤔 推理: {result['reasoning']}")
        # 处理不同模型返回的不同格式
        if "content" in result:
            print(f"💡 回答: {result['content']}")
        if "analysis" in result:
            print(f"🤔 分析: {result['analysis']}")
        if "result" in result:
            print(f"📋 结果: {result['result']}")
        if "explanation" in result:
            print(f"📝 解释: {result['explanation']}")
        
        # 如果没有以上字段，打印原始结果
        if not any(key in result for key in ["answer", "reasoning", "content", "analysis", "result", "explanation"]):
            print("📋 原始结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")

if __name__ == "__main__":
    main()