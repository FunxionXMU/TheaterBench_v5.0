import streamlit as st
import os
import json
import base64
import re
import time
from prettytable import PrettyTable
from openai import OpenAI

# streamlit run human_video_understanding_test.py

# ================= 配置部分 =================
VIDEO_DIR = "t2v_videos"

# API配置，用于翻译功能
API_CONFIGS = {
    "siliconflow": {
        "api_key": "sk-izrxbwrnxotwsvcngnnmwivxmyqukrivnjcoszzpscmfasjz",
        "base_url": "https://api.siliconflow.cn/v1"
    }
}

# 初始化翻译模型客户端
translation_client = OpenAI(
    api_key=API_CONFIGS["siliconflow"]["api_key"],
    base_url=API_CONFIGS["siliconflow"]["base_url"]
)

# ================= 辅助函数 =================
def load_data():
    """加载评估数据和MCQ盲测结果"""
    eval_files = [f for f in os.listdir('.') if f.startswith('physibench_evaluated_v') and f.endswith('.json')]
    eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not eval_files:
        return None, None, None, None
    
    eval_file = eval_files[0]
    version = re.search(r'v(\d+)', eval_file).group(1)
    
    # 查找所有可能的MCQ盲测结果文件
    mcq_files = [f for f in os.listdir('.') if f.startswith('mcq_blind_test_results_') and f.endswith('.json')]
    
    if not mcq_files:
        return None, None, None, "No MCQ blind test results files found"
    
    # 提取每个文件的版本号并选择匹配的文件
    matched_mcq_files = []
    for f in mcq_files:
        match = re.search(r'v(\d+)', f)
        if match and match.group(1) == version:
            matched_mcq_files.append(f)
    
    if not matched_mcq_files:
        return None, None, None, f"No MCQ blind test results files found for version {version}"
    
    # 选择最新的匹配文件
    mcq_file = max(matched_mcq_files, key=os.path.getmtime)
    
    with open(eval_file, "r", encoding='utf-8') as f:
        eval_results = json.load(f)
    with open(mcq_file, "r", encoding='utf-8') as f:
        mcq_results = json.load(f)
    
    # 创建MCQ结果索引
    mcq_index = {f"{r['director_data']['object_name']}_{r['director_data']['scenario_type']}": r for r in mcq_results}
    
    # 筛选已评估的视频
    evaluated_videos = []
    for eval_entry in eval_results:
        object_name = eval_entry['object_name']
        scenario_type = eval_entry['scenario_type']
        key = f"{object_name}_{scenario_type}"
        
        # 检查是否有对应的MCQ结果
        if key in mcq_index and eval_entry['status'] == 'evaluated' and eval_entry['best_video'] is not None:
            mcq_entry = mcq_index[key]
            mcq = mcq_entry['mcq']
            
            evaluated_videos.append({
                'id': key,
                'object_name': object_name,
                'scenario_type': scenario_type,
                'video_file': eval_entry['best_video'],
                'question': mcq['question'],
                'options': mcq['options'],
                'correct_answer': mcq['correct_answer'],
                'video_path': os.path.join(VIDEO_DIR, eval_entry['best_video'])
            })
    
    return evaluated_videos, version, eval_file, None

def get_video_html(video_path, width=600, height=400):
    """生成视频播放的HTML"""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    
    return f"""
    <video width="{width}" height="{height}" controls>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """

def save_results(results, version):
    """保存测试结果"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"human_video_test_results_v{version}_{timestamp}.json"
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_file

def generate_statistics(results):
    """生成统计信息"""
    if not results:
        return "没有测试结果可统计"
    
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # 按场景统计
    scenario_stats = {}
    for r in results:
        scenario = r['scenario_type']
        if scenario not in scenario_stats:
            scenario_stats[scenario] = {'correct': 0, 'total': 0}
        
        scenario_stats[scenario]['total'] += 1
        if r['is_correct']:
            scenario_stats[scenario]['correct'] += 1
    
    # 生成表格
    table = PrettyTable()
    table.field_names = ["场景类型", "正确数", "总数", "正确率"]
    
    for scenario, stats in scenario_stats.items():
        scenario_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        table.add_row([
            scenario,
            stats['correct'],
            stats['total'],
            f"{scenario_accuracy:.1f}%"
        ])
    
    # 添加总计行
    table.add_row([
        "总计",
        correct_answers,
        total_questions,
        f"{accuracy:.1f}%"
    ])
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'scenario_stats': scenario_stats,
        'table': str(table)
    }

def translate_text(text):
    """将英文文本翻译成中文，同时展示中英文"""
    try:
        response = translation_client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的翻译助手，请将下面的英文问题和选项翻译成中文，并保持原文格式。翻译时要保持专业术语的准确性，确保中文翻译易于理解。"
                },
                {
                    "role": "user",
                    "content": f"请翻译以下内容：\n{text}"
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        translated_text = response.choices[0].message.content.strip()
        return {
            "original": text,
            "translated": translated_text
        }
    except Exception as e:
        print(f"翻译失败: {e}")
        return {
            "original": text,
            "translated": "翻译失败，请检查API连接"
        }

# ================= 主应用 ================= 
def main():
    # 配置页面设置（在任何渲染前调用）
    st.set_page_config(
        page_title="视频理解人类测试",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态（在任何渲染前初始化）
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'selected_answer' not in st.session_state:
        st.session_state.selected_answer = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'evaluated_videos' not in st.session_state:
        st.session_state.evaluated_videos = None
    if 'version' not in st.session_state:
        st.session_state.version = None
    if 'translations' not in st.session_state:
        st.session_state.translations = {}  # 存储翻译结果，避免重复翻译
    
    # 加载数据（仅在首次运行时加载）
    if not st.session_state.data_loaded:
        with st.spinner("加载测试数据..."):
            evaluated_videos, version, eval_file, error = load_data()
            
        if error:
            st.error(f"❌ 数据加载失败: {error}")
            return
        
        if not evaluated_videos:
            st.error("❌ 没有找到已评估的视频数据")
            return
        
        st.session_state.evaluated_videos = evaluated_videos
        st.session_state.version = version
        st.session_state.data_loaded = True
    
    # 页面标题（仅渲染一次）
    st.markdown("## 🎬 视频理解人类测试")
    st.write("欢迎参加视频理解测试！请观看视频并回答问题。")
    
    # 获取当前视频数据
    evaluated_videos = st.session_state.evaluated_videos
    version = st.session_state.version
    
    # 侧边栏
    with st.sidebar:
        st.header("测试进度")
        progress = (st.session_state.current_index + 1) / len(evaluated_videos)
        st.progress(progress)
        st.write(f"问题 {st.session_state.current_index + 1}/{len(evaluated_videos)}")
        
        if st.button("查看测试结果", disabled=len(st.session_state.results) == 0):
            st.session_state.show_results = True
        
        if st.button("重置测试"):
            st.session_state.current_index = 0
            st.session_state.results = []
            st.session_state.show_results = False
            st.session_state.selected_answer = None
    
    # 显示结果页面
    if st.session_state.show_results:
        st.header("📊 测试结果统计")
        
        stats = generate_statistics(st.session_state.results)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总问题数", stats['total_questions'])
        col2.metric("正确答案", stats['correct_answers'])
        col3.metric("正确率", f"{stats['accuracy']:.1f}%")
        
        st.subheader("各场景统计")
        st.code(stats['table'], language="text")
        
        # 保存结果
        if st.button("保存测试结果"):
            output_file = save_results(st.session_state.results, version)
            st.success(f"✅ 测试结果已保存到: {output_file}")
            
            # 提供下载链接
            with open(output_file, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label="下载测试结果",
                data=file_bytes,
                file_name=output_file,
                mime="application/json"
            )
        
        if st.button("返回测试"):
            st.session_state.show_results = False
        
        return
    
    # 当前视频数据
    current_video = evaluated_videos[st.session_state.current_index]
    
    # 主内容区域
    st.markdown(f"### 问题 {st.session_state.current_index + 1}/{len(evaluated_videos)}")
    
    # 显示视频信息
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # 固定信息显示，减少闪烁
        info_container = st.container()
        with info_container:
            st.info(f"**对象**: {current_video['object_name']}")
            st.info(f"**场景**: {current_video['scenario_type']}")
        
        # 显示视频（固定容器，减少闪烁）
        video_container = st.container()
        with video_container:
            video_path = current_video['video_path']
            if os.path.exists(video_path):
                st.video(video_path, format="video/mp4", start_time=0)
            else:
                st.error(f"❌ 视频文件不存在: {video_path}")
    
    with col2:
        # 使用表单减少闪烁
        with st.form(key="answer_form", clear_on_submit=False):
            # 准备要翻译的文本（问题和选项）
            question = current_video['question']
            options = current_video['options']
            
            # 组合问题和选项成一个文本块
            text_to_translate = f"问题: {question}\n\n选项:\n"
            for opt_key, opt_value in options.items():
                text_to_translate += f"{opt_key}. {opt_value}\n"
            
            # 获取当前视频的翻译结果（避免重复翻译）
            video_id = current_video['id']
            if video_id not in st.session_state.translations:
                with st.spinner("正在翻译问题..."):
                    translation_result = translate_text(text_to_translate)
                    st.session_state.translations[video_id] = translation_result
            else:
                translation_result = st.session_state.translations[video_id]
            
            # 显示问题和翻译
            st.markdown("### 🤔 问题")
            
            # 创建两个列显示中英文问题
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                st.subheader("英文")
                st.write(question)
            with q_col2:
                st.subheader("中文")
                # 从翻译结果中提取中文问题
                translated_question = translation_result['translated'].split("选项:")[0].replace("问题:", "").strip()
                st.write(translated_question)
            
            # 显示选项和翻译
            st.markdown("### 📋 选项")
            
            # 从翻译结果中提取选项部分
            translated_options_part = translation_result['translated'].split("选项:")[1].strip() if "选项:" in translation_result['translated'] else ""
            
            # 解析翻译后的选项
            translated_options = {}
            if translated_options_part:
                for line in translated_options_part.split("\n"):
                    line = line.strip()
                    if line and ". " in line:
                        opt_key, opt_value = line.split(". ", 1)
                        translated_options[opt_key.strip()] = opt_value.strip()
            
            # 选项选择
            selected_answer = st.radio(
                "请选择正确答案:",
                list(options.keys()),
                format_func=lambda x: f"{x}. {options[x]}",
                key="selected_answer",
                label_visibility="collapsed"
            )
            
            # 显示中文选项
            if translated_options:
                st.markdown("#### 中文选项")
                for opt_key in options.keys():
                    if opt_key in translated_options:
                        st.write(f"{opt_key}. {translated_options[opt_key]}")
            
            # 表单按钮
            col_prev, col_next = st.columns(2)
            
            with col_prev:
                prev_button = st.form_submit_button("上一题", disabled=st.session_state.current_index == 0)
            
            with col_next:
                submit_button = st.form_submit_button("提交答案")
            
        # 处理上一题按钮
        if prev_button:
            st.session_state.current_index -= 1
            st.rerun()
            return
        
        # 处理提交答案按钮
        if submit_button:
            # 判断答案是否正确
            correct_answer = current_video['correct_answer']
            is_correct = str(selected_answer).upper() == str(correct_answer).upper()
            
            # 保存结果
            result = {
                'id': current_video['id'],
                'object_name': current_video['object_name'],
                'scenario_type': current_video['scenario_type'],
                'video_file': current_video['video_file'],
                'question': current_video['question'],
                'options': current_video['options'],
                'selected_answer': selected_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 更新结果列表
            found = False
            for i, res in enumerate(st.session_state.results):
                if res['id'] == result['id']:
                    st.session_state.results[i] = result
                    found = True
                    break
            if not found:
                st.session_state.results.append(result)
            
            # 显示反馈
            feedback_container = st.container()
            with feedback_container:
                if is_correct:
                    st.success("✅ 回答正确！")
                else:
                    st.error(f"❌ 回答错误！正确答案是: {correct_answer}. {options[correct_answer]}")
            
            # 自动前进到下一题
            if st.session_state.current_index < len(evaluated_videos) - 1:
                # 使用延迟跳转，避免闪烁
                time.sleep(0.5)
                st.session_state.current_index += 1
                st.rerun()
            else:
                st.success("🎉 恭喜！您已完成所有测试问题！")
                time.sleep(1)
                st.session_state.show_results = True
                st.rerun()

if __name__ == "__main__":
    main()