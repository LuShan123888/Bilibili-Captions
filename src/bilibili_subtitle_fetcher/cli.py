# /// script
# dependencies = ["requests", "faster-whisper", "opencc-python-reimplemented"]
# -*-

"""
B站字幕抓取工具 - CLI 版本

支持从B站视频下载字幕，若无字幕则使用 Whisper ASR 生成。
"""

import asyncio
import sys
import requests
import subprocess
import os
import re
from datetime import datetime

# 导入共享模块（同包）
from .api import get_sessdata, make_safe_filename, extract_bvid

# 尝试导入 faster-whisper（推荐，速度快）
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    USE_FASTER_WHISPER = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    USE_FASTER_WHISPER = False
    # 降级使用 openai-whisper
    try:
        import whisper
        OPENAI_WHISPER_AVAILABLE = True
    except ImportError:
        OPENAI_WHISPER_AVAILABLE = False
        print("错误: 未安装 Whisper 库。")
        print("推荐安装: pip install faster-whisper")
        print("或: pip install openai-whisper")
        sys.exit(1)

# 尝试导入 opencc 用于繁简转换
try:
    from opencc import OpenCC
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    print("提示: opencc 库未安装，字幕将保持原样（可能是繁体）。")
    print("如需自动转换为简体，请使用 'pip install opencc-python-reimplemented' 安装。")


def create_output_dir():
    """创建带时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def convert_to_simplified(srt_filename):
    """将字幕文件从繁体转换为简体"""
    if not OPENCC_AVAILABLE:
        return False

    try:
        with open(srt_filename, 'r', encoding='utf-8') as f:
            content = f.read()

        cc = OpenCC('t2s')
        simplified_content = cc.convert(content)

        with open(srt_filename, 'w', encoding='utf-8') as f:
            f.write(simplified_content)

        print(f"已将字幕转换为简体中文。")
        return True
    except Exception as e:
        print(f"警告: 繁简转换失败: {e}")
        return False

def get_video_info(url):
    """获取B站视频的标题和CID（同步版本，供CLI使用）"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/'
        }

        # 使用共享的 extract_bvid 函数提取BV号
        bvid = extract_bvid(url)

        # 获取视频信息的API
        info_api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        response = requests.get(info_api_url, headers=headers)
        response.raise_for_status()

        data = response.json()
        if data['code'] == 0:
            video_title = data['data']['title']
            cid = data['data']['cid']
            return video_title, cid, bvid
        else:
            print(f"错误: 无法获取视频信息。B站返回: {data['message']}")
            return None, None, None
    except ValueError as e:
        print(f"错误: {e}")
        return None, None, None
    except Exception as e:
        print(f"错误: 获取视频信息时出错: {e}")
        return None, None, None

def get_subtitles_from_bilibili(video_title, cid, bvid, output_dir, video_url):
    """尝试从B站下载已有的字幕"""
    try:
        sessdata = get_sessdata()

        # 使用 SESSDATA 获取字幕
        cookies = {'SESSDATA': sessdata}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': f'https://www.bilibili.com/video/{bvid}',
        }

        subtitle_api_url = f"https://api.bilibili.com/x/player/wbi/v2?bvid={bvid}&cid={cid}"
        response = requests.get(subtitle_api_url, headers=headers, cookies=cookies)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 0:
            subtitle_data = data['data'].get('subtitle', {})
            subtitle_list = subtitle_data.get('subtitles', [])

            if subtitle_list:
                # 选择中文 AI 字幕
                zh_subtitle = None
                for sub in subtitle_list:
                    if sub.get('lan') in ['ai-zh', 'zh-Hans', 'zh-CN', 'zh']:
                        zh_subtitle = sub
                        break

                if not zh_subtitle:
                    zh_subtitle = subtitle_list[0]

                print(f"发现字幕: {zh_subtitle.get('lan_doc', zh_subtitle.get('lan'))} ({len(subtitle_list)} 个可用)")

                subtitle_url = zh_subtitle.get('subtitle_url')
                if subtitle_url:
                    if not subtitle_url.startswith('http'):
                        subtitle_url = 'https:' + subtitle_url

                    # 下载字幕 JSON
                    subtitle_json = requests.get(subtitle_url, headers=headers).json()

                    # 转换为 SRT
                    srt_content = ""
                    for i, item in enumerate(subtitle_json.get('body', [])):
                        start_time = item['from']
                        end_time = item['to']
                        content = item['content']

                        start_h, start_m, start_s = int(start_time // 3600), int((start_time % 3600) // 60), start_time % 60
                        end_h, end_m, end_s = int(end_time // 3600), int((end_time % 3600) // 60), end_time % 60

                        srt_content += f"{i+1}\n"
                        srt_content += f"{start_h:02}:{start_m:02}:{start_s:06.3f}".replace('.', ',')
                        srt_content += f" --> {end_h:02}:{end_m:02}:{end_s:06.3f}".replace('.', ',')
                        srt_content += f"\n{content}\n\n"

                    safe_title = make_safe_filename(video_title)
                    subtitle_filename = os.path.join(output_dir, f"{safe_title}.srt")
                    with open(subtitle_filename, 'w', encoding='utf-8') as f:
                        f.write(srt_content)

                    # 提取纯文本用于终端输出
                    text_lines = [item['content'] for item in subtitle_json.get('body', [])]
                    text_output = '\n'.join(text_lines)

                    print(f"\n{'='*60}")
                    print(f"字幕来源: B站AI字幕 (API直接获取)")
                    print(f"字幕内容 (纯文本):")
                    print(f"{'='*60}")
                    print(text_output)
                    print(f"{'='*60}")
                    print(f"\n字幕下载成功！已保存为: {subtitle_filename}")
                    print(f"共 {len(subtitle_json.get('body', []))} 条字幕")
                    return True

            print("该视频没有自带字幕。")
            return False
        else:
            print(f"错误: B站返回: {data.get('message', '未知错误')}")
            return False

    except Exception as e:
        print(f"错误: 检查或下载字幕时发生错误: {e}")
        return False

def generate_subtitles_with_asr(video_url, safe_video_title, model_size="base", output_dir=".", bvid=None):
    """使用ASR技术生成字幕"""
    print("\n--- 步骤 2: 正在下载视频并提取音频用于ASR ---")
    # 如果提供了bvid，使用标准视频URL格式（兼容稍后观看等非标准URL）
    if bvid:
        video_url = f"https://www.bilibili.com/video/{bvid}"
    try:
        video_filename = os.path.join(output_dir, f"{safe_video_title}.mp4")
        audio_filename = os.path.join(output_dir, f"{safe_video_title}.wav")

        # 检查音频文件是否已存在
        if not os.path.exists(audio_filename):
            # 使用 yt-dlp 下载视频
            if not os.path.exists(video_filename):
                print(f"正在下载视频: {video_url}")
                subprocess.run(['yt-dlp', '-o', video_filename, video_url], check=True, capture_output=True)
                print("视频下载完成。")

            # 使用 ffmpeg 提取音频
            print("正在提取音频...")
            subprocess.run(['ffmpeg', '-y', '-i', video_filename, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_filename],
                           check=True, capture_output=True)
            print("音频提取完成。")
        else:
            print("使用已存在的音频文件。")

        # 优先使用 faster-whisper（速度更快）
        if USE_FASTER_WHISPER:
            transcribe_with_faster_whisper(audio_filename, safe_video_title, model_size, output_dir)
        elif OPENAI_WHISPER_AVAILABLE:
            transcribe_with_openai_whisper(audio_filename, safe_video_title, model_size, output_dir)
        else:
            print("错误: 没有可用的 Whisper 实现")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"错误: 外部命令执行失败: {e}")
        return False
    except Exception as e:
        print(f"错误: 生成字幕时发生未知错误: {e}")
        return False

def transcribe_with_faster_whisper(audio_filename, video_title, model_size, output_dir):
    """使用 faster-whisper 生成字幕（速度快 4-5 倍）"""
    import time
    print(f"正在使用 faster-whisper ({model_size} 模型) 生成字幕...")

    # 检测可用设备
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # faster-whisper 暂不支持 MPS，回退到 CPU
        compute_type = "int8"  # 使用量化加速
    else:
        device = "cpu"
        compute_type = "int8"  # CPU 使用 int8 量化加速

    print(f"使用设备: {device}, 计算类型: {compute_type}")

    # 加载模型
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=os.cpu_count() or 4,  # 使用所有 CPU 核心
        download_root=os.path.expanduser("~/.cache/whisper")
    )

    start_time = time.time()

    # 执行转录
    segments, info = model.transcribe(
        audio_filename,
        language='zh',
        beam_size=5,  # 束搜索大小
        vad_filter=True,  # 使用 VAD 过滤静音
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )

    elapsed = time.time() - start_time

    # 保存为SRT格式
    srt_filename = os.path.join(output_dir, f"{video_title}_asr.srt")
    srt_content = ""

    # 收集纯文本用于终端输出
    text_lines = []
    with open(srt_filename, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            text_lines.append(text)

            start_time_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}".replace('.', ',')
            end_time_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}".replace('.', ',')

            line = f"{i+1}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
            srt_content += line
            f.write(line)

    text_output = '\n'.join(text_lines)

    # 输出纯文本字幕内容
    print(f"\n{'='*60}")
    print(f"字幕来源: Whisper ASR语音识别 (AI生成)")
    print(f"字幕内容 (纯文本):")
    print(f"{'='*60}")
    print(text_output)
    print(f"{'='*60}")

    print(f"ASR字幕生成成功！耗时: {elapsed:.1f}秒")
    print(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")

    # 自动转换为简体中文
    convert_to_simplified(srt_filename)

def transcribe_with_openai_whisper(audio_filename, video_title, model_size, output_dir):
    """使用 openai-whisper 生成字幕"""
    import whisper  # 这里导入
    print(f"正在使用 openai-whisper ({model_size} 模型) 生成字幕...")
    model = whisper.load_model(model_size)

    result = model.transcribe(
        audio_filename,
        language='zh',
        task='transcribe',
        verbose=False,
        fp16=False,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
    )

    srt_filename = os.path.join(output_dir, f"{video_title}_asr.srt")
    srt_content = ""

    # 收集纯文本用于终端输出
    text_lines = []
    with open(srt_filename, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments']):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            text_lines.append(text)

            start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}".replace('.', ',')
            end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}".replace('.', ',')

            line = f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
            srt_content += line
            f.write(line)

    text_output = '\n'.join(text_lines)

    # 输出纯文本字幕内容
    print(f"\n{'='*60}")
    print(f"字幕来源: Whisper ASR语音识别 (AI生成)")
    print(f"字幕内容 (纯文本):")
    print(f"{'='*60}")
    print(text_output)
    print(f"{'='*60}")

    print(f"ASR字幕生成成功！已保存为: {srt_filename}")
    convert_to_simplified(srt_filename)

def main() -> None:
    """CLI入口点"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: bilibili-subtitle <B站视频URL> [模型大小]")
        print("模型大小可选: base, small, medium (默认), large")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"

    # 验证模型大小
    valid_models = ["base", "small", "medium", "large"]
    if model_size not in valid_models:
        print(f"警告: 无效的模型大小 '{model_size}'，使用默认 'medium' 模型")
        model_size = "medium"

    print(f"使用 Whisper 模型: {model_size}")

    # 创建带时间戳的输出目录
    output_dir = create_output_dir()
    print(f"输出目录: {output_dir}")

    video_title, cid, bvid = get_video_info(video_url)

    if not all([video_title, cid, bvid]):
        print("无法获取视频信息，程序退出。")
        sys.exit(1)

    safe_title = make_safe_filename(video_title)

    # 尝试下载现有字幕
    if not get_subtitles_from_bilibili(video_title, cid, bvid, output_dir, video_url):
        # 如果没有现有字幕，则使用ASR生成
        generate_subtitles_with_asr(video_url, safe_title, model_size, output_dir, bvid)


if __name__ == "__main__":
    main()