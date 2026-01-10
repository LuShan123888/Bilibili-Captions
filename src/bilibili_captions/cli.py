# /// script
# dependencies = ["requests", "faster-whisper", "opencc-python-reimplemented"]
# -*-

"""
B站字幕抓取工具 - CLI 版本

支持从B站视频下载字幕，若无字幕则使用 Whisper ASR 生成。
"""

import sys
import requests
import subprocess
import os

from .api import get_sessdata, extract_bvid

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


def convert_to_simplified(text: str) -> str:
    """将文本从繁体转换为简体"""
    if not OPENCC_AVAILABLE:
        return text

    try:
        cc = OpenCC('t2s')
        return cc.convert(text)
    except Exception as e:
        print(f"警告: 繁简转换失败: {e}")
        return text

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

def get_subtitles_from_bilibili(video_title, cid, bvid, video_url):
    """尝试从B站获取并显示字幕"""
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

                    # 提取纯文本并输出
                    text_lines = [item['content'] for item in subtitle_json.get('body', [])]
                    text_output = '\n'.join(text_lines)
                    text_output = convert_to_simplified(text_output)

                    print(f"\n{'='*60}")
                    print(f"字幕来源: B站AI字幕 (API直接获取)")
                    print(f"视频标题: {video_title}")
                    print(f"{'='*60}")
                    print(text_output)
                    print(f"{'='*60}")
                    print(f"\n共 {len(subtitle_json.get('body', []))} 条字幕")
                    return True

            print("该视频没有自带字幕。")
            return False
        else:
            print(f"错误: B站返回: {data.get('message', '未知错误')}")
            return False

    except Exception as e:
        print(f"错误: 检查或下载字幕时发生错误: {e}")
        return False

def generate_subtitles_with_asr(video_url, model_size="base", bvid=None):
    """使用ASR技术生成字幕"""
    import tempfile
    print("\n--- 步骤 2: 正在下载视频并提取音频用于ASR ---")

    # 如果提供了bvid，使用标准视频URL格式
    if bvid:
        video_url = f"https://www.bilibili.com/video/{bvid}"

    # 使用临时目录处理 ASR
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            safe_video_title = "video"
            video_filename = os.path.join(temp_dir, f"{safe_video_title}.mp4")
            audio_filename = os.path.join(temp_dir, f"{safe_video_title}.wav")

            # 下载视频
            print(f"正在下载视频: {video_url}")
            subprocess.run(['yt-dlp', '-o', video_filename, video_url], check=True, capture_output=True)
            print("视频下载完成。")

            # 提取音频
            print("正在提取音频...")
            subprocess.run(['ffmpeg', '-y', '-i', video_filename, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_filename],
                           check=True, capture_output=True)
            print("音频提取完成。")

            # 执行转录并输出到终端
            if USE_FASTER_WHISPER:
                transcribe_with_faster_whisper(audio_filename, model_size)
            elif OPENAI_WHISPER_AVAILABLE:
                transcribe_with_openai_whisper(audio_filename, model_size)
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

def transcribe_with_faster_whisper(audio_filename, model_size):
    """使用 faster-whisper 生成字幕并输出到终端"""
    import time
    print(f"正在使用 faster-whisper ({model_size} 模型) 生成字幕...")

    # 检测可用设备
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # faster-whisper 暂不支持 MPS，回退到 CPU
        compute_type = "int8"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"使用设备: {device}")

    # 加载模型
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=os.cpu_count() or 4,
        download_root=os.path.expanduser("~/.cache/whisper")
    )

    start_time = time.time()

    # 执行转录
    segments, info = model.transcribe(
        audio_filename,
        language='zh',
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    elapsed = time.time() - start_time

    # 收集字幕文本
    text_lines = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text_lines.append(convert_to_simplified(text))

    text_output = '\n'.join(text_lines)

    # 输出到终端
    print(f"\n{'='*60}")
    print(f"字幕来源: Whisper ASR语音识别 (AI生成)")
    print(f"{'='*60}")
    print(text_output)
    print(f"{'='*60}")

    print(f"ASR字幕生成成功！耗时: {elapsed:.1f}秒")
    print(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"共 {len(text_lines)} 条字幕")

def transcribe_with_openai_whisper(audio_filename, model_size):
    """使用 openai-whisper 生成字幕并输出到终端"""
    import whisper
    print(f"正在使用 openai-whisper ({model_size} 模型) 生成字幕...")
    model = whisper.load_model(model_size)

    result = model.transcribe(
        audio_filename,
        language='zh',
        task='transcribe',
        verbose=False,
        fp16=False,
    )

    # 收集字幕文本
    text_lines = []
    for segment in result['segments']:
        text = segment['text'].strip()
        if text:
            text_lines.append(convert_to_simplified(text))

    text_output = '\n'.join(text_lines)

    # 输出到终端
    print(f"\n{'='*60}")
    print(f"字幕来源: Whisper ASR语音识别 (AI生成)")
    print(f"{'='*60}")
    print(text_output)
    print(f"{'='*60}")

    print(f"ASR字幕生成成功！")
    print(f"共 {len(text_lines)} 条字幕")

def main() -> None:
    """CLI入口点"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: bilibili-captions <B站视频URL> [模型大小]")
        print("模型大小可选: base, small, medium (默认), large")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"

    # 验证模型大小
    valid_models = ["base", "small", "medium", "large"]
    if model_size not in valid_models:
        print(f"警告: 无效的模型大小 '{model_size}'，使用默认 'medium' 模型")
        model_size = "medium"

    print(f"使用模型: {model_size}")

    video_title, cid, bvid = get_video_info(video_url)

    if not all([video_title, cid, bvid]):
        print("无法获取视频信息，程序退出。")
        sys.exit(1)

    print(f"视频标题: {video_title}")

    # 尝试从 API 获取字幕，无字幕则使用 ASR
    if not get_subtitles_from_bilibili(video_title, cid, bvid, video_url):
        generate_subtitles_with_asr(video_url, model_size, bvid)


if __name__ == "__main__":
    main()