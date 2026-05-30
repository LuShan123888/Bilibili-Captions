---
name: video-captions
description: "Download video subtitles/captions from Bilibili, YouTube, or local files. Supports API subtitles and Whisper ASR fallback. TRIGGER on: 'download subtitles', 'download captions', 'get subtitles', 'get captions', '字幕', '视频字幕', 'transcribe', 'ASR', 'video-captions'."
---

# video-captions

Download video subtitles from Bilibili, YouTube, or transcribe local audio/video files via Whisper ASR.

## Prerequisites

System dependencies required:

```bash
brew install yt-dlp ffmpeg
```

Python package installed:

```bash
uv tool install video-captions
# or in a project with uv:
uv add video-captions
```

## Commands Reference

| User intent | Command |
|---|---|
| Download subtitles (text) | `video-captions "<URL>"` |
| Download subtitles (SRT) | `video-captions --format srt "<URL>"` |
| Download subtitles (JSON) | `video-captions --format json "<URL>"` |
| Specify browser for cookies | `video-captions --browser arc "<URL>"` |
| Transcribe local file | `video-captions "/path/to/video.mp4"` |
| Use smaller ASR model | `video-captions --model small "/path/to/file.mp4"` |
| Show video title & subtitle count | `video-captions -v "<URL>"` |

## Supported Platforms

| Platform | URL patterns |
|---|---|
| Bilibili | `https://www.bilibili.com/video/BV*`, BV ID alone |
| YouTube | `https://www.youtube.com/watch?v=*`, `https://youtu.be/*` |
| Local file | Any local path to audio/video file |

## Options

| Flag | Values | Default | Description |
|---|---|---|---|
| `--browser` | `arc`, `auto`, `chrome`, `edge`, `firefox`, `brave` | `arc` | Browser to read cookies from |
| `--model` | `base`, `small`, `medium`, `large` | `large` | Whisper ASR model size (used when no API subtitles) |
| `--format` | `text`, `srt`, `json` | `text` | Output format |
| `-v`, `--verbose` | flag | off | Show video title and subtitle count |

## How It Works

1. Auto-detects platform from URL/path
2. Tries to fetch subtitles from platform API first
3. Falls back to Whisper ASR if no subtitles available
4. Auto-converts traditional Chinese to simplified Chinese

## Output Formats

- **text** — Plain text, one line per subtitle segment. Best for reading and summarizing.
- **srt** — SRT subtitle format with timestamps. Best for video playback.
- **json** — Structured JSON with metadata. Best for programmatic processing.

## Browser Cookies

Some videos require login cookies to access subtitles. The tool auto-reads cookies from browsers:

- `arc` (default) — use Arc cookies
- `auto` — tries all installed browsers
- `chrome`, `edge`, `firefox`, `brave` — use a specific browser

If subtitle access fails, try specifying a browser where you're logged in:

```bash
video-captions --browser arc "https://www.bilibili.com/video/BV1xx"
```

## ASR Model Sizes

Only used when platform has no subtitles (Whisper ASR fallback). Apple Silicon only.

| Model | Speed | Accuracy |
|---|---|---|
| `base` | Fastest | Lower |
| `small` | Fast | Good |
| `medium` | Balanced | Better |
| `large` | Slowest | Best |

## Common Workflows

### 1. Quick subtitle download

```bash
video-captions "https://www.bilibili.com/video/BV16YC3BrEDz"
```

### 2. Get SRT for video editing

```bash
video-captions --format srt "https://www.youtube.com/watch?v=kQ-aFczITCg"
```

### 3. Transcribe a local recording

```bash
video-captions --model medium "/path/to/meeting-recording.mp4"
```

### 4. Get structured data for processing

```bash
video-captions --format json "https://www.bilibili.com/video/BV16YC3BrEDz"
```

### 5. Handle cookie-protected content

```bash
video-captions --browser arc "https://www.bilibili.com/video/BV1xx"
```

## Error Handling

| Error | Cause | Resolution |
|---|---|---|
| "不支持的来源" | URL or path not recognized | Check URL format or file path |
| Cookie/auth failure | Login required | Use `--browser` to specify a logged-in browser |
| ASR failure | Non-audio file or unsupported format | Ensure file contains audio track; supported: mp3, wav, mp4, mkv, etc. |
| ffmpeg not found | Missing system dependency | Run `brew install ffmpeg` |
| yt-dlp not found | Missing system dependency | Run `brew install yt-dlp` |
