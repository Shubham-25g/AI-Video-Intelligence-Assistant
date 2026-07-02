import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "Downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best", 
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        
        # BYPASS CONFIGURATION: Force client layers that do not aggressively enforce PO tokens
        "extractor_args": {
            "youtube": {
                "player_client": ["web_safari", "android_vr"],
                "formats": ["missing_pot"]
            }
        }
    }
    
    # 1. Apply Cloud Proxy if configured
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url:
        ydl_opts["proxy"] = proxy_url
        
    # 2. Authenticate using Cloud Secrets Cookies if configured
    cookies_content = os.getenv("YT_COOKIES")
    temp_cookies_path = os.path.join(DOWNLOAD_DIR, "temp_cookies.txt")
    
    if cookies_content:
        print("Injecting normalized browser authentication session cookies...")
        # Clean up any potential copy-paste formatting anomalies from the cloud environment
        cleaned_lines = [line.strip() for line in cookies_content.strip().splitlines() if line.strip()]
        cleaned_cookies = "\n".join(cleaned_lines)
        
        with open(temp_cookies_path, "w", encoding="utf-8") as f:
            f.write(cleaned_cookies + "\n")
        ydl_opts["cookiefile"] = temp_cookies_path
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
        return filename
        
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Media extraction failed: {str(e)}")
        
    finally:
        # Guarantee that the temporary cookie file is deleted immediately for safety
        if os.path.exists(temp_cookies_path):
            try:
                os.remove(temp_cookies_path)
            except Exception:
                pass


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_mins: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_mins * 60 * 1000

    if len(audio) <= chunk_ms:
        return [wav_path]

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i+1}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = convert_to_wav(download_youtube_audio(source))
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks