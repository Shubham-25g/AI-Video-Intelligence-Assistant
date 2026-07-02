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
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
        return filename
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        # Catch cloud data-center IP bans from YouTube's firewall
        if "403" in error_msg or "Forbidden" in error_msg:
            raise RuntimeError(
                "🛑 YouTube has blocked this cloud server's public IP address (HTTP 403 Forbidden).\n\n"
                "Because Streamlit Cloud runs on public data center servers (AWS), YouTube frequently blacklists their entire network to prevent scraping bots.\n\n"
                "⚡ **How to bypass this instantly:** Switch to the **'📁 Upload Local File Asset'** tab at the top of the page and drag-and-drop your audio/video file directly! It will process flawlessly without hitting external network restrictions."
            )
        else:
            raise RuntimeError(f"Media extraction failed: {error_msg}")


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