import sys
import argparse
import os
import subprocess
import json

# Suppress huggingface_hub symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel
from openai import OpenAI

GLOBAL_MODEL = None
GLOBAL_DEBUG = False


def check_dependencies(api_base):
    """Check that all required external tools are available before starting work."""
    errors = []

    # --- Check ffmpeg ---
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, **kwargs,
        )
        if result.returncode != 0:
            errors.append("ffmpeg found but returned an error. Reinstall ffmpeg.")
    except FileNotFoundError:
        errors.append(
            "ffmpeg not found!\n"
            "  -> Install ffmpeg and add it to PATH,\n"
            "     or place ffmpeg.exe next to this script."
        )

    # --- Check LM Studio API with a real completion call ---
    try:
        client = OpenAI(base_url=api_base, api_key="lm-studio")
        response = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        # If we get here, the API is reachable and a model is loaded
    except Exception as e:
        errors.append(
            f"LM Studio API is not reachable ({api_base}):\n"
            f"  {e}\n"
            f"  -> Start LM Studio and load a model before running the script."
        )

    if errors:
        print("\n" + "=" * 50)
        print("ERROR: Not all dependencies are available!")
        print("=" * 50)
        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. {err}")
        print("\n" + "=" * 50)
        sys.exit(1)
    else:
        print("[OK] All dependencies verified.")


def transcribe_audio(file_path, model_size="turbo", log_msg="downloading if not cached"):
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        print(f"Loading Whisper model '{model_size}' ({log_msg}, please wait)...")
        # Using 'auto' for device so it works on both CPU and GPU
        GLOBAL_MODEL = WhisperModel(model_size, device="auto", compute_type="default")
    
    print(f"Transcribing {file_path}...")
    segments, info = GLOBAL_MODEL.transcribe(file_path, beam_size=5)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    print("Recognizing speech (this may take a while depending on PC power)...")
    
    total_mins = info.duration / 60.0
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        current_mins = segment.end / 60.0
        percent = (current_mins / total_mins) * 100
        print(f"\rAudio transcribed: {current_mins:.2f} of {total_mins:.2f} mins ({percent:.1f}% done)...", end="", flush=True)
        
    print("\nTranscription complete!")
    return results

def find_semantic_break(text_chunk, api_base="http://localhost:1234/v1", model_name="local-model"):
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    prompt = (
        "You are an AI that helps split a book into logical parts. "
        "I will provide a text fragment. Silence between phrases is indicated by special markers like [Pause: X sec]. "
        "Find the most logical sentence to split the scene (end of a scene, complete thought, transition), paying special attention to places "
        "where the pause is longer than usual (e.g., 1-3 seconds). "
        "Reply ONLY with the exact sentence where the cut should occur, WITHOUT the pause marker and WITHOUT any extra text.\n\n"
        f"Text:\n{text_chunk}\n\nSplit sentence:"
    )
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def split_audio_ffmpeg(input_file, start_t, end_t, output_file):
    print(f"Splitting: {start_t} to {end_t if end_t else 'end'} -> {output_file}")
    command = ["ffmpeg", "-y", "-i", input_file, "-ss", str(start_t)]
    if end_t is not None:
        command.extend(["-to", str(end_t)])
    command.extend(["-c", "copy", output_file])
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg Error:\n{result.stderr}")
    except FileNotFoundError:
        print("\nERROR: 'ffmpeg' not found! Ensure FFmpeg is installed and added to the system PATH (or put ffmpeg.exe in the script folder).")
        sys.exit(1)

def _main_logic():
    parser = argparse.ArgumentParser(description="Semantic Audiobook Splitter")
    parser.add_argument("--input", help="Input audiobook file")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--target-mins", type=float, default=7.0, help="Target chunk duration in minutes")
    parser.add_argument("--api-base", default="http://localhost:1234/v1", help="LM Studio API base URL")
    parser.add_argument("--llm-model", default="local-model", help="LLM Model name")
    parser.add_argument("--whisper-model", default="turbo", help="Whisper Model name")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--transcription", default=None, help="Path to existing transcription JSON file (skip transcription step)")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print("\n=== FIRST RUN ===")
        config = {
            "out_dir": "output",
            "target_mins": 7.0,
            "api_base": "http://localhost:1234/v1",
            "llm_model": "local-model",
            "whisper_model": "turbo",
            "debug": False,
            "whisper_models": [
                {"name": "tiny", "log": "downloading ~75MB if not cached", "languages": "Multi"},
                {"name": "base", "log": "downloading ~145MB if not cached", "languages": "Multi"},
                {"name": "small", "log": "downloading ~490MB if not cached", "languages": "Multi"},
                {"name": "medium", "log": "downloading ~1.5GB if not cached", "languages": "Multi"},
                {"name": "turbo", "log": "downloading ~1.5GB if not cached", "languages": "Multi"},
                {"name": "large-v3", "log": "downloading ~3.1GB if not cached", "languages": "Multi"}
            ]
        }
        with open(args.config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"Settings saved to {args.config}\n")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
        if args.out_dir == "output" and "out_dir" in config:
            args.out_dir = config["out_dir"]
        if args.target_mins == 7.0 and "target_mins" in config:
            args.target_mins = config["target_mins"]
        if args.api_base == "http://localhost:1234/v1" and "api_base" in config:
            args.api_base = config["api_base"]
        if args.llm_model == "local-model" and "llm_model" in config:
            args.llm_model = config["llm_model"]
        if hasattr(args, 'whisper_model') and args.whisper_model == "turbo" and "whisper_model" in config:
            args.whisper_model = config["whisper_model"]
        if config.get("debug", False):
            args.debug = True
            
    global GLOBAL_DEBUG
    GLOBAL_DEBUG = args.debug
    if GLOBAL_DEBUG:
        sys.stdout = Logger("execution_log.txt")

    if not args.input:
        in_file = input("Enter the full path to the audio file (.mp3, .m4b, etc.): ").strip('"').strip("'")
        args.input = in_file
    
    if not args.input:
        print("ERROR: File path not specified!")
        return
    
    # --- Pre-flight dependency checks (before any heavy work) ---
    check_dependencies(args.api_base)
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Create a subfolder per audiobook: output/<filename>/
    args.out_dir = os.path.join(args.out_dir, base_name)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- Resume / transcription logic ---
    segments = None
    transcription_file = os.path.join(args.out_dir, f"STT_{base_name}.json")
    
    if args.transcription:
        # Explicit transcription file provided via --transcription
        if not os.path.exists(args.transcription):
            print(f"ERROR: Transcription file not found: {args.transcription}")
            return
        print(f"Loading existing transcription from {args.transcription}...")
        with open(args.transcription, "r", encoding="utf-8") as f:
            segments = json.load(f)
        print(f"Loaded {len(segments)} segments.")
    elif os.path.exists(transcription_file):
        # Auto-detected existing transcription
        print(f"\n>> Existing transcription found: {transcription_file}")
        answer = input("Use it? (y/n, default y): ").strip().lower()
        if answer in ("", "y", "yes"):
            print(f"Loading transcription from {transcription_file}...")
            with open(transcription_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
            print(f"Loaded {len(segments)} segments.")
    
    if segments is None:
        # Need to transcribe
        whisper_mod = getattr(args, 'whisper_model', 'turbo')
        whisper_log = "downloading if not cached"
        if "whisper_models" in config:
            for wm in config["whisper_models"]:
                if wm.get("name") == whisper_mod:
                    whisper_log = wm.get("log", whisper_log)
                    break
        
        segments = transcribe_audio(args.input, whisper_mod, whisper_log)
        if not segments:
            print("No speech detected.")
            return
        
        print(f"Total transcribed segments: {len(segments)}")
        
        with open(transcription_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
        print(f"Transcription saved to {transcription_file}")
    
    print(f"\nTotal segments to process: {len(segments)}")
    
    target_sec = args.target_mins * 60.0
    
    current_chunk = []
    current_start_t = 0.0
    part_idx = 1
    
    for seg in segments:
        current_chunk.append(seg)
        if seg["end"] - current_start_t >= target_sec:
            # Get context of last ~2 mins for the LLM
            context_duration = 120.0
            context_segs = [s for s in current_chunk if s["end"] >= seg["end"] - context_duration]
            
            context_text_parts = []
            for i, s in enumerate(context_segs):
                pause = context_segs[i+1]["start"] - s["end"] if i < len(context_segs)-1 else 0.0
                if pause > 0.5:
                    context_text_parts.append(f"{s['text']} [Pause: {pause:.1f} sec]")
                else:
                    context_text_parts.append(s['text'])
            context_text = " ".join(context_text_parts)
            
            print(f"\n---\nFinding semantic break around {seg['end']:.2f}s ...")
            break_sentence = find_semantic_break(context_text, args.api_base, args.llm_model)
            print(f"LLM suggested break: '{break_sentence}'")
            
            cut_time = seg["end"]
            for s in reversed(context_segs):
                clean_break = break_sentence.split("[Pause")[0].strip()
                if clean_break and (clean_break.lower() in s["text"].lower() or s["text"].lower() in clean_break.lower()):
                    cut_time = s["end"]
                    print(f"Matched sentence at timestamp: {cut_time:.2f}s")
                    break
            
            out_file = os.path.join(args.out_dir, f"{part_idx:03d}_{base_name}.mp3")
            split_audio_ffmpeg(args.input, current_start_t, cut_time, out_file)
            
            part_idx += 1
            current_start_t = cut_time
            current_chunk = [s for s in current_chunk if s["start"] >= cut_time]
            
    if current_chunk:
        out_file = os.path.join(args.out_dir, f"{part_idx:03d}_{base_name}.mp3")
        split_audio_ffmpeg(args.input, current_start_t, None, out_file)
        
    print("Done!")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    try:
        _main_logic()
    except Exception as e:
        if GLOBAL_DEBUG:
            with open("debug.log", "w", encoding="utf-8") as f:
                import traceback
                f.write("CRASH REPORT:\n")
                f.write(traceback.format_exc())
        
        debug_msg = " Details are saved in debug.log" if GLOBAL_DEBUG else ""
        print(f"\nScript crashed with a critical error!{debug_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
