import sys
import argparse
import os
import subprocess
import json
import re
from difflib import SequenceMatcher

# Suppress huggingface_hub symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel
from openai import OpenAI

GLOBAL_MODEL = None
GLOBAL_DEBUG = False


def _normalize_text(text):
    return " ".join((text or "").split()).strip()


def _enrich_pause_metadata(segments):
    """Populate gap_before/gap_after using current segment boundaries."""
    if not segments:
        return segments

    for i, seg in enumerate(segments):
        if i == 0:
            seg["gap_before"] = 0.0
        else:
            prev_end = float(segments[i - 1].get("end", 0.0))
            cur_start = float(seg.get("start", 0.0))
            seg["gap_before"] = max(0.0, cur_start - prev_end)

    for i, seg in enumerate(segments):
        if i == len(segments) - 1:
            seg["gap_after"] = 0.0
        else:
            cur_end = float(seg.get("end", 0.0))
            next_start = float(segments[i + 1].get("start", 0.0))
            seg["gap_after"] = max(0.0, next_start - cur_end)

    return segments


def _strip_pause_markers(text):
    return re.sub(r"\s*\[(?:Long)?Pause:[^\]]+\]", "", text or "").strip()


def _tokenize_text(text):
    return re.findall(r"[A-Za-zА-Яа-яЁё]+", (text or "").lower())


def _is_heading_token(token):
    if not token:
        return False

    keywords = ["глава", "часть", "chapter", "part", "chap"]
    for kw in keywords:
        if token == kw or token.startswith(kw):
            return True
        if SequenceMatcher(None, token, kw).ratio() >= 0.72:
            return True
    return False


def _is_heading_marker_text(text):
    """Heuristic guard: keep only markers that look like chapter/part headings."""
    tokens = _tokenize_text(text)
    if not tokens:
        return False

    # Heading token should appear at the beginning, allowing minor ASR typos.
    first_tokens = tokens[:3]
    return any(_is_heading_token(tok) for tok in first_tokens[:2])


def _parse_yes_no(answer, default=True):
    normalized = (answer or "").strip().lower()
    if not normalized:
        return default

    yes_values = {"y", "yes", "д", "да", "+", "1", "true"}
    no_values = {"n", "no", "н", "нет", "-", "0", "false"}

    if normalized in yes_values:
        return True
    if normalized in no_values:
        return False
    return default


def ask_yes_no(prompt, default=True):
    answer = input(prompt)
    return _parse_yes_no(answer, default=default)


def _extract_marker_indexes(raw_text, min_idx, max_idx):
    """Extract marker indexes from LLM response."""
    raw = (raw_text or "").strip()
    if not raw:
        return []

    candidate = raw
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        candidate = raw[start:end]

    parsed = None
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = None

    indexes = []
    if isinstance(parsed, dict):
        markers = parsed.get("markers", [])
        if isinstance(markers, list):
            for marker in markers:
                if isinstance(marker, dict):
                    idx = marker.get("index")
                else:
                    idx = marker
                if isinstance(idx, int) and min_idx <= idx < max_idx:
                    indexes.append(idx)
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, int) and min_idx <= item < max_idx:
                indexes.append(item)

    if indexes:
        return sorted(set(indexes))

    # Fallback for non-JSON replies.
    for match in re.findall(r"(?<!\d)(\d{1,6})(?!\d)", raw):
        idx = int(match)
        if min_idx <= idx < max_idx:
            indexes.append(idx)

    return sorted(set(indexes))


def find_structure_markers_with_llm(segments, api_base, model_name):
    """Find probable chapter/part boundaries semantically via LLM."""
    if not segments:
        return []

    client = OpenAI(base_url=api_base, api_key="lm-studio")
    total = len(segments)
    window_size = 220
    overlap = 40
    step = max(1, window_size - overlap)
    marker_indexes = set()

    for window_start in range(0, total, step):
        window_end = min(total, window_start + window_size)
        window_lines = []

        for idx in range(window_start, window_end):
            seg = segments[idx]
            text = _normalize_text(seg.get("text", ""))
            if not text:
                continue

            ts = float(seg.get("start", 0.0))
            gap_before = float(seg.get("gap_before", 0.0))
            window_lines.append(f"{idx}|{ts:.2f}|gap_before={gap_before:.2f}|{text[:180]}")

        if not window_lines:
            continue

        prompt = (
            "You analyze audiobook transcripts and detect ONLY starts of chapter/part headings. "
            "A marker should be selected only for a chapter/part heading line (or its clear ASR-corrupted form). "
            "Treat likely ASR misspellings and near-homophones as valid clues when context strongly supports a heading, "
            "for example: 'клава пятая' should be interpreted as likely 'глава пятая', "
            "'чясть вторая' as likely 'часть вторая', and 'глва шестая' as likely 'глава шестая'. "
            "If a detected heading appears very early and would create the first chunk shorter than 90 seconds (typical title/music intro), do not mark it. "
            "For chapter/part headings, place the marker on the heading line itself so split happens BEFORE the heading, not after the previous sentence. "
            "Example: keep '... и они ушли. музыкальное интро ... <cut> Глава третья ...', "
            "and avoid '... и они ушли. <cut> музыкальное интро ... Глава третья ...'. "
            "A large gap_before value (for example >5s) usually means music/insert/pause before the next spoken line and is a strong structural clue. "
            "Do not mark regular paragraph transitions.\n\n"
            "Input format per line: index|start_seconds|gap_before=seconds|text\n"
            "Return STRICT JSON ONLY in this format:\n"
            "{\"markers\":[{\"index\":123,\"reason\":\"short reason\"}]}\n"
            "If nothing is found, return exactly: {\"markers\":[]}\n\n"
            f"Transcript lines:\n{chr(10).join(window_lines)}"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise JSON-only assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            indexes = _extract_marker_indexes(content, window_start, window_end)
            marker_indexes.update(indexes)
        except Exception as e:
            print(f"LLM chapter detection warning for window {window_start}-{window_end}: {e}")

    # Rule-based backup: keep explicit heading-like lines, especially after long pauses.
    backup_indexes = set()
    for idx, seg in enumerate(segments):
        text = _normalize_text(seg.get("text", ""))
        if not _is_heading_marker_text(text):
            continue

        gap_before = float(seg.get("gap_before", 0.0))
        tokens = _tokenize_text(text)
        explicit_heading_start = bool(tokens) and _is_heading_token(tokens[0]) and len(tokens) <= 14

        if gap_before >= 5.0 or explicit_heading_start:
            backup_indexes.add(idx)

    if backup_indexes:
        marker_indexes.update(backup_indexes)
        print(f"Added {len(backup_indexes)} heading marker(s) from rule-based backup.")

    markers = []
    dropped_non_headings = 0
    for idx in sorted(marker_indexes):
        seg = segments[idx]
        text = _normalize_text(seg.get("text", ""))
        if not _is_heading_marker_text(text):
            dropped_non_headings += 1
            continue
        markers.append({
            "index": idx,
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        })

    if dropped_non_headings:
        print(f"Filtered out {dropped_non_headings} non-heading marker(s) from LLM output.")

    # Keep one marker per very close timestamp region.
    deduped = []
    for marker in markers:
        if not deduped:
            deduped.append(marker)
            continue

        prev = deduped[-1]
        if marker["start"] - prev["start"] < 30.0:
            continue
        deduped.append(marker)

    return deduped


def split_by_markers(input_file, out_dir, base_name, markers):
    """Split audio using detected chapter/part boundaries."""
    cut_points = sorted({m["start"] for m in markers if m["start"] > 0.5})

    if cut_points and cut_points[0] < 90.0:
        print(f"Skipping early first marker at {cut_points[0]:.2f}s (intro shorter than 90s).")
        cut_points = cut_points[1:]

    current_start_t = 0.0
    part_idx = 1

    for cut_time in cut_points:
        if cut_time - current_start_t < 1.0:
            continue

        out_file = os.path.join(out_dir, f"{part_idx:03d}_{base_name}.mp3")
        split_audio_ffmpeg(input_file, current_start_t, cut_time, out_file)
        part_idx += 1
        current_start_t = cut_time

    out_file = os.path.join(out_dir, f"{part_idx:03d}_{base_name}.mp3")
    split_audio_ffmpeg(input_file, current_start_t, None, out_file)


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

    transcribe_attempts = [
        {
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        },
        {
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
        },
        {
            "beam_size": 5,
            "word_timestamps": True,
        },
        {
            "beam_size": 5,
        },
    ]

    segments = None
    info = None
    for attempt_idx, kwargs in enumerate(transcribe_attempts, 1):
        try:
            segments, info = GLOBAL_MODEL.transcribe(file_path, **kwargs)
            if attempt_idx > 1:
                print(f"[WARN] Fallback STT mode used: {kwargs}")
            break
        except TypeError as e:
            if attempt_idx == len(transcribe_attempts):
                raise
            print(f"[WARN] STT options not supported ({e}), retrying with simpler options...")

    if segments is None or info is None:
        raise RuntimeError("Transcription failed: model returned no segments/info")
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    print("Recognizing speech (this may take a while depending on PC power)...")
    
    total_mins = info.duration / 60.0
    results = []
    for segment in segments:
        raw_start = float(segment.start)
        raw_end = float(segment.end)

        words = getattr(segment, "words", None) or []
        word_starts = [float(w.start) for w in words if getattr(w, "start", None) is not None]
        word_ends = [float(w.end) for w in words if getattr(w, "end", None) is not None]

        speech_start = word_starts[0] if word_starts else raw_start
        speech_end = word_ends[-1] if word_ends else raw_end

        if speech_end < speech_start:
            speech_start, speech_end = raw_start, raw_end

        results.append({
            "start": speech_start,
            "end": speech_end,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "text": segment.text.strip(),
        })
        current_mins = raw_end / 60.0
        percent = (current_mins / total_mins) * 100
        print(f"\rAudio transcribed: {current_mins:.2f} of {total_mins:.2f} mins ({percent:.1f}% done)...", end="", flush=True)
        
    print("\nTranscription complete!")
    return _enrich_pause_metadata(results)

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
    content = response.choices[0].message.content or ""
    return content.strip()

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
        use_existing_stt = ask_yes_no("Use it? (y/n or +/- , default y): ", default=True)
        if use_existing_stt:
            print(f"Loading transcription from {transcription_file}...")
            with open(transcription_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
            print(f"Loaded {len(segments)} segments.")

    if segments is not None:
        segments = _enrich_pause_metadata(segments)
    
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

    print("\nChecking transcript for chapter/part boundaries via LLM...")
    structure_markers = find_structure_markers_with_llm(segments, args.api_base, args.llm_model)
    if structure_markers:
        print(f"\nFound {len(structure_markers)} possible chapter/part markers.")
        preview_count = min(5, len(structure_markers))
        for marker in structure_markers[:preview_count]:
            print(f"  - {marker['start']:.2f}s: {marker['text']}")
        if len(structure_markers) > preview_count:
            print(f"  ... and {len(structure_markers) - preview_count} more")

        split_by_chapters = ask_yes_no("Split by detected chapters/parts? (y/n or +/- , default y): ", default=True)
        if split_by_chapters:
            print("Splitting by detected chapter/part markers...")
            split_by_markers(args.input, args.out_dir, base_name, structure_markers)
            print("Done!")
            return

        print("Continuing with semantic timing-based splitting...")
    
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
                if i < len(context_segs) - 1:
                    pause = float(s.get("gap_after", context_segs[i + 1]["start"] - s["end"]))
                else:
                    pause = 0.0

                if pause >= 5.0:
                    context_text_parts.append(f"{s['text']} [LongPause: {pause:.1f} sec]")
                elif pause > 0.5:
                    context_text_parts.append(f"{s['text']} [Pause: {pause:.1f} sec]")
                else:
                    context_text_parts.append(s['text'])
            context_text = " ".join(context_text_parts)
            
            print(f"\n---\nFinding semantic break around {seg['end']:.2f}s ...")
            break_sentence = find_semantic_break(context_text, args.api_base, args.llm_model)
            print(f"LLM suggested break: '{break_sentence}'")
            
            cut_time = seg["end"]
            for s in reversed(context_segs):
                clean_break = _strip_pause_markers(break_sentence)
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
