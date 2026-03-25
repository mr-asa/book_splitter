# Semantic Audiobook Splitter

A local Python utility that uses **faster-whisper** and a local **LLM** (via LM Studio/Ollama) to intelligently split audiobook files (`.mp3`, `.m4b`) into semantic chapters based on the narrative and pause durations.

Instead of splitting strictly by time (which often cuts off words or scenes abruptly), this script transcribes the audio, finds logical scene breaks (e.g., end of a dialogue, paragraph ends, or long pauses), and slices the audio using `ffmpeg`.

## Requirements
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in the system PATH.
- [LM Studio](https://lmstudio.ai/) running an open-source LLM on the default port `1234`.

## Installation & Usage
Simply double-click `run.bat` in Windows. 

1. It will automatically create a Python virtual environment and install the required packages (`faster-whisper`, `openai`).
2. On the first run, the script will prompt you for the path to your audiobook and generate a `config.json` file.
3. The script will transcribe the file and output the chunked MP3 files into the `output/` directory.

### Configuration (`config.json`)
You can tweak the settings manually in `config.json`:
- `input_file`: The absolute path to your audiobook.
- `out_dir`: Name of the directory where chunks will be saved (default: `"output"`).
- `target_mins`: Target length of each chunk in minutes (default: `7.0`). The script will look for a logical break near this accumulated time.
- `api_base`: The API URL for your local LLM (default: LM Studio `http://localhost:1234/v1`).
- `llm_model`: LLM model identifier (default: `"local-model"`).
- `whisper_model`: The faster-whisper model to use (default: `"turbo"`).
- `debug`: Enable debug mode to write execution and crash tracebacks to disk (default: `false`).

## How It Works
1. **Transcription**: The script converts the audio to timestamped text using `faster-whisper`.
2. **Context Assembly**: When the accumulated audio reaches `target_mins`, the script gathers the latest 2 minutes of text, annotating it with silence gaps (e.g., `[Pause: 1.5 sec]`).
3. **Semantic Break Identification**: The text is sent to the local LLM. The LLM analyzes the context and selects the best sentence to split the scene without interrupting the flow of thoughts.
4. **Audio Slicing**: FFmpeg precisely slices the MP3 at the exact timestamp of the identified sentence.
