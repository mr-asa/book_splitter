import sys
import os

sys.argv = ['split_audio.py', '--input', 'test.mp3', '--target-mins', '7.0']
import split_audio

def mock_transcribe(x):
    print("Mock transcribe")
    return [{"start": i*20.0, "end": (i+1)*20.0, "text": f"Text {i}"} for i in range(50)]

def mock_find(chunk, api_base, model_name):
    print("Mock find_semantic_break")
    return "Text 20"

def mock_split(a,b,c,d):
    print(f"Mock split {a} from {b} to {c} -> {d}")

split_audio.transcribe_audio = mock_transcribe
split_audio.find_semantic_break = mock_find
split_audio.split_audio_ffmpeg = mock_split

try:
    split_audio.main()
except Exception as e:
    print(f"Exception caught in main: {e}")
    import traceback
    traceback.print_exc()
