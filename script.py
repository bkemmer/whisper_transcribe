from faster_whisper import WhisperModel
from pathlib import Path
from time import perf_counter
from datetime import datetime

DATA_PATH = Path('data')
OUTPUT_PATH = Path('transcription.md')

model_size = "large-v3"

print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
t0 = perf_counter()
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print(f"Model loaded in {perf_counter() - t0:.2f}s")
t_transcribe = perf_counter()

REPLACEMENTS = {
    "caralho": "Carol",
    "cara": "Carol",
}

def apply_replacements(text):
    # for wrong, correct in REPLACEMENTS.items():
    #     text = text.replace(wrong, correct)
    return text

lines = []

for p in sorted(DATA_PATH.iterdir()):
    print(f"Running file {p}")
    t_iter = perf_counter()
    segments, info = model.transcribe(audio=str(p), language='pt', beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    lines.append(f"## {p.name}\n")
    lines.append(f"*Language: {info.language} ({info.language_probability:.2f})*\n\n")

    for segment in segments:
        line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, apply_replacements(segment.text))
        print(line)
        lines.append(line + "\n")

    print(f"File {p.name} transcribed in {perf_counter() - t_iter:.2f}s")
    lines.append("\n")

OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
print(f"\nTranscription saved to {OUTPUT_PATH}")
print(f"Total transcription time: {perf_counter() - t_transcribe:.2f}s")
