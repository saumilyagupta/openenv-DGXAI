# Cell 09 — Audio pipeline

Kokoro-82M text-to-speech and faster-whisper-small automatic-speech-recognition
wrappers that sit at the env boundary. Per `docs/modules/audio.md`, both
engines are process-wide singletons with lazy dep loading and an LRU cache on
the TTS path; the training loop never imports this cell (`§6.3`).
