import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment
import math
import os

AUDIO_PATH = '/content/drive/My Drive/Whisper Hokkien/test/Hokkien conversation.mp3'
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "Chinese"
TASK = "transcribe"
CHECKPOINT_FOLDER = "checkpoints"


processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE, task=TASK)


def load_latest_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir)
             if "-" in f and f.split("-")[-1].isdigit()],
            key=lambda x: int(x.split("-")[-1])
        )
        print(checkpoints)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f'loading {latest_checkpoint}')
            return os.path.join(checkpoint_dir, latest_checkpoint)
    return None


def split_audio(audio_path, chunk_length_ms=29000):
    audio = AudioSegment.from_file(audio_path)
    audio_length_ms = len(audio)
    num_chunks = math.ceil(audio_length_ms / chunk_length_ms)

    chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, audio_length_ms)
        chunk = audio[start_time:end_time]
        chunk_file = f"chunk_{i}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunks.append(chunk_file)

    return chunks


def test_model(model_id, audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_checkpoint,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    chunks = split_audio(audio_path)
    print(chunks)

    full_transcription = ""
    for chunk in chunks:
        result = pipe(chunk)
        print(result)
        full_transcription += result["text"] + " "
        # os.remove(chunk)

    return full_transcription.strip()


if __name__ == '__main__':
    model_checkpoint = load_latest_checkpoint(CHECKPOINT_FOLDER)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_checkpoint, device_map="auto")
    transcript = test_model(model_checkpoint, AUDIO_PATH)

    print(transcript)
