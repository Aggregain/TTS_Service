# main.py
import torch
import soundfile as sf
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VitsModel
import ruaccent
from huggingface_hub import snapshot_download # <-- Import the downloader

# --- 1. Configuration and Model Loading ---

# Check for CUDA availability for faster inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define model details
REPO_ID = "Misha24-10/F5-TTS_RUSSIAN"
MODEL_SUBFOLDER = "F5TTS_v1_Base_v2"
SAMPLING_RATE = 48000

# --- Download the specific model subfolder ---
print(f"Downloading model files from subfolder: {MODEL_SUBFOLDER}...")
try:
    # This function downloads only the necessary files into a local cache
    # and returns the path to that directory.
    local_model_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=f"{MODEL_SUBFOLDER}/**" # Pattern to grab all files in the subfolder
    )
    print(f"✅ Model files downloaded to: {local_model_path}")

    # --- Load the model from the local path ---
    print("Loading model from local path...")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    model = VitsModel.from_pretrained(local_model_path, trust_remote_code=True).to(DEVICE)
    print("✅ TTS model and processor loaded successfully.")

except Exception as e:
    print(f"❌ An error occurred during model loading: {e}")
    processor = None
    model = None


# --- 2. Helper Function for Stress Format Conversion ---

def convert_ruaccent_to_plus(text_with_apostrophe: str) -> str:
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    words = text_with_apostrophe.split(' ')
    processed_words = []
    for word in words:
        if "'" in word:
            pos = word.find("'")
            if pos > 0 and word[pos - 1] in vowels:
                new_word = word[:pos - 1] + '+' + word[pos - 1] + word[pos + 1:]
                processed_words.append(new_word)
            else:
                processed_words.append(word.replace("'", ""))
        else:
            processed_words.append(word)
    return ' '.join(processed_words)

# --- 3. FastAPI Service Setup ---

app = FastAPI(
    title="Russian TTS Service",
    description="A service for Russian Text-to-Speech using F5-TTS and RuAccent.",
)

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_base64: str
    text_processed: str

@app.on_event("startup")
async def startup_event():
    if not all([processor, model]):
        raise RuntimeError("TTS Model did not load correctly. Please check logs for errors.")
    print("✅ Application startup successful. Service is ready.")


@app.post("/synthesize", response_model=TTSResponse, summary="Synthesize Russian Speech")
async def synthesize_speech(request: TTSRequest):
    try:
        stressed_text_apostrophe = ruaccent.process_all(request.text)
        final_text = convert_ruaccent_to_plus(stressed_text_apostrophe)
        inputs = processor(text=final_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.squeeze().cpu().numpy()

        buffer = io.BytesIO()
        sf.write(buffer, waveform, SAMPLING_RATE, format='WAV')
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return TTSResponse(audio_base64=audio_base64, text_processed=final_text)

    except Exception as e:
        print(f"An error occurred during synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
