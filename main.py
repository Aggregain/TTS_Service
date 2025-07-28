import torch
import soundfile as sf
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, VitsModel
from ruaccent import RuAccent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "Misha24-10/F5-TTS_RUSSIAN"
MODEL_SUBFOLDER = "F5TTS_v1_Base_v2"
SAMPLING_RATE = 48000  # From the model's config.json

print("Loading RuAccent model...")
try:
    accentizer = RuAccent()
    accentizer.load()
    print("✅ RuAccent model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading RuAccent model: {e}")
    accentizer = None

print(f"Loading TTS model: {MODEL_ID}/{MODEL_SUBFOLDER}...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER)
    model = VitsModel.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER).to(DEVICE)
    print("✅ TTS model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading TTS model: {e}")
    processor = None
    model = None

def convert_ruaccent_to_plus(text_with_apostrophe: str) -> str:
    """
    Converts text stressed with an apostrophe (e.g., "приве'т") to
    text stressed with a plus sign before the vowel (e.g., "прив+ет"),
    which is the format the F5-TTS model expects.
    """
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
    if not all([accentizer, processor, model]):
        raise RuntimeError("Models did not load correctly. Please check logs.")

@app.post("/synthesize", response_model=TTSResponse, summary="Synthesize Russian Speech")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesizes speech from Russian text.
    - Applies automatic stress using RuAccent.
    - Generates audio using the F5-TTS model.
    - Returns the audio as a base64 encoded WAV string.
    """
    try:
        stressed_text_apostrophe = accentizer.process_all(request.text)

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
