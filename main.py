# main.py
import os
import torch
import soundfile as sf
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ruaccent
from huggingface_hub import hf_hub_download

# --- IMPORTANT: Add f5-tts to Python's path ---
# This allows us to import the custom model code
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'f5-tts'))

# --- Now, import from the f5-tts repository ---
from text import text_to_sequence
from models import SynthesizerTrn
import commons
import utils

# --- 1. Configuration and Model Loading ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Download model files from the Hugging Face Hub ---
print("Downloading model files...")
REPO_ID = "Misha24-10/F5-TTS_RUSSIAN"
try:
    # Download the config and vocab from the BASE model folder
    config_path = hf_hub_download(repo_id=REPO_ID, filename="F5TTS_v1_Base/config.json")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="F5TTS_v1_Base/vocab.txt")

    # Download the fine-tuned model weights from the V2 folder
    model_path = hf_hub_download(repo_id=REPO_ID, filename="F5TTS_v1_Base_v2/model_last.pt")
    
    # The downloaded files are just paths, we need to move the vocab file
    # to where the code expects it (inside the text folder)
    # This is a quirk of how this specific repo is structured.
    f5_tts_text_dir = os.path.join('f5-tts', 'text')
    expected_vocab_path = os.path.join(f5_tts_text_dir, 'vocab.txt')
    if not os.path.exists(expected_vocab_path):
        import shutil
        shutil.copy(vocab_path, expected_vocab_path)
        print(f"Copied vocab.txt to {expected_vocab_path}")

    print("✅ All necessary files downloaded.")

except Exception as e:
    print(f"❌ Failed to download files: {e}")
    config_path, model_path = None, None

# --- Load the model using the custom code from the repo ---
net_g = None
if config_path and model_path:
    print("Loading model using custom f5-tts code...")
    try:
        hps = utils.get_hparams_from_file(config_path)
        
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).to(DEVICE)
        
        _ = net_g.eval()
        _ = utils.load_checkpoint(model_path, net_g, None)
        
        print("✅ Fine-tuned model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model with custom code: {e}")
else:
    print("Skipping model loading due to download failure.")


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
    title="F5-TTS Russian Inference Service",
    description="Runs the fine-tuned Misha24-10/F5-TTS_RUSSIAN model.",
)

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_base64: str

@app.on_event("startup")
async def startup_event():
    if not net_g:
        raise RuntimeError("TTS Model did not load correctly. Please check logs.")
    print("✅ Application startup successful. Service is ready.")

@app.post("/synthesize", response_model=TTSResponse, summary="Synthesize Russian Speech")
async def synthesize_speech(request: TTSRequest):
    try:
        stressed_text_apostrophe = ruaccent.process_all(request.text)
        final_text = convert_ruaccent_to_plus(stressed_text_apostrophe)
        
        # Use the repo's text_to_sequence function
        stn_tst = text_to_sequence(final_text, hps.symbols, hps.data.text_cleaners)
        
        with torch.no_grad():
            x_tst = torch.LongTensor(stn_tst).to(DEVICE).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([len(stn_tst)]).to(DEVICE)
            
            # Perform inference
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.cpu().float().numpy()

        # Save to an in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio, hps.data.sampling_rate, format='WAV')
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return TTSResponse(audio_base64=audio_base64)

    except Exception as e:
        print(f"An error occurred during synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
