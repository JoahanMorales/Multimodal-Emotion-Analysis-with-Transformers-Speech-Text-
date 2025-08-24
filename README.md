# Multimodal Emotion Analysis with Transformers (Speech + Text)

This repository contains a prototype for **multimodal emotion recognition**, integrating **speech and text** signals using Transformer-based models.  
The system combines **speech-to-text transcription**, **emotion classification**, **audio-based emotion recognition**, and **automatic translation**, along with correlation, clustering, and visualization tools.

---

## ✨ Features

- **Multimodal approach**: integrates **speech (audio)** and **text**.
- **Transformer-based models**:
  - [Whisper-small (ASR)](https://huggingface.co/openai/whisper-small) → speech-to-text.
  - [RoBERTuito-base-uncased-emotion](https://huggingface.co/pysentimiento/robertuito-base-uncased-emotion) → text emotion recognition (Spanish).
  - [Wav2Vec2 emotion recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) → audio emotion recognition.
- **Data analysis**:
  - Correlations (Pearson / Spearman).
  - Cosine similarity between emotion embeddings.
  - PCA & K-Means clustering.
- **Visualization**:
  - Probability bars, heatmaps, time-series plots.
- **Translation**: optional automatic translation of results into English (`googletrans`).

---

## 🔧 Requirements

- Python 3.9+
- (Optional) CUDA GPU for faster inference with Whisper / Wav2Vec2

### Install dependencies

```bash
pip install -r requirements.txt
▶️ Usage
Command Line
bash
Copiar
Editar
python src/main.py --audio_path data/audio/sample.wav --save_dir outputs --translate_en true
Minimal Python Example
python
Copiar
Editar
import torch
from transformers import pipeline
from googletrans import Translator

# Device
device = 0 if torch.cuda.is_available() else -1

# 1) Whisper ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
asr_out = asr("data/audio/sample.wav")
text = asr_out["text"]

# 2) Text emotions (RoBERTuito)
emo_text = pipeline("text-classification", model="pysentimiento/robertuito-base-uncased-emotion", top_k=None)
emo_text_out = emo_text(text)

# 3) Audio emotions (Wav2Vec2)
emo_audio = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device)
emo_audio_out = emo_audio("data/audio/sample.wav")

# 4) Translate
translator = Translator()
text_en = translator.translate(text, src='es', dest='en').text

print("Transcription:", text)
print("Text Emotion:", emo_text_out)
print("Audio Emotion:", emo_audio_out)
print("Translation (EN):", text_en)
📊 Analysis & Visualization
Correlation between text and audio results.

Cosine similarity for emotion embeddings.

PCA for dimensionality reduction.

K-Means for clustering.

Graphs saved under outputs/figures/.

⚠️ Ethical Note
This project is for academic purposes only.
It must not replace professional mental health services.
Any application in psychology must ensure privacy, consent, and human supervision.

📌 References
Vaswani, A., et al. (2017). Attention is all you need. NeurIPS. https://arxiv.org/abs/1706.03762

Lian, Z., & Tao, J. (2021). CTNet: Conversational Transformer Network for Emotion Recognition. IEEE/ACM TASLP, 29, 985–1000. https://doi.org/10.1109/TASLP.2021.3049898

Whisper-small (ASR model)

RoBERTuito-base-uncased-emotion

Wav2Vec2 emotion recognition

🧾 License
MIT License
