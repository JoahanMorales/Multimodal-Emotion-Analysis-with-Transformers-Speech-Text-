# Multimodal Emotion Analysis with Transformers (Speech + Text)

This repository contains a prototype for **multimodal emotion recognition**, integrating **speech and text** signals using Transformer-based models.  
The system combines **speech-to-text transcription**, **emotion classification**, **audio-based emotion recognition**, and **automatic translation**, along with correlation, clustering, and visualization tools.

---

##  Features

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

##  Requirements

- Python 3.9+
- (Optional) CUDA GPU for faster inference with Whisper / Wav2Vec2

### Install dependencies

```bash
pip install -r requirements.txt
