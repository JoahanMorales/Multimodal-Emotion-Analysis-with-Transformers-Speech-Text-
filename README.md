# Multimodal Emotion Analysis with Transformers (Speech + Text)

This repository contains a prototype for **multimodal emotion recognition and analysis**, integrating speech and text signals using **Transformer-based models**. The system combines **speech-to-text transcription**, **emotion classification**, **audio analysis**, and **translation**, along with correlation, clustering, and visualization tools.

---

## ✨ Features

- **Multimodal approach**: integrates **speech** (audio) and **text**.
- **Transformer-based models**:
  - [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) for speech-to-text (ASR).
  - [`pysentimiento/robertuito-base-uncased-emotion`](https://huggingface.co/pysentimiento/robertuito-base-uncased-emotion) for text emotion recognition in Spanish.
  - [`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) for audio emotion recognition.
- **Data analysis**:
  - Correlations (**Pearson / Spearman**) between text and audio emotion scores.
  - **Cosine similarity** to compare emotion embeddings.
  - **Scaling/normalization** with MinMax or StandardScaler.
  - **Smoothing** (Savitzky–Golay), **PCA**, **K-Means clustering**.
- **Visualization**:
  - Probability bar plots, heatmaps, and time-series graphs.
- **Translation**: optional automatic translation of results into English using `googletrans`.

---

## 📁 Project Structure (suggested)

