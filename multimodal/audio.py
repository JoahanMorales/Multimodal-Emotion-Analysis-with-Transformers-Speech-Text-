import soundfile as sf
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import librosa
from scipy import signal

print("Cargando modelo de clasificación de emociones...")
try:
    classifier_audio = pipeline(
        "audio-classification", 
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    print("✓ Usando modelo avanzado con 8 emociones")
    usar_modelo_avanzado = True
except Exception as e:
    print(f"⚠️ Error con modelo avanzado: {e}")
    try:
        classifier_audio = pipeline(
            "audio-classification", 
            model="superb/wav2vec2-base-superb-er"
        )
        print("✓ Usando modelo básico con 4 emociones")
        usar_modelo_avanzado = False
    except Exception as e2:
        print(f"❌ Error con ambos modelos: {e2}")
        classifier_audio = pipeline(
            "audio-classification", 
            model="harshit345/xlsr-wav2vec-speech-emotion-recognition"
        )
        print("✓ Usando modelo alternativo de emociones")
        usar_modelo_avanzado = False

# Leer y preprocesar audio
audio_path = "Experiencia_1.wav"
print(f"Procesando audio: {audio_path}")

audio, samplerate = librosa.load(audio_path, sr=16000)

def preprocess_audio(audio, sr):
    """Preprocesar audio para mejor análisis emocional"""
    audio = librosa.util.normalize(audio)
    
    nyquist = sr // 2
    high_pass_freq = 80
    b, a = signal.butter(3, high_pass_freq / nyquist, btype='high')
    audio = signal.filtfilt(b, a, audio)
    
    window_size = int(0.01 * sr)
    if len(audio) > 2 * window_size:
        fade_in = np.linspace(0, 1, window_size)
        fade_out = np.linspace(1, 0, window_size)
        audio[:window_size] *= fade_in
        audio[-window_size:] *= fade_out
    
    return audio

audio = preprocess_audio(audio, samplerate)

# Configuración de fragmentos
num_fragmentos = 8
duracion = len(audio) / samplerate
duracion_individual = duracion / num_fragmentos
overlap = 0.2
step = duracion_individual * (1 - overlap)

print(f"Duración total: {duracion:.1f}s")
print(f"Analizando {num_fragmentos} fragmentos de {duracion_individual:.1f}s cada uno")
print("Procesando fragmentos...")

resultados = []
segment_times = []
valid_fragments = 0

for i in range(num_fragmentos):
    start_time = i * step
    end_time = start_time + duracion_individual
    
    if end_time > duracion:
        end_time = duracion
    
    start_sample = int(start_time * samplerate)
    end_sample = int(end_time * samplerate)
    fragmento = audio[start_sample:end_sample]
    
    if len(fragmento) < samplerate * 0.5:
        continue
    
    try:
        result = classifier_audio(fragmento, sampling_rate=samplerate)
        resultados.append(result)
        segment_times.append(start_time + duracion_individual/2)
        valid_fragments += 1
        print(f"  Fragmento {valid_fragments}: {start_time:.1f}s - {end_time:.1f}s")
        
    except Exception as e:
        print(f"  Error en fragmento {i+1}: {e}")
        continue

if not resultados:
    print("Error: No se pudieron procesar fragmentos de audio")
    exit()

print(f"✓ Procesados {valid_fragments} fragmentos exitosamente")

# Mapeo de emociones
emociones_original = [r['label'] for r in resultados[0]]

etiquetas_es_avanzado = {
    'angry': 'Enojado',
    'disgust': 'Disgusto', 
    'fear': 'Miedo',
    'fearful': 'Miedo',
    'happy': 'Feliz',
    'neutral': 'Neutral',
    'sad': 'Triste',
    'surprise': 'Sorpresa',
    'surprised': 'Sorpresa',
    'calm': 'Calma'
}

etiquetas_es_basico = {
    'ang': 'Enojado', 
    'neu': 'Neutral', 
    'hap': 'Feliz', 
    'sad': 'Triste'
}

# Determinar qué diccionario usar
if any(emo in ['angry', 'disgust', 'fear'] for emo in emociones_original):
    etiquetas_es = etiquetas_es_avanzado
    print("✓ Usando mapeo avanzado de emociones")
else:
    etiquetas_es = etiquetas_es_basico
    print("✓ Usando mapeo básico de emociones")

# Preparar datos para graficar
todas_emociones = set()
for res in resultados:
    for r in res:
        if r['label'] in etiquetas_es:
            emo_es = etiquetas_es[r['label']]
        else:
            emo_es = r['label'].capitalize()
        todas_emociones.add(emo_es)

todas_emociones = list(todas_emociones)
print(f"Todas las emociones encontradas: {todas_emociones}")

# Inicializar datos
scores_por_emocion = {}
for emo in todas_emociones:
    scores_por_emocion[emo] = [0]

segment_times_plot = [0] + segment_times

# Procesar cada fragmento
for i, res in enumerate(resultados):
    scores_fragmento = {}
    for r in res:
        if r['label'] in etiquetas_es:
            emo_es = etiquetas_es[r['label']]
        else:
            emo_es = r['label'].capitalize()
        scores_fragmento[emo_es] = r['score']
    
    for emo in todas_emociones:
        if emo in scores_fragmento:
            scores_por_emocion[emo].append(scores_fragmento[emo])
        else:
            scores_por_emocion[emo].append(0.0)

# Colores mejorados
colores = {
    'Enojado': '#FF4444',
    'Neutral': '#808080',
    'Feliz': '#44AA44',
    'Triste': '#4444FF',
    'Disgusto': '#AA4444',
    'Miedo': '#AA44AA',
    'Sorpresa': '#FF8800',
    'Calma': '#44AAAA'
}

# CREAR GRÁFICOS
plt.figure(figsize=(16, 8))

# === GRÁFICO 1: ESCALA AUTOMÁTICA - TODAS LAS EMOCIONES ===
plt.subplot(1, 2, 1)
for emo in todas_emociones:
    color = colores.get(emo, '#000000')
    plt.plot(segment_times_plot, scores_por_emocion[emo], 
            marker='o', label=emo, color=color, 
            linewidth=2.5, markersize=7, alpha=0.8)

plt.xlabel("Tiempo (segundos)", fontsize=12)
plt.ylabel("Nivel de Confianza", fontsize=12)
# Calcular límites automáticos con un poco de margen - TODAS las emociones
max_score = max(max(scores_por_emocion[emo]) for emo in todas_emociones)
min_score = min(min(scores_por_emocion[emo]) for emo in todas_emociones)
margen = (max_score - min_score) * 0.1
plt.ylim(max(0, min_score - margen), max_score + margen)

plt.title("Todas las Emociones - Escala Automática", fontweight='bold', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# === GRÁFICO 2: SOLO EMOCIONES DOMINANTES ===
plt.subplot(1, 2, 2)
# Filtrar MUY estrictamente - solo emociones realmente dominantes
emociones_significativas = []
umbral_estricto = 0.12  # Solo emociones con score > 12% (muy estricto)

for emo in todas_emociones:
    scores = scores_por_emocion[emo][1:]  # Excluir el 0 inicial
    max_score = max(scores) if scores else 0
    promedio = np.mean(scores) if scores else 0
    
    # Criterio MUY estricto: score máximo > 12% Y promedio > 8%
    if max_score > umbral_estricto and promedio > 0.08:
        emociones_significativas.append(emo)

print(f"Emociones REALMENTE dominantes (Max>{umbral_estricto:.0%} Y Prom>8%): {emociones_significativas}")

# Si no hay emociones que cumplan el criterio estricto, usar las 2-3 más altas
if len(emociones_significativas) == 0:
    print("⚠️ Ninguna emoción cumple criterio estricto. Mostrando las 3 más altas...")
    # Calcular score promedio para cada emoción
    promedios = {}
    for emo in todas_emociones:
        scores = scores_por_emocion[emo][1:]
        promedios[emo] = np.mean(scores) if scores else 0
    
    # Tomar las 3 emociones con mayor promedio
    emociones_significativas = sorted(promedios.keys(), 
                                    key=lambda x: promedios[x], 
                                    reverse=True)[:3]
    print(f"Las 3 emociones más altas: {emociones_significativas}")

# SOLO graficar las emociones verdaderamente significativas
for emo in emociones_significativas:
    color = colores.get(emo, '#000000')
    plt.plot(segment_times_plot, scores_por_emocion[emo], 
            marker='o', label=emo, color=color, 
            linewidth=4, markersize=10, alpha=1.0)  # Líneas más gruesas

plt.xlabel("Tiempo (segundos)", fontsize=12)
plt.ylabel("Nivel de Confianza", fontsize=12)

# Escala automática solo para emociones dominantes
if emociones_significativas:
    max_score_dom = max(max(scores_por_emocion[emo]) for emo in emociones_significativas)
    min_score_dom = min(min(scores_por_emocion[emo]) for emo in emociones_significativas)
    margen_dom = (max_score_dom - min_score_dom) * 0.1
    plt.ylim(max(0, min_score_dom - margen_dom), max_score_dom + margen_dom)

plt.title("Solo Emociones Dominantes", fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Información adicional
plt.figtext(0.02, 0.02, f'Duración: {duracion:.1f}s | Fragmentos: {valid_fragments} | '
                        f'Umbral dominancia: >{umbral_estricto:.0%}', fontsize=10, alpha=0.7)

plt.show()

# === ESTADÍSTICAS MEJORADAS ===
print("\n" + "="*60)
print("ANÁLISIS DE EMOCIONES")
print("="*60)

for emo in todas_emociones:
    if emo in scores_por_emocion and len(scores_por_emocion[emo]) > 1:
        scores = scores_por_emocion[emo][1:]  # Excluir el 0 inicial
        promedio = np.mean(scores)
        maximo = np.max(scores)
        
        # Calcular porcentaje del tiempo que esta emoción fue dominante
        veces_dominante = 0
        for i in range(1, len(segment_times_plot)):
            scores_momento = [scores_por_emocion[e][i] for e in todas_emociones]
            if max(scores_momento) == scores_por_emocion[emo][i] and scores_por_emocion[emo][i] > 0:
                veces_dominante += 1
        
        porcentaje_dominante = (veces_dominante / len(scores)) * 100
        
        # Marcar si es una emoción significativa
        es_dominante = "🎯" if emo in emociones_significativas else "  "
        
        print(f"{es_dominante} {emo:12}: Promedio={promedio:.3f} | Máximo={maximo:.3f} | "
              f"Dominante={porcentaje_dominante:.1f}% del tiempo")

print("="*60)
print("🎯 = Emociones con presencia significativa")
print("="*60)