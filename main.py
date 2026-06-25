import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ------------------------------------------------------------------ #
# Graceful imports for optional audio dependencies
# ------------------------------------------------------------------ #
try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
CLASSES      = ["Siren", "Dog_Bark", "Drilling", "Street_Music"]
SR           = 22050
DURATION     = 2.0  # seconds
N_MFCC       = 20
N_SAMPLES_PER_CLASS = 120

# ------------------------------------------------------------------ #
# High-Quality Synthesized Audio Generators for Urban Scene Classes
# ------------------------------------------------------------------ #
def synthesize_urban_wave(class_name, duration=DURATION, sr=SR, rng=None):
    if rng is None:
        rng = np.random.RandomState()
        
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    if class_name == "Siren":
        # Frequency sweeps oscillating between 600 Hz and 1200 Hz
        # 1.2 cycles of sweep per second
        inst_freq = 900.0 + 300.0 * np.sin(2.0 * np.pi * 1.2 * t)
        phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
        # High harmonics to simulate horn distortion
        signal = np.sin(phase) + 0.35 * np.sin(2.0 * phase) + 0.15 * np.sin(3.0 * phase)
        
    elif class_name == "Dog_Bark":
        # Multiple quick barks at onsets 0.2s, 0.7s, 1.2s, 1.7s
        onsets = [0.2, 0.7, 1.2, 1.7]
        for start_t in onsets:
            start_idx = int(start_t * sr)
            decay_len = 0.25
            end_idx = min(start_idx + int(decay_len * sr), len(signal))
            t_bark = t[start_idx:end_idx] - start_t
            
            # 1. Bark Vocalization Pitch Sweep (160 Hz -> 80 Hz)
            f_sweep = 80.0 + 80.0 * np.exp(-12.0 * t_bark)
            phase = 2.0 * np.pi * np.cumsum(f_sweep) / sr
            vocal = np.sin(phase) + 0.5 * np.sin(2.0 * phase)
            
            # 2. Harsh Bark Noise Burst (bandpass-filtered noise)
            noise = rng.normal(0, 1.5, len(t_bark))
            noise = np.convolve(noise, [1.0, -0.1, -0.4], mode='same')
            
            # 3. Envelope: sharp attack, exponential decay
            envelope = np.exp(-24.0 * t_bark)
            bark_wave = (0.45 * vocal + 0.55 * noise) * envelope
            signal[start_idx:end_idx] += bark_wave

    elif class_name == "Drilling":
        # Constant low-frequency sawtooth mechanical hum (85 Hz)
        f_engine = 85.0
        phase = 2.0 * np.pi * f_engine * t
        engine_wave = (phase % (2.0 * np.pi)) / (2.0 * np.pi) - 0.5
        
        # High-frequency hammer grinding pulses (periodic noise at 14 Hz)
        grind_noise = rng.normal(0, 0.35, len(t))
        grind_noise = np.convolve(grind_noise, [1.0, -0.95], mode='same')  # high-pass hiss
        # Amplitude modulator for drill hammer beats
        grind_mod = 0.5 * (1.2 + np.sin(2.0 * np.pi * 14.0 * t))
        
        signal = 0.35 * engine_wave + 0.65 * grind_noise * grind_mod

    else: # Street_Music
        # A simple, repeating melodic arpeggio playing on a clean sine wave
        # Notes: C4 (261.63), E4 (329.63), G4 (392.00), C5 (523.25)
        note_dur = 0.25 # seconds per note
        note_freqs = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63]
        
        for step in range(8):
            t_start = step * note_dur
            start_idx = int(t_start * sr)
            if start_idx >= len(signal): break
            end_idx = min(int((t_start + note_dur) * sr), len(signal))
            
            f_note = note_freqs[step % len(note_freqs)]
            t_note = t[start_idx:end_idx]
            
            # Generate tone
            phase = 2.0 * np.pi * f_note * t_note
            tone = np.sin(phase) + 0.25 * np.sin(2.0 * phase)
            
            # Note envelope (soft attack, decay)
            envelope = np.minimum(1.0, (t_note - t_start) * 50.0) * np.exp(-3.0 * (t_note - t_start))
            signal[start_idx:end_idx] = tone * envelope

        # Add background urban ambient low-frequency hum (30 Hz + soft noise)
        ambient_hum = 0.08 * np.sin(2.0 * np.pi * 30.0 * t)
        ambient_noise = 0.03 * rng.normal(0, 1.0, len(t))
        signal += ambient_hum + ambient_noise

    # Rescale signal amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
        
    return signal.astype(np.float32)

# ------------------------------------------------------------------ #
# Mel-Frequency Cepstral Coefficients Extractor (Pure-Numpy)
# ------------------------------------------------------------------ #
def extract_mfcc_numpy(signal, sr=SR, n_mfcc=N_MFCC):
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    n_fft     = 512
    n_filters = 40
    pre_emph  = 0.97
    
    # 1. Pre-emphasis
    emph = np.concatenate([[signal[0]], signal[1:] - pre_emph * signal[:-1]])
    
    # 2. Framing
    n_frm  = 1 + (len(emph) - frame_len) // hop_len
    if n_frm <= 0:
        return np.zeros((1, n_mfcc), dtype=np.float32)
        
    frames = np.stack([emph[i*hop_len : i*hop_len+frame_len] * np.hamming(frame_len)
                       for i in range(n_frm)])
                       
    # 3. Power Spectrum
    pow_sp = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    
    # 4. Mel Filterbank
    def hz2mel(h): return 2595.0 * np.log10(1.0 + h / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    
    mel_pts = np.linspace(hz2mel(0.0), hz2mel(sr / 2.0), n_filters + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, cen, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, cen):
            if cen > lo: fb[m-1, k] = (k - lo) / (cen - lo)
        for k in range(cen, hi):
            if hi > cen: fb[m-1, k] = (hi - k) / (hi - cen)
            
    # Apply filterbank
    mel_sp = np.dot(pow_sp, fb.T)
    mel_sp = np.where(mel_sp == 0, 1e-10, mel_sp)
    log_ml = np.log(mel_sp)
    
    # 5. Discrete Cosine Transform (DCT-II Orthonormal)
    N   = log_ml.shape[1]
    k   = np.arange(n_mfcc, dtype=np.float64)
    n   = np.arange(N, dtype=np.float64)
    cos = np.cos(np.pi * k[:, None] * (2.0 * n[None, :] + 1) / (2.0 * N))
    out = np.dot(log_ml, cos.T)
    out[:, 0]  *= np.sqrt(1.0 / (4.0 * N))
    out[:, 1:] *= np.sqrt(1.0 / (2.0 * N))
    
    return out.astype(np.float32)

# ------------------------------------------------------------------ #
# Unified Feature Extraction Wrapper
# ------------------------------------------------------------------ #
def extract_mfcc_sequence(signal, sr=SR, n_mfcc=N_MFCC):
    if LIBROSA_OK:
        # Match hop size of 10ms (220 samples at 22050 Hz)
        mfccs = librosa.feature.mfcc(y=signal.astype(np.float32), sr=sr, n_mfcc=n_mfcc,
                                     n_fft=512, hop_length=220)
        return mfccs.T.astype(np.float32)
    else:
        return extract_mfcc_numpy(signal, sr=sr, n_mfcc=n_mfcc)

# ------------------------------------------------------------------ #
# Main Entry Point
# ------------------------------------------------------------------ #
def main():
    print("====================================================")
    print("Project 58: Audio Scene Classification (UrbanSound)")
    print("====================================================")
    print(f"  librosa     : {'available' if LIBROSA_OK else 'NOT installed (using Numpy fallback)'}\n")

    # 1. Synthesize Audio Scenes
    print("Step 1: Generating synthetic urban audio database...")
    rng = np.random.RandomState(42)
    audio_data, labels = [], []
    
    for c_idx, class_name in enumerate(CLASSES):
        for _ in range(N_SAMPLES_PER_CLASS):
            sig = synthesize_urban_wave(class_name, duration=DURATION, rng=rng)
            sig *= rng.uniform(0.75, 1.0)
            
            audio_data.append(sig)
            labels.append(c_idx)
            
    # Extract sequential features: shape will be (samples, timesteps, n_mfcc, 1)
    temp_feat = extract_mfcc_sequence(audio_data[0])
    time_steps = temp_feat.shape[0]
    
    X = np.zeros((len(audio_data), time_steps, N_MFCC, 1), dtype=np.float32)
    for idx, wave in enumerate(audio_data):
        feat = extract_mfcc_sequence(wave)
        # Handle tiny shape differences if hop rounding shifts
        min_steps = min(time_steps, feat.shape[0])
        X[idx, :min_steps, :, 0] = feat[:min_steps, :]
        
    y = np.array(labels, dtype=np.int32)
    print(f"  Spectrogram Tensor Shape: {X.shape}")
    print(f"  Classes                 : {CLASSES}\n")

    # 2. Train/Val/Test Split
    def strat_split(X_data, y_data, ratio, seed):
        rng_split = np.random.RandomState(seed)
        tr_idx, te_idx = [], []
        for c in np.unique(y_data):
            idx = rng_split.permutation(np.where(y_data == c)[0])
            n = max(1, int(len(idx) * ratio))
            te_idx.extend(idx[:n])
            tr_idx.extend(idx[n:])
        return X_data[tr_idx], X_data[te_idx], y_data[tr_idx], y_data[te_idx]

    X_tv, X_test, y_tv, y_test = strat_split(X, y, 0.20, 101)
    X_train, X_val, y_train, y_val = strat_split(X_tv, y_tv, 0.15, 202)
    
    # 3. Z-Score Spectrogram Normalization
    mean_val = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_val  = X_train.std(axis=(0, 1, 2), keepdims=True)
    std_val  = np.where(std_val == 0, 1.0, std_val)
    
    X_train = (X_train - mean_val) / std_val
    X_val   = (X_val - mean_val) / std_val
    X_test  = (X_test - mean_val) / std_val
    
    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}\n")

    # 4. Build & Train 2D CNN Network
    print("Step 2: Constructing 2D CNN Environmental Sound Classifier...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, N_MFCC, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ], name="UrbanSoundCNN")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    EPOCHS, BATCH_SIZE = 25, 32
    print(f"\nStep 3: Training 2D CNN for {EPOCHS} epochs...")
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        shuffle=True,
                        verbose=1)

    # Save model
    model.save("audio_scene_cnn_model.keras")
    print("\n[OK] Model saved as 'audio_scene_cnn_model.keras'")

    # Evaluate model
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"[OK] Evaluation Test Accuracy: {test_acc*100:.1f}%\n")

    # 5. Draw Results Grid Dashboard
    print("Step 4: Drawing results dashboard...")
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ── Panel 1: Accuracy & Loss Curves ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['accuracy'], color='#2ecc71', linewidth=2, label='Train Acc')
    ax1.plot(history.history['val_accuracy'], color='#3498db', linewidth=2, label='Val Acc')
    ax1.axhline(test_acc, color='#e74c3c', linestyle='--', label=f'Test Acc: {test_acc*100:.1f}%')
    ax1.set_title("CNN Accuracy Convergence", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 2: Confusion Matrix ──
    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    for true, pred in zip(y_test, y_pred):
        cm[true, pred] += 1
        
    im = ax2.imshow(cm, cmap='Greens', interpolation='nearest')
    ax2.set_title("Audio Scene Confusion Matrix", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2)
    tick_marks = np.arange(len(CLASSES))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(CLASSES, rotation=20, ha='right', fontsize=9)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(CLASSES, fontsize=9)
    ax2.set_xlabel('Predicted Scene', fontweight='bold')
    ax2.set_ylabel('True Scene', fontweight='bold')
    
    # Annotate CM numbers
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm[i, j] > np.max(cm)/2 else "black"
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color=color, fontweight='bold')

    # ── Panel 3: Visual 2D MFCC spectrogram comparison across genres ──
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract one sample spectrogram from each class
    spec_siren   = X_test[y_test == 0][0][:, :, 0]
    spec_bark    = X_test[y_test == 1][0][:, :, 0]
    spec_drill   = X_test[y_test == 2][0][:, :, 0]
    spec_music   = X_test[y_test == 3][0][:, :, 0]
    
    # Stack them side by side with spacer columns
    spacer = np.zeros((time_steps, 8))
    full_visual = np.hstack([spec_siren, spacer, spec_bark, spacer, spec_drill, spacer, spec_music])
    
    im3 = ax3.imshow(full_visual.T, aspect='auto', cmap='magma', origin='lower')
    ax3.set_title("2D Spectrogram Textures (Siren | Bark | Drill | Music)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time Frame Step")
    ax3.set_ylabel("MFCC Filter Index")
    
    # Annotate separators
    w = spec_siren.shape[0]
    ax3.axvline(w, color='white', linestyle='--', alpha=0.8)
    ax3.axvline(2*w + 8, color='white', linestyle='--', alpha=0.8)
    ax3.axvline(3*w + 16, color='white', linestyle='--', alpha=0.8)

    # ── Panel 4: 2D CNN Architecture Schematic ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_title("CNN 2D Sound Event Processing Pipeline", fontsize=12, fontweight='bold')

    # Draw boxes for sequence workflow
    boxes = [
        (5.0, 9.2, "2D Spectrogram Input", f"[{time_steps} x {N_MFCC} x 1] Tensor", "#34495e"),
        (5.0, 7.2, "Conv2D (16 Filters, 3x3) + MaxPool", "Edge, Harmonic & Transient Detection", "#2980b9"),
        (5.0, 5.2, "Conv2D (32 Filters, 3x3) + MaxPool", "Deep Acoustic Feature Synthesis", "#8e44ad"),
        (5.0, 3.2, "Flatten & Fully Connected L2", "Feature Dense Projection", "#27ae60"),
        (5.0, 1.2, "Scene Classification Output", "Softmax outputs [Siren, Bark, Drill, Music]", "#d35400")
    ]
    for x, y_coord, title, desc, color in boxes:
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x - 3.8, y_coord - 0.65), 7.6, 1.3,
            boxstyle="round,pad=0.08", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2.0))
        ax4.text(x, y_coord + 0.15, title, ha='center', va='center', fontsize=10.0, color=color, fontweight='bold')
        ax4.text(x, y_coord - 0.35, desc, ha='center', va='center', fontsize=8.0, color='#444444')
        if y_coord > 2.0:
            ax4.annotate('', xy=(x, y_coord - 0.73), xytext=(x, y_coord - 1.48),
                         arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=2.0))

    fig.suptitle("Project 58: Audio Scene Classification (UrbanSound)\n"
                 f"Convolutional Neural Network (CNN)  |  Test Set Accuracy: {test_acc*100:.1f}%",
                 fontsize=14, fontweight='bold', color='#2c3e50')
                 
    output_filename = "audio_scene_results.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Evaluation dashboard saved as '{output_filename}'")
    print("====================================================")


if __name__ == "__main__":
    main()
