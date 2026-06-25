# Project 58: Audio Scene Classification (UrbanSound8K)

This project implements a **Convolutional Neural Network (CNN)** to classify urban environmental sounds (Siren, Dog Bark, Drilling, Street Music) by converting audio tracks into 2D Mel-Frequency Cepstral Coefficient (MFCC) spectrogram maps.

---

## 1. Concept & Theory

### A. Audio Scene Classification
Environmental sound classification requires detecting short, highly variable sound events in noisy channels. Unlike speech or structured music, urban noise is characterized by its transients, pitch sweeps, and distinct noise envelopes.

### B. Urban Acoustic Signatures
1. **Siren**: Characterized by frequency modulation (pitch sweeps). In a spectrogram, this appears as smooth, oscillating **sinusoidal frequency bands** over time.
2. **Dog Bark**: A sudden, transient acoustic event. Spectrograms display **vertical broadband energy bursts** with a rapid exponential decay.
3. **Drilling**: A continuous, industrial mechanical sound. It consists of **thick, low-frequency harmonics** combined with periodic high-frequency grinding spikes (representing the drill bit impacts).
4. **Street Music**: Structured melodic steps (tonal keys) playing above a soft, low-energy ambient street hum.

### C. 2D CNN Spectrogram Extraction
Mel-Frequency Cepstral Coefficients (MFCCs) capture the spectral envelope. By keeping the frames sequential, we construct a 2D matrix of shape `(TimeSteps, N_MFCC, 1)` representing the sound texture. 
A 2D CNN slides Convolutional kernels over this grid, allowing the network to detect features like:
- Vertical transient edges (dog barks, drum hits).
- Wavy horizontal contours (sirens).
- Flat, continuous harmonic bars (street music, drilling).

---

## 2. Pipeline Overview
1. **Synthetic Urban Sound Synthesis**: Synthesizes 2-second audio clips for **4 sound classes**:
   - **Siren**: High-pitch frequency sweeps oscillating between 600 Hz and 1200 Hz.
   - **Dog Bark**: Rapid, transient-heavy noise bursts with short exponential decays.
   - **Drilling**: Constant mechanical low-pitch sawtooth buzz combined with periodic high-frequency grinding pulses.
   - **Street Music**: Pleasant, structured sine-wave melody arpeggios overlaying soft background street noise.
2. **2D MFCC Feature Extraction**: Converts raw audio waveforms into 2D spectrogram matrices of shape `(TimeSteps, N_MFCC, 1)`. Uses `librosa` if available, and falls back to a custom pure-Numpy implementation if not.
3. **CNN Classifier**:
   - `Input` shape: `(TimeSteps, N_MFCC, 1)`.
   - `Conv2D` layer (16 filters, $3 \times 3$, ReLU) + `MaxPooling2D` ($2 \times 2$).
   - `Conv2D` layer (32 filters, $3 \times 3$, ReLU) + `MaxPooling2D` ($2 \times 2$).
   - `Flatten` $\rightarrow$ `Dense` (64 units, dropout) $\rightarrow$ `Dense` (4 outputs, Softmax).
4. **Visual Dashboard**: Automatically saves training curves, a validation confusion matrix, visual comparison of 2D MFCC maps across all 4 classes, and the CNN pipeline flowchart to `audio_scene_results.png`.

---

## 3. How to Run
Run the pipeline to train the model and generate the evaluation dashboard:
```bash
python main.py
```
