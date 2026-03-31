~BuildCored Day 6🐬

 It capturew mic input, detects your breathing rhythm from the audio amplitude, 
 visualizes each breath as a live waveform, and compute breaths-per-minute with a rolling average.

 
Tech stack: pyaudio, scipy.signal (Butterworth filter), matplotlib animation
Hardware concept: Microphone as analog sensor. 
Signal windowing and envelope detection — foundational DSP concepts used in every audio IC, medical sensor, and IoT microphone.

Tip: Breathe slowly and deliberately close to your mic. Normal room noise won't trigger it, and that's the whole point of the filter😊
