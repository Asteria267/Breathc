"""
██████╗ ██████╗ ███████╗ █████╗ ████████╗██╗  ██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██║  ██║
██████╔╝██████╔╝█████╗  ███████║   ██║   ███████║
██╔══██╗██╔══██╗██╔══╝  ██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║███████╗██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝  ╚═╝   ╚═╝  ╚═╝

Day 6 - Build Core'd Orcas
Breathe near the mic. Watch your rhythm. See your BPM live.
"""

import pyaudio
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

SAMPLE_RATE     = 44100   # samples per second (standard audio)
CHUNK           = 1024    # how many samples per read
BUFFER_SECONDS  = 10      # seconds of audio to show on screen
FILTER_LOW      = 0.1     # Hz — low cutoff for butterworth
FILTER_HIGH     = 0.5     # Hz — high cutoff for butterworth
BREATH_THRESHOLD = 0.02   # amplitude above this = breath detected
MIN_BREATH_GAP  = 1.5     # minimum seconds between breaths (debounce)
ROLLING_WINDOW  = 60      # seconds to average BPM over

# ─────────────────────────────────────────────
#  BUTTERWORTH FILTER SETUP
# ─────────────────────────────────────────────

def make_filter(low, high, fs, order=3):
    """
    Butterworth bandpass filter.
    Keeps only frequencies between low and high Hz.
    Breathing = 0.1 to 0.5 Hz → everything else = noise → removed.
    """
    nyq    = fs / 2.0          # nyquist frequency = half sample rate
    low_n  = low / nyq         # normalize to 0-1
    high_n = high / nyq
    b, a   = butter(order, [low_n, high_n], btype='band')
    return b, a

b, a = make_filter(FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)

# ─────────────────────────────────────────────
#  AUDIO SETUP
# ─────────────────────────────────────────────

pa     = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,           # mono mic
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────

buffer_size    = SAMPLE_RATE * BUFFER_SECONDS
raw_buffer     = deque([0.0] * buffer_size, maxlen=buffer_size)
envelope_buffer = deque([0.0] * buffer_size, maxlen=buffer_size)

breath_times   = deque()          # timestamps of detected breaths
last_breath    = 0.0              # time of last breath (debounce)
bpm            = 0.0              # current BPM
breath_count   = 0                # total breaths detected
in_breath      = False            # currently in a breath spike?

# ─────────────────────────────────────────────
#  MATPLOTLIB SETUP
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
fig.patch.set_facecolor('#0d0d0f')

for ax in [ax1, ax2]:
    ax.set_facecolor('#0d0d0f')
    ax.tick_params(colors='#666')
    ax.spines['bottom'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['right'].set_color('#333')

# raw waveform plot
line_raw,  = ax1.plot([], [], color='#378ADD', linewidth=0.8, alpha=0.7)
ax1.set_xlim(0, buffer_size)
ax1.set_ylim(-0.3, 0.3)
ax1.set_title('raw mic signal', color='#888', fontsize=11)
ax1.set_ylabel('amplitude', color='#888', fontsize=9)

# envelope plot (filtered, smoothed)
line_env,  = ax2.plot([], [], color='#00ff78', linewidth=1.5)
threshold_line = ax2.axhline(y=BREATH_THRESHOLD, color='#ffb400',
                              linewidth=1, linestyle='--', alpha=0.7)
ax2.set_xlim(0, buffer_size)
ax2.set_ylim(-0.01, 0.15)
ax2.set_title('breathing envelope (filtered)', color='#888', fontsize=11)
ax2.set_ylabel('amplitude', color='#888', fontsize=9)

# BPM text
bpm_text    = ax2.text(0.02, 0.85, 'BPM: --', transform=ax2.transAxes,
                        color='#ffb400', fontsize=18, fontweight='bold')
breath_text = ax2.text(0.02, 0.70, 'breaths: 0', transform=ax2.transAxes,
                        color='#888', fontsize=11)
status_text = ax2.text(0.75, 0.85, 'breathe slowly...', transform=ax2.transAxes,
                        color='#666', fontsize=11)

plt.tight_layout(pad=2.0)

# ─────────────────────────────────────────────
#  ANIMATION UPDATE
# ─────────────────────────────────────────────

def update(frame):
    global bpm, last_breath, breath_count, in_breath

    # ── read audio chunk from mic ──
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)
    except Exception:
        return line_raw, line_env

    # add to raw buffer
    raw_buffer.extend(samples)

    # ── compute envelope (amplitude over time) ──
    # take absolute value → smooth with rolling mean
    abs_samples  = np.abs(samples)
    envelope_val = float(np.mean(abs_samples))
    envelope_buffer.extend([envelope_val] * len(samples))

    # ── filter the envelope buffer for breath detection ──
    env_array = np.array(envelope_buffer)

    # only filter if we have enough data
    if len(env_array) > 100:
        try:
            filtered = filtfilt(b, a, env_array)
            filtered = np.abs(filtered)
        except Exception:
            filtered = env_array
    else:
        filtered = env_array

    # ── BREATH DETECTION ──
    now         = time.time()
    current_val = float(np.mean(filtered[-CHUNK:]))

    if current_val > BREATH_THRESHOLD and not in_breath:
        # spike started
        if now - last_breath > MIN_BREATH_GAP:
            in_breath    = True
            last_breath  = now
            breath_count += 1
            breath_times.append(now)

            # remove old breath times outside rolling window
            while breath_times and now - breath_times[0] > ROLLING_WINDOW:
                breath_times.popleft()

            # BPM = breaths in window / window minutes
            if len(breath_times) >= 2:
                window = min(now - breath_times[0], ROLLING_WINDOW)
                bpm    = len(breath_times) / (window / 60.0)
            else:
                bpm = 0.0

    elif current_val < BREATH_THRESHOLD * 0.7:
        in_breath = False

    # ── UPDATE PLOTS ──
    x = np.arange(buffer_size)

    line_raw.set_data(x, list(raw_buffer))
    line_env.set_data(x, list(envelope_buffer))

    # update text
    if bpm > 0:
        bpm_text.set_text(f'BPM: {bpm:.1f}')
        bpm_text.set_color('#00ff78' if 10 <= bpm <= 25 else '#ff3c3c')
    else:
        bpm_text.set_text('BPM: --')

    breath_text.set_text(f'breaths: {breath_count}')

    if in_breath:
        status_text.set_text('BREATH DETECTED')
        status_text.set_color('#00ff78')
    else:
        status_text.set_text('listening...')
        status_text.set_color('#666')

    return line_raw, line_env, bpm_text, breath_text, status_text


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

print("\n BreathClock - Day 6")
print("   Breathe slowly near your mic.")
print("   Close the plot window to quit.\n")

ani = animation.FuncAnimation(
    fig, update,
    interval=50,        # update every 50ms
    blit=True,
    cache_frame_data=False
)

try:
    plt.show()
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\n BreathClock closed.")
