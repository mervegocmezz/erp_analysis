# %%
# 
# ERP analysis: Go/No-Go
# 
#   NoGo-N2 (conflict detection)  -> fronto-central: FC1, FC2, Cz  ~200-300 ms
#   NoGo-P3 (response inhibition) -> fronto-central: FC1, FC2, Cz  ~300-500 ms
#   Beta ERD/ERS (motor)          -> C3, C4  (NOT a voltage
#                                     peak -- see step 8 below
# to catch the brain in the act of pressing the brakes: Beta ERD shows us the 
# motor system firing up to make a move on Go trials, while Beta ERS shows 
# the active motor suppression when hitting the brakes on No-Go trials)


# NOTE ON ELECTRODES: our easycap-M1 32-channel layout does not include an
# FCz channel -- FCz was very likely used as the online reference during
# recording, so it was never saved as a regular data channel. FC1/FC2 (which
# flank FCz) plus Cz are the nearest available approximation of the classic
# fronto-central N2/P3 site. Run `print(epochs.ch_names)` to check exact channel list.
 


import re
import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
plt.ion()

from mne import read_epochs, combine_evoked
from mne.viz import plot_compare_evokeds


def print_peak_measures(ch, tmin, tmax, lat, amp):
    print(f"  Channel:        {ch}")
    print(f"  Time window:    {tmin*1e3:.0f}-{tmax*1e3:.0f} ms")
    print(f"  Peak latency:   {lat*1e3:.1f} ms")
    print(f"  Peak amplitude: {amp*1e6:.2f} uV")


def safe_roi(inst, roi):
    """Keep only channels that actually exist in this data, and say so if
    any are missing -- avoids a crash when a montage doesn't have every
    channel you expect (e.g. FCz used as the recording reference)."""
    present = [ch for ch in roi if ch in inst.ch_names]
    missing = [ch for ch in roi if ch not in inst.ch_names]
    if missing:
        print(f"Note: {missing} not in this data, using {present} instead.")
    return present


# %%
# ---- 1. Load the epoched data -----------------------------------------
f_name = ('/Users/mervegocmez/Coding/data/thesis/derivatives/preprocessed/'
          'sub-NB002/ses-S001/eeg/'
          'sub-NB002_ses-S001_task-nb_acq-eeg_run-001_desc-go_no_go_epo.fif.gz')
epochs = read_epochs(f_name, preload=True)
print(epochs)
print(epochs.event_id)  # {'go': 11, 'nogo': 12}
print(epochs.ch_names)  # checking channel names 

# %%
# ---- 2. Average within each condition = the ERP ------------------------
ev_go = epochs['go'].average()
ev_nogo = epochs['nogo'].average()

# %%
# ---- 3. Look at the waveform + GFP ---------------------------------------
ev_nogo.plot(gfp=True, spatial_colors=True, titles='NoGo (all channels)')
ev_go.plot(gfp=True, spatial_colors=True, titles='Go (all channels)')

# %%
# ---- 4. Waveform + scalp topography ---------------------------------------

# I also added 0.17 s: 0.25 s was just a transition, while 0.17 s catches the actual 
# NoGo-N2 negative peak (~170 ms)—way better for showing conflict detection topography.

ev_nogo.plot_joint(times=[0.17, 0.25, 0.4], title='NoGo: waveform + topography')
ev_go.plot_joint(times=[0.17, 0.25, 0.4], title='Go: waveform + topography')

# %%
# ---- 5. NoGo-N2 -- fronto-central, ~200-300 ms -----------------------------
roi_n2p3 = safe_roi(ev_nogo, ['FC1', 'FC2', 'Cz'])
ch, lat, amp = ev_nogo.copy().pick(roi_n2p3).get_peak(
    tmin=0.18, tmax=0.32, mode='neg', return_amplitude=True)

print("** NoGo-N2 (fronto-central) **")
print_peak_measures(ch, 0.18, 0.32, lat, amp)

# %%
# ---- 6. NoGo-P3 -- fronto-central, ~300-500 ms -----------------------------
ch, lat, amp = ev_nogo.copy().pick(roi_n2p3).get_peak(
    tmin=0.3, tmax=0.5, mode='pos', return_amplitude=True)

print("** NoGo-P3 (fronto-central) **")
print_peak_measures(ch, 0.3, 0.5, lat, amp)

# %%
# ---- 7. Compare go vs nogo, and the difference wave -------------------------
plot_compare_evokeds(
    {'go': ev_go, 'nogo': ev_nogo}, picks=roi_n2p3, combine='mean',
    title='N2/P3 (FC1/FC2/Cz): go vs nogo')

diff = combine_evoked([ev_nogo, ev_go], weights=[1, -1])
diff.plot_joint(times=[0.17, 0.22, 0.3, 0.4], title='Go/No-Go: nogo minus go')

# %%
# ---- 8. Beta ERD/ERS at motor electrodes -------------------------------------
# Like frontal theta in N-Back, this is a POWER change (13-30 Hz), not a
# phase-locked voltage deflection, so it needs compute_tfr rather than
# averaging raw epochs. apply_baseline(..., mode='percent') expresses power
# at each time point as % change from the pre-stimulus baseline: a DROP is
# ERD (desynchronization, expected around response preparation), a rise
# above 0% afterwards is ERS (resynchronization/rebound).
#
# CAVEAT: this epoch only runs to 0.8 s post-stimulus. The ERD should be
# visible, but the ERS rebound often follows the actual button-press by
# several hundred ms and may be clipped by this window, so if the rebound
# looks cut off at the right edge, that's the epoch length, not necessarily
# a real absence of ERS.
roi_motor = safe_roi(epochs, ['C3', 'C4'])
beta_freqs = np.arange(13, 30, 1)

power_go = epochs['go'].compute_tfr(
    method='morlet', freqs=beta_freqs, n_cycles=beta_freqs / 2,
    picks=roi_motor, average=True, return_itc=False)

power_go.apply_baseline((-0.2, 0), mode='percent')

power_nogo = epochs['nogo'].compute_tfr(
    method='morlet', freqs=beta_freqs, n_cycles=beta_freqs / 2,
    picks=roi_motor, average=True, return_itc=False
)
power_nogo.apply_baseline((-0.2, 0), mode='percent')


# %% Beta ERD/ERS at motor electrodes .. RUN THIS ONE!

#  the new code, use this one rather than the previous one, because I wanted to keep the range the same):
vlim_range = (-0.6, 0.6)

power_go.plot(
    picks=roi_motor, 
    combine='mean', 
    title='Go: beta ERD/ERS (C3/C4)',
    vlim=vlim_range)


power_nogo.plot(
    picks=roi_motor, 
    combine='mean', 
    title='NoGo: beta ERD/ERS (C3/C4)',
    vlim=vlim_range)


# %%

# ---- 9. Save every figure from this script ----------------------------------
# Grabs every matplotlib figure still open at this point (all the plots
# above) and writes each to its own PNG, task-named subfolder, one shared
# output root so every task's figures live side by side but don't collide.
OUT_ROOT = Path('/Users/mervegocmez/Coding/data/thesis/derivatives/figures')
TASK_NAME = 'go_no_go'
out_dir = OUT_ROOT / TASK_NAME
out_dir.mkdir(parents=True, exist_ok=True)


def _safe_filename(text):
    """Sanitize title strings into safe filenames."""
    name = re.sub(r'[^\w\-]+', '_', text.strip())
    return name.strip('_') or 'figure'


for num in plt.get_fignums():
    fig = plt.figure(num)
    fig.canvas.draw()  # Forces full rendering before saving
    
    # Retrieve title safely
    if fig._suptitle is not None and fig._suptitle.get_text():
        title = fig._suptitle.get_text()
    elif fig.axes and fig.axes[0].get_title():
        title = fig.axes[0].get_title()
    else:
        title = f'figure_{num}'
        
    fname = f'{num:02d}_{_safe_filename(title)}.png'
    path = out_dir / fname
    
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close('all')
print("Execution complete. All figures exported successfully.")
# %%
