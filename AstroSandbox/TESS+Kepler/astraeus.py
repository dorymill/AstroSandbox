# Machine Learning Preprocessor

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

# Grab and clean lightcurves
target = input('\nEnter host star ID | ')
print('\nFetching, cleaning, and stitching lightcurves. . .')
lcs = lk.search_lightcurve(target, author='Kepler', cadence='long').download_all(quality_bitmask='hard')
lc_clean = lcs.stitch(corrector_func=lambda x: x.flatten().normalize())

# Calculate BLS Fit Parameters
print('\nCalculating BLS Fit. . .')
# Models show high sensitivty to threshold values of about 10k/20 samples in period space
periodsamp = np.linspace(1,20,10000)
bls = lc_clean.to_periodogram(method='bls', period=periodsamp, frequency_factor=500)

Per = bls.period_at_max_power.value
t_0 = bls.transit_time_at_max_power.value
d_hrs = 23.93446959*bls.duration_at_max_power.value

# ML Preprocessing Phase

# Global View
period, t0, duration_hours = Per, t_0, d_hrs

temp_fold = lc_clean.fold(period, epoch_time=t0)
fractional_duration = (duration_hours / 24.0) / period
phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)

lc_fold = lc_flat.fold(period, epoch_time=t0)

lc_global = lc_fold.bin(time_bin_size=0.005).normalize() - 1
lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1

# Local View
phase_mask = (lc_fold.phase > -10*fractional_duration) & (lc_fold.phase < 10*fractional_duration)
lc_zoom = lc_fold[phase_mask]

lc_local = lc_zoom.bin(time_bin_size=0.0005).normalize() - 1
lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) ) * 2.0 + 1

# Fancy Plotting

fig, (axs) = plt.subplots(2,2)

axs[0,0].plot()
lc_clean.scatter(ax=axs[0,0], color='k')
axs[0,0].title.set_text('%s Lightcurve' %target)
axs[0,0].get_legend().remove()

axs[0,1].plot()
bls.plot(ax=axs[0,1], color='k')
axs[0,1].title.set_text('BLS Periodogram')
axs[0,1].get_legend().remove()

axs[1,0].plot()
axs[1,0].title.set_text('Global Phase Curve')
lc_global.scatter(ax=axs[1,0], color='k')
axs[1,0].get_legend().remove()

axs[1,1].plot()
axs[1,1].title.set_text('Local Phase Curve')
lc_local.normalize().scatter(ax=axs[1,1], color='k')
axs[1,1].get_legend().remove()

plt.tight_layout()
plt.savefig('Results\%s.png' %target)
plt.show()

gdf = lc_global.to_pandas()
ldf = lc_local.to_pandas()

#gdf.to_csv('Test\%s_global.csv' %targets[i])
#ldf.to_csv('Test\%s_local.csv'% targets[i])