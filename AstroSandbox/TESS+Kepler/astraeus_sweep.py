# Machine Learning Prepprocessor

from ipywidgets.widgets.widget_string import Password
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
#warnings.filterwarnings("error")
import os

# Grab and clean lightcurves
inlist = input('\nEnter inlist | ')
data = pd.read_csv(inlist)

targets = data['kepid']
title = data['kepler_name']
Peri = data['koi_period']
peak = data['koi_time0bk']
dur = data['koi_duration']

for i in range(0,len(targets)):

    os.system('cls' if os.name == 'nt' else 'clear')
    print('\nFetching, cleaning, and stitching lightcurves for %s. . .'%targets[i])
   
    try:
        lcs = lk.search_lightcurve(targets[i], author='Kepler', cadence='long').download_all(quality_bitmask='hard')
        lc_clean= lcs.stitch(corrector_func=lambda x: x.remove_outliers(sigma=10, sigma_upper=1).flatten().normalize())

        # Calculate BLS Fit Parameters
        print('\nCalculating BLS Fit. . .')
        # Models show high sensitivty to threshold values of about 10k/20 samples in period space
        periodsamp = np.linspace(1,20,10000)
        bls = lc_clean.to_periodogram(method='bls', period=periodsamp, frequency_factor=500)

        Per = Peri[i]
        t_0 = peak[i]
        d_hrs = dur[i]

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
        axs[0,0].title.set_text('%s Lightcurve' %title[i])
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
        plt.savefig('Results\%s.png' %title[i])
        plt.clf()   
        #plt.show()
    except:
        print('\nData grab failed. Trying next candidate. . .')
        pass