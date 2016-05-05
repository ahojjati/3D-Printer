python peakfind.py -> spits out threshold_peaks.csv, the row indexes of the peaks based on accelerometer
python createptaccel.py -> spits out peaktimesaccel.csv, the actual times in milliseconds of accelerometer epaks

1. 
python splicewavaccel.py -> slices the wav files based on the accelerometer peak times

2. 
python peakfilter.py -> uses stfft to filter out peaks based on difference, pulls in peaktimesaccel.csv and spits out threshold_peaks.csv
python splitwavfilter.py -> uses threshold_peaks.csv to splice based on filtered peak times