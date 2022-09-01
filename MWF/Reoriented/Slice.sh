#!/bin/bash
subjects=($(ls *.nii.gz))
for sub in ${subjects[@]}; do
	python3 slices.py $sub -f Slices/ -ax 0 &
done
