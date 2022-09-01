#!/bin/bash
subjects=($(ls *.nii.gz))
for sub in ${subjects[@]}; do
	fslreorient2std $sub "Reoriented/$sub" &
done
