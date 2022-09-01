#!/bin/bash
subjects=($(ls *.nii))
for sub in ${subjects[@]}; do
	fslreorient2std $sub "Reoriented/$sub" &
done
