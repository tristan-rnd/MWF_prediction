#!/bin/bash
subjects=($(ls *.nii))
for sub in ${subjects[@]}; do
	mv $sub ${sub::-77} &
done
