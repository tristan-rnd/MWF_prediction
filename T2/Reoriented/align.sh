#!/usr/bin/env bash
subjects=($(ls *.nii.gz))
for t2 in ${subjects[@]}; do
	echo "Flirt $t2 to ${t2::-9}T1.nii.g" &
	flirt -in $t2 -ref ~/PycharmProjects//MWF_prediction/T1/Reoriented/${t2::-9}T1.nii.gz -dof 6 -out Aligned/${t2} &
done
