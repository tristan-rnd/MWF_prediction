#!/bin/bash
subjects=($(ls *_2))
for sub in ${subjects[@]}; do
	mv $sub ${sub::-31} &
done
