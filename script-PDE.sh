#!/bin/sh
# Runs the PDE experiment multiple times with the same initialization and collects the results in the folder "results"

optimizer="BBI"
seed="42"
epochs="2000"
threshold0="200"
threshold="2000"
nFixedBounces="10"
lr="8e-6"
deltaEn="0.0"
feature="1"

for run in {1..18}
do
	name="$optimizer-seed$seed-lr$lr-threshold$threshold-threshold0$threshold0-deltaEn$deltaEn-nFixedBounces$nFixedBounces-feature$feature-run$run"
	echo $name
	python3 main.py PDE_PoissonD --optimizer $optimizer --lr $lr --seed $seed --epochs $epochs --feature $feature --threshold $threshold --threshold0 $threshold0 --deltaEn $deltaEn --nFixedBounces $nFixedBounces -n $name

done
