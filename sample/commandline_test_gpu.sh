#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
testdatafile=$inputdir"drugcell_test.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

modelfile="../pretrained_model/drugcell_v1.pt"

resultdir="Result_sample"
hiddendir="Hidden_sample"

cudaid=$1

if [$cudaid = ""]; then
	cudaid=0
fi

mkdir $resultdir
mkdir $hiddendir

source activate pytorch3drugcell

python -u ../code/predict_drugcell.py -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype $mutationfile -fingerprint $drugfile -hidden $hiddendir -result $resultdir -predict $inputdir/drugcell_test.txt -load $modelfile -cuda $cudaid > test_sample.log
