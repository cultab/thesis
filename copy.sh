#!/bin/sh


files=$(
cat <<- EOF
OVA.hpp
SMO.hpp
naive_serial.cu
GPUSVM.hpp
OVA.hpp
SMO.hpp
SVM_common.hpp
dataset.hpp
matrix.hpp
types.hpp
vector.hpp
Makefile
empty.d
cuda_helpers.h
EOF
)

for file in $files; do
	cp -rfv --update "./src/$file" ./server/src/
done

