#!/bin/bash
FILEDIR="../inputs/"
FNAMES=("easy/webpbn000001.mk easy/webpbn000003.mk intermediate/webpbn000022.mk intermediate/webpbn000045.mk hard/webpbn002438.mk hard/webpbn000428.mk bench/webpbn001739.mk bench/webpbn001837.mk bench/webpbn003541.mk bench/webpbn007604.mk" )
for fname in $FNAMES; do
  echo $FILEDIR$fname
  $1 -f $FILEDIR$fname
done
