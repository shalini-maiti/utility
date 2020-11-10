#!/bin/bash
 
## define an array ##
model=( Dell HP Oracle )
checkpoint = ()
test_datatype = (STB, MHP, Freihand)
 
## get item count using ${arrayname[@]} ##
for m in "${model[@]}"
do
  echo "${m}"
  # do something on $m #
done