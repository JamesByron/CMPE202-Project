#!/bin/bash
echo Usage: get.sh <filename.txt>
echo This script collects LLC results from the four types of simulatioss that were run.
echo 
cat lru/$1 | grep "LLC" | grep "TOTAL"
cat random/$1 | grep "LLC" | grep "TOTAL"
cat bzt-h/$1 | grep "LLC" | grep "TOTAL"
cat allt-h/$1 | grep "LLC" | grep "TOTAL"
cat bzt-hm/$1 | grep "LLC" | grep "TOTAL"
cat allt-hm/$1 | grep "LLC" | grep "TOTAL"
echo 
cat lru/$1 | grep "LLC" | grep "LOAD"
cat random/$1 | grep "LLC" | grep "LOAD"
cat bzt-h/$1 | grep "LLC" | grep "LOAD"
cat allt-h/$1 | grep "LLC" | grep "LOAD"
cat bzt-hm/$1 | grep "LLC" | grep "LOAD"
cat allt-hm/$1 | grep "LLC" | grep "LOAD"
echo 
cat lru/$1 | grep "LLC" | grep "RFO"
cat random/$1 | grep "LLC" | grep "RFO"
cat bzt-h/$1 | grep "LLC" | grep "RFO"
cat allt-h/$1 | grep "LLC" | grep "RFO"
cat bzt-hm/$1 | grep "LLC" | grep "RFO"
cat allt-hm/$1 | grep "LLC" | grep "RFO"
echo 
cat lru/$1 | grep "LLC" | grep "WRITEBACK"
cat random/$1 | grep "LLC" | grep "WRITEBACK"
cat bzt-h/$1 | grep "LLC" | grep "WRITEBACK"
cat allt-h/$1 | grep "LLC" | grep "WRITEBACK"
cat bzt-hm/$1 | grep "LLC" | grep "WRITEBACK"
cat allt-hm/$1 | grep "LLC" | grep "WRITEBACK"