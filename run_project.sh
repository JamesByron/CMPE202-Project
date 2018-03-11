#!/bin/bash
echo 1 / 5 : bzip2_10M
./run_champsim.sh bimodal-no-no-cramer-1core 1 10 bzip2_10M
echo 2 / 5 : astar_313B
./run_champsim.sh bimodal-no-no-cramer-1core 1 10 astar_313B
echo 3 / 5 : bwaves_1609B
./run_champsim.sh bimodal-no-no-cramer-1core 1 10 bwaves_1609B
echo 4 / 5 : graph_analytics_10M
./run_champsim.sh bimodal-no-no-cramer-1core 1 10 graph_analytics_10M
echo 5 / 5 : h264ref_351B
./run_champsim.sh bimodal-no-no-cramer-1core 1 10 h264ref_351B
echo Done