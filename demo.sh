#!/usr/bin/env bash


mkdir outs
echo [DEMO] 'running monopaitite graph'
python run_specgreedy_mon.py ./data/ca-HepPh_SNAP.edgelist ./outs/out.res ' ' False 10
echo [DEMO] 'done!'
echo


echo [DEMO] 'running bipaitite graph'
python run_specgreedy_bip.py ./data/amazon_user_art_rate_time.edgelist ./outs/out.res_bip ',' False even 10 1.0
echo [DEMO] 'done!'
echo

