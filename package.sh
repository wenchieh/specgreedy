#!/usr/bin/env bash

PROJECT_VERSION=specgreedy-1.0

rm ${PROJECT_VERSION}.tar.gz
rm -rf ${PROJECT_VERSION}
mkdir ${PROJECT_VERSION}
cp -R ./{run.sh,demo.sh,package.sh, run_*.py,src,./src,./outs,./data,Makefile} ./${PROJECT_VERSION}
tar cvzf ${PROJECT_VERSION}.tar.gz --exclude='._*' ./${PROJECT_VERSION}
rm -rf ${PROJECT_VERSION}
echo done.