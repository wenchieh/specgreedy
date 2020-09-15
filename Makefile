#
# Makefile for SPECGREEDY
#

all: clean demo

clean:
	rm -rf outs
	mkdir outs
	find ./src -name *.pyc | xargs rm -rf

demo:
	@echo "run demo."
	./demo.sh
    @echo
	
tar:
	rm -rf output/*
	@echo [PROJECT]
	./package.sh
	@echo [PROJECT] "done."
	@echo