all: build

build: semblance.c su.c
	gcc semblance.c su.c -o build/reg -lm -fopenmp -std=gnu99
