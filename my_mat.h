/*****************************
 * Header of my math library *
 * By Jeng-Hau Lin           *
 *****************************/

#ifndef __MY_MAT__
#define __MY_MAT__

#include <math.h>
#include <stdlib.h>
#include <stdio.h> 
#include <fcntl.h>// for file cotrol to access dataset
#include <omp.h>  // for OpenMP multi-threading of loops
#include <time.h> // for random seed

#define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

float act_pct(float x);
float dact_pct(float x);
float sigmoid(float x);
float dsigmoid(float x);
float softmax(float x, float result[], const int n_c);
float crossEntropy(float target[], float result[], const int n_c);

float bnz(float x);
float sbnz(float x);

int checkMax(float * output, int nc, int ans_byte);

int rndArr(float *mtx, const size_t n_c);
int rndMtx(float *mtx, const size_t n_r ,const size_t n_c);

int shufArr(int *arr, const size_t nc);

int prtArr(float arr[], const size_t n);
int prtMtx(float *mtx, const size_t n_r, const size_t n_c);

#endif

