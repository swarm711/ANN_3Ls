/************************************
 * Header of Multi-layer Perceptron *
 * save weightings in memory        *
 * 3 layers:                        *
 *          1 input layer           *
 *          2 hidden layer          *
 *          3 output layer          *
 * By Jeng-Hau Lin                  *
 ************************************/

#ifndef __MLP_3Ls__
#define __MLP_3Ls__

#include "my_mat.h"

#define BATCH_SIZE 10 // size of mini-batch
#define N_SAMPLES 100 // The number of samples

#define N_1 28*28 // The number of nodes in the 1st layer
#define N_2 1000 // The number of nodes in the hidden layer
#define N_3 10 // The number of nodes in the hidden layer
#define N_LAYERS 3 // the number of layers


typedef struct NeuNet{
    /* arrange the weightings in the following format:
      wt_1,1_2,1  wt_1,2_2,1  wt_1,3_2,1  wt_1,4,2,1      wt_1,1_2,2  wt_1,2_2,2  wt_1,3_2,2  wt_1,4_2,2 ... 
     */
    float * wt12; // the weightings between layer 1 and 2
    float * wt23; // the weightings between layer 2 and 3
   
    float in2[N_2]; // the wighted sums of the 2nd nodes
    float in3[N_3]; // the wighted sums of the 3rd nodes

    float act1[N_1]; // the activations of the 1nd nodes
    float act2[N_2]; // the activations of the 2nd nodes
    float act3[N_3]; // the activations of the 3nd nodes

    int nLs; // the number of layers
    int ns[N_LAYERS]; // the number of nodes in the 1st layer
} NNet;

float (*g)(float); // Use function pointer to choose differet activation functions
float (*dg)(float);
int forward(u_char input[N_1], NNet *pnNet, float output[]);

float cost1(const float target[], float output[]);
int backward(const float target[], float alpha, NNet *pnNet, float * ppDw[], int *cnt);

int saveWeights(const char str[], NNet *pnNet);
int readWeights(const char str[], NNet *pnNet);

#endif

