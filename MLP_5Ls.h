/************************************
 * Header of Multi-layer Perceptron *
 * save weights in memory        *
 * 5 layers:                        *
 *          1 input layer           *
 *          2 hidden layer          *
 *          3 hidden layer          *
 *          4 hidden layer          *
 *          5 output layer          *
 * By Jeng-Hau Lin                  *
 ************************************/

#ifndef __MLP_5Ls__
#define __MLP_5Ls__

#include "my_mat.h"

#define BN 0 // Set to turn on Batch Normalization, reset to turn off Batch Normalization
#define BNZ 1 // Set to turn on binarize, reset to turn off binarize

#define BATCH_SIZE 10 // size of mini-batch
#define N_SAMPLES 60 // The number of samples

#define N_1 28*28 // The number of nodes in the 1st layer
#define N_2 1000 // The number of nodes in the hidden layer
#define N_3 100 // The number of nodes in the hidden layer
#define N_4 10 // The number of nodes in the hidden layer
#define N_5 10 // The number of nodes in the 5th layer
#define N_LAYERS 4 // the number of layers

//#define ALPHA 0.0099
#define ALPHA 0.01
#define GAMMA 1.0 // 6 is the magic number for gamma
#define BETA 0.0
#define EPSILON 0.0000000001
//#define EPSILON 0.001

#define B1 0.875
#define B2 0.9990234375

# define LAMBDA 0.9999

//#define g(x) act_pct(x)
//#define dg(x) dact_pct(x)
#define g(x)  sigmoid(x)
#define dg(x)  dsigmoid(x)

typedef struct NeuNet{
    /* arrange the weights in the following format:
      wt_1,1_2,1  wt_1,2_2,1  wt_1,3_2,1  wt_1,4,2,1     wt_1,1_2,2  wt_1,2_2,2  wt_1,3_2,2  wt_1,4_2,2 ... 
     */
    float * pW12; // the weights between layer 1 and 2, size [N_2*N_1]
    float * pW23; // the weights between layer 2 and 3, size [N_3*N_2]
    float * pW34; // the weights between layer 2 and 3, size [N_3*N_4]

    float * pI2; // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_2]
    float * pI3; // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_3]
    float * pI4; // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_4]
    float * pA1; // the activations of the 1nd nodes, size [BATCH_SIZE*N_1]
    float * pA2; // the activations of the 2nd nodes, size [batch_size*n_2]
    float * pA3; // the activations of the 3nd nodes, size [BATCH_SIZE*N_3]
    float * pA4; // the activations of the 3nd nodes, size [BATCH_SIZE*N_4]

    float * pGB2; // the array of gamma and beta [gamma ... beta ...], size [2*N_2]
    float * pGB3; // the array of gamma and beta [gamma ... beta ...], size [2*N_3]

    float * pxh2; // the normalized wighted sums of the 2nd nodes, size [BATCH_SIZE*N_2]
    float * pxh3; // the normalized wighted sums of the 2nd nodes, size [BATCH_SIZE*N_3]
    float * pBa2; // the scaled and shifted wighted sums of the 2nd nodes, size [BATCH_SIZE*N_2]
    float * pBa3; // the scaled and shifted wighted sums of the 2nd nodes, size [BATCH_SIZE*N_3]

    float * pU2; // average of in2 across a mini-batch, size [N_2]
    float * pU3; // average of in3 across a mini-batch, size [N_3]
    float * pDs2; // standard deviation of in2 across a mini-batch, size [N_2]
    float * pDs3; // standard deviation of in3 across a mini-batch, size [N_3]

    float * pGyki12; // d(l)/dyk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
    float * pGyki23; // d(l)/dyk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]
    float * pGxhki12; // d(l)/dxhk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
    float * pGxhki23; // d(l)/dxhk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]
    float * pGxki12; // d(l)/dxk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
    float * pGxki23; // d(l)/dxk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]

    float * pGu12; // d(l)/du of layers, size [N_2]
    float * pGu23; // d(l)/du of layers, size [N_3]
    float * pGds12; // d(l)/dds of layers, size [N_2]
    float * pGds23; // d(l)/dds of layers, size [N_3]

    float * pM12; // the momentum of w12, size [N_2*N_1]
    float * pM23; // the momentum of w12, size [N_3*N_2]
    float * pM34; // the momentum of w12, size [N_4*N_3]
    float * pV12; // the inertia of w12, size [N_2*N_1]
    float * pV23; // the inertia of w12, size [N_3*N_2]
    float * pV34; // the inertia of w12, size [N_3*N_4]

    float * pGWk12; // d(l)/dSk of layer 2 across a mini-batch, size [N_2*N_1]
    float * pGWk23; // d(l)/dSk of layer 3 across a mini-batch, size [N_3*N_2]
    float * pGWk34; // d(l)/dSk of layer 3 across a mini-batch, size [N_4*N_3]

    float * pMgb2; // moment of gamma and beta on layer 2, size [2*N_2]
    float * pMgb3; // moment of gamma and beta on layer 3, size [2*N_3]

    float * pVgb2; // moment of gamma and beta on layer 2, size [2*N_2]
    float * pVgb3; // moment of gamma and beta on layer 3, size [2*N_3]

    float * pGGB12; //averaged d(l)/dgamma and d(l)/dbeta of layer across a mini-batc2, size [2*N_2]
    float * pGGB23; //averaged d(l)/dgamma and d(l)/dbeta of layer3 across a mini-batc, size [2*N_3]

    int nLs; // the number of layers
    int ns[N_LAYERS]; // the number of nodes in the 1st layer
} NNet;

int forwardIn(u_char * input, NNet *pNet, int b);
int forward(float * pAi, float * pWj, float * pIj, float * pAj, int ni, int nj, int b);
int forwardBnz(float * pAi, float * pWj, float * pIj, float * pAj, int ni, int nj, int b);

int backwardOut(const int bytes[],float * target, NNet *pNet, float * pAj, float * pIj, float * pAi, float * pGWij, float * pGsj);
int backwardGyb(int nj, int nk, float * pGybij, float * pWjk, float * pGsk);
int backwardBN(int ni, int nj, float * pGybij, float * pGBj, float * pGxhbij, float * pGdsij, float * pAj,
               float * pUj, float * pDsj, float * pGuij, float * pGxbij, float * pGGBij, float * pxhj);
int backwardGW(int ni, int nj, float * pGsj, float * pIj, float * pGxbij, float * pAi, float * pGWij);
int backward(const int bytes[], NNet *pNet);

int accumulate(NNet *pNet, float alpha, int t);

int batchNml(NNet * pNet, int n);

float cost1(const float target[], float output[]);
float cost2(const float target[], float output[]);

int saveWeights(const char str[], NNet *pNet);
int readWeights(const char str[], NNet *pNet);

int adamax(float * m, float * vt, float * theta, float ita, float * gt, int t); 

#endif
