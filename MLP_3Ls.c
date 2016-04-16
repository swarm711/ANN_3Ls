/************************************
 * Source of Multi-layer Perceptron *
 * save weightings in memory        *
 * 5 layers:                        *
 *          1 input layer           *
 *          2 hidden layer          *
 *          3 output layer          *
 * By Jeng-Hau Lin                  *
 ************************************/

#include "MLP_3Ls.h"

//#define PERC // Uncomment this if perceptron is wanted

/******************************************************
 * Plain implementation of a 3-layer neural net       *
 * forward propagation                                *
 * Input:                                             *
 *       input[]: the training/test image             *
 *       *pnNet: pointer to the neural network struct *
 * Output:                                            *
 *       output[]: the activations of the last layer  *
 ******************************************************/
int forward(u_char input[N_1], NNet *pnNet, float output[]){
    float (*g)(float); // Use function pointer to choose differet activation functions
#ifdef PERC
    g = &act_pct;
#else
    g = &sigmoid;
#endif
    int i,j,k; // i for 1st, j for 2nd, and k for 3rd
#pragma omp parallel for private(i)
    for(i=0;i<pnNet->ns[0];++i){
        pnNet->act1[i] = ((float)input[i]-128.0)/255.0;
    }

#pragma omp parallel for private(j,i)
    for(j=0;j<pnNet->ns[1];++j){
        pnNet->in2[j] = 0.0;
        for(i=0;i<pnNet->ns[0];++i){
            /* the input of a neuron is the sum of weighted activactions from higher layer */
            pnNet->in2[j] += pnNet->act1[i] * pnNet->wt12[j*N_1+i];
        }
        /* activation function can bring the nonlinearity */
        pnNet->act2[j] = g(pnNet->in2[j]);
    }

#pragma omp parallel for private(k,j)
    for(k=0;k<pnNet->ns[2];++k){
        pnNet->in3[k] = 0;
        for(j=0;j<pnNet->ns[1];++j){
            pnNet->in3[k] += pnNet->act2[j] * pnNet->wt23[k*N_2+j];
        }
        pnNet->act3[k] = g(pnNet->in3[k]);
        output[k] = pnNet->act3[k];
    }

    return 0;
}

/******************************************************
 * Plain implementation of a 3-layer neural net       *
 * backward propagation                               *
 * Input:                                             *
 *       target[]: for calculation of error           *
 *       *pnNet: pointer to the neural network struct *
 *       **ppDw: retain the delta for minibatch       *
 *       *cnt: the counter toward BATCH-SIZE          *
 * Output:                                            *
 *       *pnNet: weights could be updated             *
 *       **ppDw: delta of weight could be reseted     *
 *       *cnt: the counter could be reseted           *
 ******************************************************/
int backward(const float target[], float alpha, NNet *pnNet, float ** ppDw, int *cnt){
    float (*dg)(float);
#ifdef PERC
    dg = &dact_pct;
#else
    dg = &dsigmoid;
#endif

    int k, j, i; // k for 3rd, j for 2nd, and i for 1st

    /* Start from the lowest layer */
    float Err3[N_3];
    float Delta3[N_3];
#pragma omp parallel for private(k,j)
    for(k=0;k<pnNet->ns[2];++k){
        Err3[k] = target[k] - pnNet->act3[k];
        Delta3[k] = dg(pnNet->in3[k]) * Err3[k];

        for(j=0;j<pnNet->ns[1];++j){
            *(ppDw[1] +k*N_2 + j) += alpha * pnNet->act2[j] * Delta3[k]; // Delta weight for the weights between layer 3 and 2
            /* update the weights if cnt reachs BATCH_SIZE */
            if(*cnt >= BATCH_SIZE){
                pnNet->wt23[k*N_2+j] += *(ppDw[1] +k*N_2 + j);
                *(ppDw[1] +k*N_2 + j) = 0.0;
            }
        }
    }

    float sum3[N_2]; // Collect the error from its lower layer
    float Delta2[N_2];
#pragma omp parallel for private(j,k,i)
    for(j=0;j<pnNet->ns[1];++j){
        /* sum columnwisely over the lower neurons connected to the current neuron to collect the errors */
        /* this came from the chain rule of derivative */
        sum3[j] = 0;
        for(k=0;k<pnNet->ns[2];++k)
            sum3[j] += pnNet->wt23[k*N_2+j] * Delta3[k];

        Delta2[j] = dg(pnNet->in2[j]) * sum3[j];

        for(i=0;i<pnNet->ns[0];++i){
            *(ppDw[0] + j*N_1 + i) += alpha * pnNet->act1[i] * Delta2[j]; // Delta weight for the weights between layer 2 and 1
            /* update the weights if cnt reachs BATCH_SIZE */
            if(*cnt >= BATCH_SIZE){
                pnNet->wt12[j*N_1+i] += *(ppDw[0] + j*N_1 + i);
                *(ppDw[0] + j*N_1 + i) = 0.0;
            }
        }
    }

    if(*cnt >= BATCH_SIZE){
        *cnt = 0;
    }

    return 0;
}

/* square error */
float cost1(const float target[], float output[]){

    float delta = 0.0;
    float cost1 = 0.0;
    int i;
    for(i=0;i!=N_3;++i){
        delta = target[i] - output[i];
        cost1 += 0.5*delta*delta;
    }

    return cost1;
}

int saveWeights(const char str[], NNet *pnNet){
    FILE *fp = fopen(str, "wb");

    fwrite(pnNet->ns,sizeof(int), pnNet->nLs, fp);
    fwrite(pnNet->wt12,sizeof(float), (pnNet->ns[0])*(pnNet->ns[1]), fp);
    fwrite(pnNet->wt23,sizeof(float), (pnNet->ns[1])*(pnNet->ns[2]), fp);

    fclose(fp);
    return 0;
}

int readWeights(const char str[], NNet *pnNet){
    FILE *fp = fopen(str, "rb");

    int c;
    c = fread(pnNet->ns,sizeof(int), pnNet->nLs, fp);
    c = fread(pnNet->wt12,sizeof(float), (pnNet->ns[0])*(pnNet->ns[1]), fp);
    c = fread(pnNet->wt23,sizeof(float), (pnNet->ns[1])*(pnNet->ns[2]), fp);

    fclose(fp);
    return 0;
}
