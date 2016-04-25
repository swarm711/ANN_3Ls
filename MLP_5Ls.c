/************************************
 * Source of Multi-layer Perceptron *
 * save weightings in memory        *
 * 5 layers:                        *
 *          1 input layer           *
 *          2 hidden layer          *
 *          3 hidden layer          *
 *          4 hidden layer          *
 *          5 output layer          *
 * By Jeng-Hau Lin                  *
 ************************************/

#include "MLP_5Ls.h"
#include <assert.h>

int forwardIn(u_char * input, NNet *pNet, int b){
    int i; // i for 1st and j for 2nd
#pragma omp parallel for private(i)
    for(i=0;i<N_1;++i){
        pNet->pA1[b*N_1+i] = ((float)input[i]-128.0)/255.0;
    }
    return 0;
}

int forward(float * pAi, float * pWij, float * pIj, float * pAj, int ni, int nj, int b){
    int i,j; // i for 1st and j for 2nd
#pragma omp parallel for private(j,i)
    for(j=0;j<nj;++j){
        pIj[b*nj+j] = 0.0;
        for(i=0;i<ni;++i){
            pIj[b*nj+j] += pAi[b*ni+i] * pWij[j*ni+i];
        }
        pAj[b*nj+j] = g(pIj[b*nj+j]);
    }
    return 0;
}

int forwardBnz(float * pAi, float * pWij, float * pIj, float * pAj, int ni, int nj, int b){
    //printf("fBnz\n");
    int i,j; // i for 1st and j for 2nd
#pragma omp parallel for private(j,i)
    for(j=0;j<nj;++j){
        pIj[b*nj+j] = 0.0;
        for(i=0;i<ni;++i){
            pAi[b*ni+i] = bnz(pAi[b*ni+i]);
            pWij[j*ni+i] = bnz(pWij[j*ni+i]);
            pIj[b*nj+j] += pAi[b*ni+i] * pWij[j*ni+i];
            //pIj[b*nj+j] += bnz(pAi[b*ni+i]) * bnz(pWij[j*ni+i]);
            //pIj[b*nj+j] += (pAi[b*ni+i]) * bnz(pWij[j*ni+i]);
        }
        pAj[b*nj+j] = g(pIj[b*nj+j]);
    }
    return 0;
}

int backwardOut(const int bytes[],float * target, NNet *pNet, float * pAj, float * pIj, float * pAi, float * pGWij, float * pGsj){
    int b, j, i; // k for 3rd, j for 2nd, and i for 1st
    int nj = pNet->ns[pNet->nLs-1];
    //printf("nj= %d", nj );
    int ni = pNet->ns[pNet->nLs-2];
    //printf("ni= %d", ni );
    //assert(0);
    float Err[nj];
//#pragma omp parallel for private(b)
    for(b=0;b<BATCH_SIZE;++b){
        // The 4-th an 3rd layer can combine in the same b-loop because no batch normalization on layer 4
        target[bytes[b]] = 1.0; 
//#pragma omp parallel for private(j,i)
        for(j=0;j<nj;++j){
            Err[j] = target[j] - pAj[b*nj+j];

            pGsj[b*nj+j] = dg(pIj[b*nj+j]) * Err[j];

            for(i=0;i<ni;++i){ // No batch normalization on layer 4, so calculate dl/dWk directly
                pGWij[j*ni+i] += pAi[b*ni+i] * pGsj[b*nj+j]/((float)BATCH_SIZE);
            }
        }
        target[bytes[b]] = 0.0; 
    }
    return 0;
}

int backwardGyb(int nj, int nk, float * pGybij, float * pWjk, float * pGsk){
    /// dl/dyk2
    int b, k, j; // k for 3rd, j for 2nd, and i for 1st
//#pragma omp parallel for private(b)
    for(b=0;b<BATCH_SIZE;++b){
        #pragma omp parallel for private(j,k)
        for(j=0;j<nj;++j){
            pGybij[b*nj+j] = 0.0;
            for(k=0;k<nk;++k){
                pGybij[b*nj+j] += pWjk[k*nj+j] * pGsk[b*nk+k];
            }
        }
    }

    return 0;
}

int backwardBN(int ni, int nj, float * pGybij, float * pGBj, float * pGxhbij, float * pGdsij, float * pAj,
               float * pUj, float * pDsj, float * pGuij, float * pGxbij, float * pGGBij, float * pxhj){

    int b, k, j; // k for 3rd, j for 2nd, and i for 1st
//#pragma omp parallel for private(j,b) 
    for(j=0;j<nj;++j){
//#pragma omp parallel for private(b) 
        for(b=0;b<BATCH_SIZE;++b){
            pGxhbij[b*nj+j] = pGybij[b*nj+j] * pGBj[j]; // Use dl/dxk to calculate dl/dWk in another nested loops
        }

        pGGBij[j] = 0.0; // The first half is for gamma
        pGGBij[nj+j] = 0.0; // The second half is for beta
//#pragma omp parallel for private(b) 
        for(b=0;b<BATCH_SIZE;++b){
            pGGBij[j]      += pGybij[b*nj+j] * pxhj[b*nj+j]; // The first half is for gamma
            pGGBij[nj+j]   += pGybij[b*nj+j]; // The second half is for beta
        }
    /// dl/dds2
        pGdsij[j] = 0.0;
//#pragma omp parallel for private(b) 
        for(b=0;b<BATCH_SIZE;++b){
            pGdsij[j] += pGxhbij[b*nj+j] * (pAj[b*nj+j] - pUj[j]) * (-0.5) *pow(pDsj[j] + EPSILON,-1.5);
        }
    /// dl/du2
        pGuij[j] = 0.0;
//#pragma omp parallel for private(b) 
        for(b=0;b<BATCH_SIZE;++b){
            pGuij[j] += pGxhbij[b*nj+j] * (-1.0)/sqrt(pDsj[j] + EPSILON) +
                (pGdsij[j]) * (-2.0/((float)BATCH_SIZE)) * (pAj[b*nj+j]-pUj[j]);
        }
    /// dl/dxk2
//#pragma omp parallel for private(b) 
        for(b=0;b<BATCH_SIZE;++b){
            pGxbij[b*nj+j] = pGxhbij[b*nj+j] / sqrt(pDsj[j] +EPSILON) +
                pGdsij[j]*(2.0/((float)BATCH_SIZE))*(pAj[b*nj+j] - pUj[j]) +
                pGuij[j]/((float)BATCH_SIZE);
        }
    }
    return 0;
}

int backwardGW(int ni, int nj, float * pGsj, float * pIj, float * pGxbij, float * pAi, float * pGWij){
    int i,j,b;
    /// dl/dWk2
//#pragma omp parallel for private(b) 
    for(b=0;b<BATCH_SIZE;++b){
#pragma omp parallel for private(j,i) 
        for(j=0;j<nj;++j){
            pGsj[b*nj+j] = dg(pIj[b*nj+j]) * pGxbij[b*nj+j]; // Resume the chain rule here
            for(i=0;i<ni;++i){
                pGWij[j*ni+i] += pAi[b*ni+i] * pGsj[b*nj+j]/((float)BATCH_SIZE);
            }
        }
    }
    return 0;
}


int accumulate(NNet *pNet, float alpha, int t){
    float B1t = pow(B1, t);
    int i,j,k,l;
#pragma omp parallel for private(l,k)
    for(l=0;l<N_4;++l){
        for(k=0;k<N_3;++k){
            adamax(pNet->pM34+l*N_3+k, pNet->pV34+l*N_3+k, pNet->pW34+l*N_3+k, alpha, pNet->pGWk34+l*N_3+k, t);
            pNet->pGWk34[l*N_3+k] = 0.0;
        }
    }

#pragma omp parallel for private(k,j)
    for(k=0;k<pNet->ns[2];++k){
        for(j=0;j<pNet->ns[1];++j){
            adamax(pNet->pM23+k*N_2+j, pNet->pV23+k*N_2+j, pNet->pW23+k*N_2+j,  (1.0/sqrt((float)N_3))*alpha, pNet->pGWk23+k*N_2+j, t);
            pNet->pGWk23[k*N_2+j] = 0.0;
        }
        if(BN||BNZ) adamax(pNet->pMgb3+k,     pNet->pVgb3+k,     pNet->pGB3+k,      4*alpha, pNet->pGGB23+k,    t);
        if(BN||BNZ) adamax(pNet->pMgb3+N_3+k, pNet->pVgb3+N_3+k, pNet->pGB3+N_3+k, 0.5*alpha, pNet->pGGB23+N_3+k,t);
    }

    //printf("pGyki12: ");prtArr(pNet->pGyki12,8);
    //prtArr(pNet->pxh2,8);
    //prtArr(pNet->pGGB12,8);
#pragma omp parallel for private(j,i)
    for(j=0;j<pNet->ns[1];++j){
        for(i=0;i<pNet->ns[0];++i){
            adamax(pNet->pM12+j*N_1+i, pNet->pV12+j*N_1+i, pNet->pW12+j*N_1+i, (1.0/sqrt((float)N_2))*alpha, pNet->pGWk12+j*N_1+i, t);
            pNet->pGWk12[j*N_1+i] = 0.0;
        }
        //adamax(pNet->pMgb2+j,     pNet->pVgb2+j,     pNet->pGB2+j,      10*alpha, pNet->pGGB12+j,    t);
        if(BN||BNZ) adamax(pNet->pMgb2+j,     pNet->pVgb2+j,     pNet->pGB2+j,      4*alpha, pNet->pGGB12+j,    t);
        if(BN||BNZ) adamax(pNet->pMgb2+N_2+j, pNet->pVgb2+N_2+j, pNet->pGB2+N_2+j, 0.5*alpha, pNet->pGGB12+N_2+j,t);
    }

    return 0;
}

int adamax(float * m, float * vt, float * theta, float ita, float * gt, int t){
    *m = B1*(*m) + (1.0-B1) * (*gt);
    *vt = max(B2*(*vt), fabs(*gt));

    *theta += (ita/(1.0 - pow(B1,t)))*(*m)/(*vt+EPSILON);

    return 0; 
}


int batchNml(NNet * pNet, int n){
    //printf("batchNml\n");
    float * pU;
    float * pD;
    float * pinA;
    float * pninA;
    float * poutA;
    float * pgb;
    if(n==1){ // Got to change someday
        pU = pNet->pU2;
        pD = pNet->pDs2;
        pinA = pNet->pA2;
        pninA = pNet->pxh2;
        poutA = pNet->pBa2;
        pgb = pNet->pGB2;
    }else{
        pU = pNet->pU3;
        pD = pNet->pDs3;
        pinA = pNet->pA3;
        pninA = pNet->pxh3;
        poutA = pNet->pBa3;
        pgb = pNet->pGB3;
    }


    int i,b, nc;
    nc = pNet->ns[n];
    /*
    if(!(BN||BNZ)){ /// Turn off normalization
        for(i=0;i<nc;++i)
            for(b=0;b<BATCH_SIZE;++b)
                poutA[b*nc+i] = pinA[b*nc+i];
        return 0;
    }
    */
#pragma omp parallel for private(i,b)
    for(i=0;i<nc;++i){

        pU[i] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pU[i] += pinA[b*nc+i];
        }
        pU[i] /= (float) BATCH_SIZE;

        /// shift
        for(b=0;b<BATCH_SIZE;++b){
            pninA[b*nc+i] = pinA[b*nc+i] - pU[i];
        }

        // *(dvi+i) = 0.0;
        pD[i] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pD[i] += pninA[b*nc+i] * pninA[b*nc+i];
        }
        pD[i] /= (float)BATCH_SIZE;

        // normalize --> scale --> shift
        for(b=0;b<BATCH_SIZE;++b){
            pninA[b*nc+i] /= sqrt(pD[i]+EPSILON);

            poutA[b*nc+i] = pninA[b*nc+i]*pgb[i] + pgb[nc+i];
        }
    }
    //printf("u2: ");prtArr(pU,8);
    //printf("D2: ");prtArr(pD,8);
    //printf("xh2: ");prtArr(pNet->pxh2,8);
    //printf("Ba2: ");prtArr(pNet->pBa2,8);
    //assert(0);

    return 0;
}

float cost1(const float target[], float output[]){
    float delta = 0.0;
    float cost1 = 0.0;
    int i;
//#pragma omp parallel for private(i)
    for(i=0;i!=N_5;++i){
        delta = target[i] - output[i];
        cost1 += 0.5*delta*delta;
    }

    return cost1;
}

/* Cross entropy and softmax */
float cost2(const float target[], float output[]){
    float delta = 0.0;
    float cost1 = 0.0;
    int i;
//#pragma omp parallel for private(i)
    for(i=0;i!=N_5;++i){
        cost1 -= target[i]*log(output[i]);
    }

    return cost1;
}

int saveWeights(const char str[], NNet *pNet){
    FILE *fp = fopen(str, "wb");

    //fwrite(&(pNet->nLs),sizeof(int), 1, fp);
    fwrite(pNet->ns,   sizeof(int),    pNet->nLs, fp);
    fwrite(pNet->pW12, sizeof(float), (pNet->ns[0])*(pNet->ns[1]), fp);
    fwrite(pNet->pW23, sizeof(float), (pNet->ns[1])*(pNet->ns[2]), fp);
    fwrite(pNet->pW34, sizeof(float), (pNet->ns[2])*(pNet->ns[3]), fp);
    fwrite(pNet->pGB2, sizeof(float), 2*(pNet->ns[1]), fp);
    fwrite(pNet->pGB3, sizeof(float), 2*(pNet->ns[2]), fp);

    fclose(fp);
    return 0;
}

int readWeights(const char str[], NNet *pNet){
    FILE *fp = fopen(str, "rb");

    int c;
    //c = fread(&(pNet->nLs),sizeof(int), 1, fp);
    c = fread(pNet->ns,   sizeof(int),    pNet->nLs, fp);
    c = fread(pNet->pW12, sizeof(float), (pNet->ns[0])*(pNet->ns[1]), fp);
    c = fread(pNet->pW23, sizeof(float), (pNet->ns[1])*(pNet->ns[2]), fp);
    c = fread(pNet->pW34, sizeof(float), (pNet->ns[2])*(pNet->ns[3]), fp);
    c = fread(pNet->pGB2, sizeof(float), 2*(pNet->ns[1]), fp);
    c = fread(pNet->pGB3, sizeof(float), 2*(pNet->ns[2]), fp);

    fclose(fp);
    return 0;
}

int backward(const int bytes[], NNet *pNet){
    int b, l, k, j, i; // k for 3rd, j for 2nd, and i for 1st
    float Errk[N_4];
    float Deltal[N_4];
    float target[N_4];
//#pragma omp parallel for private(b)
    for(b=0;b<BATCH_SIZE;++b){
        // The 4-th an 3rd layer can combine in the same b-loop because no batch normalization on layer 4
        target[bytes[b]] = 1.0; 
//#pragma omp parallel for private(l,k)
        for(l=0;l<N_4;++l){
            Errk[l] = target[l] - pNet->pA4[b*N_4+l];

            Deltal[l] = Errk[l] * dg(pNet->pI4[b*N_4+l]);

            for(k=0;k<N_3;++k){ // No batch normalization on layer 4, so calculate dl/dWk directly
                pNet->pGWk34[l*N_3+k] += pNet->pBa3[b*N_3+k] * Deltal[l]/((float)BATCH_SIZE);
            }
        }
        target[bytes[b]] = 0.0; 

        /// dl/dyk3
//#pragma omp parallel for private(k,l)
        for(k=0;k<N_3;++k){
            pNet->pGyki23[b*N_3+k] = 0.0;
            for(l=0;l<N_4;++l){
                pNet->pGyki23[b*N_3+k] += pNet->pW34[l*N_3+k] * Deltal[l];
            }

            pNet->pGxhki23[b*N_3+k] = pNet->pGyki23[b*N_3+k] * pNet->pGB3[k]; // Use dl/dxk to calculate dl/dWk in another nested loops
        }
    }

    //printf("after layer 3\n");
    /// dl/dds3
//#pragma omp parallel for private(k,b) 
    for(k=0;k<N_3;++k){
        pNet->pGds23[k] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGds23[k] += pNet->pGxhki23[b*N_3+k]*(pNet->pA3[b*N_3+k]-pNet->pU3[k])*(-0.5)*pow(pNet->pDs3[k]+EPSILON,-1.5);
        }
    }
    /// dl/du3
//#pragma omp parallel for private(k,b) 
    for(k=0;k<N_3;++k){
        pNet->pGu23[k] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGu23[k] += pNet->pGxhki23[b*N_3+k]*(-1.0)/sqrt(pNet->pDs3[k]+EPSILON) +
                (pNet->pGds23[k])*(-2.0/((float)BATCH_SIZE))*(pNet->pA3[b*N_3+k]-pNet->pU3[k]);
        }
    }
    /// dl/dxk3
//#pragma omp parallel for private(k,b) 
    for(k=0;k<N_3;++k){
        pNet->pGGB23[k] = 0.0; // The first half is for gamma
        pNet->pGGB23[N_3+k] = 0.0; // The second half is for beta

        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGxki23[b*N_3+k] = pNet->pGxhki23[b*N_3+k] / sqrt(pNet->pDs3[k]+EPSILON) +
                pNet->pGds23[k]*(2.0/((float)BATCH_SIZE))*(pNet->pA3[b*N_3+k]-pNet->pU3[k]) +
                pNet->pGu23[k]/((float)BATCH_SIZE);

            pNet->pGGB23[k] += pNet->pGyki23[b*N_3+k]*pNet->pxh3[b*N_3+k]; // The first half is for gamma
            pNet->pGGB23[N_3+k] += pNet->pGyki23[b*N_3+k]; // The second half is for beta
        }
    }
    /// dl/dWk3
    float * Deltak = malloc(sizeof(float)*BATCH_SIZE*N_3);
    for(b=0;b<BATCH_SIZE;++b){
//#pragma omp parallel for private(k,j) 
        for(k=0;k<N_3;++k){
            Deltak[b*N_3+k] = dg(pNet->pI3[b*N_3+k]) * pNet->pGxki23[b*N_3+k]; // Resume the chain rule here
            //Deltak[b*N_3+k] = dg(pNet->pI3[b*N_3+k]) * pNet->pGyki23[b*N_3+k]; // Resume the chain rule here
            for(j=0;j<N_2;++j){
                pNet->pGWk23[k*N_2+j] += pNet->pA2[b*N_2+j] * Deltak[b*N_3+k]/((float)BATCH_SIZE);
            }
        }
    }

    /// dl/dyk2
    for(b=0;b<BATCH_SIZE;++b){
//#pragma omp parallel for private(j,k)
        for(j=0;j<N_2;++j){
            pNet->pGyki12[b*N_2+j] = 0.0;
            for(k=0;k<N_3;++k){
                pNet->pGyki12[b*N_2+j] += pNet->pW23[k*N_2+j] * Deltak[b*N_3+k];
            }

            pNet->pGxhki12[b*N_2+j] = pNet->pGyki12[b*N_2+j] * pNet->pGB2[j]; // Use dl/dxk to calculate dl/dWk in another nested loops
        }
    }
    free(Deltak);
    /// dl/dds2
//#pragma omp parallel for private(j,b) 
    for(j=0;j<N_2;++j){
        pNet->pGds12[j] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGds12[j] += pNet->pGxhki12[b*N_2+j]*(pNet->pA2[b*N_2+j]-pNet->pU2[j])*(-0.5)*pow(pNet->pDs2[j]+EPSILON,-1.5);
        }
    }
    /// dl/du2
//#pragma omp parallel for private(j,b) 
    for(j=0;j<N_2;++j){
        pNet->pGu12[j] = 0.0;
        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGu12[j] += pNet->pGxhki12[b*N_2+j]*(-1.0)/sqrt(pNet->pDs2[j]+EPSILON) +
                (pNet->pGds12[j])*(-2.0/((float)BATCH_SIZE))*(pNet->pA2[b*N_2+j]-pNet->pU2[j]);
        }
    }
    /// dl/dxk2
//#pragma omp parallel for private(j,b) 
    for(j=0;j<N_2;++j){
        pNet->pGGB12[j] = 0.0; // The first half is for gamma
        pNet->pGGB12[N_2+j] = 0.0; // The second half is for beta

        for(b=0;b<BATCH_SIZE;++b){
            pNet->pGxki12[b*N_2+j] = pNet->pGxhki12[b*N_2+j] / sqrt(pNet->pDs2[j]+EPSILON) +
                pNet->pGds12[j]*(2.0/((float)BATCH_SIZE))*(pNet->pA2[b*N_2+j]-pNet->pU2[j]) +
                pNet->pGu12[j]/((float)BATCH_SIZE);

            pNet->pGGB12[j] += pNet->pGyki12[b*N_2+j]*pNet->pxh2[b*N_2+j]; // The first half is for gamma
            pNet->pGGB12[N_2+j] += pNet->pGyki12[b*N_2+j]; // The second half is for beta
        }
    }
    /// dl/dWk2
    float * Deltaj = malloc(sizeof(float)*BATCH_SIZE*N_2);
    //float Deltaj[N_2];
    for(b=0;b<BATCH_SIZE;++b){
//#pragma omp parallel for private(j,i) 
        for(j=0;j<N_2;++j){
            Deltaj[b*N_2+j] = dg(pNet->pI2[b*N_2+j]) * pNet->pGxki12[b*N_2+j]; // Resume the chain rule here
            //Deltaj[j] = dg(pNet->pI2[b*N_2+j]) * pNet->pGxki12[b*N_2+j]; // Resume the chain rule here
            //Deltaj[b*N_2+j] = dg(pNet->pI2[b*N_2+j]) * pNet->pGyki12[b*N_2+j]; // Resume the chain rule here
            for(i=0;i<N_1;++i){
                pNet->pGWk12[j*N_1+i] += pNet->pA1[b*N_1+i] * Deltaj[b*N_2+j]/((float)BATCH_SIZE);
                //pNet->pGWk12[j*N_1+i] += pNet->pA1[b*N_1+i] * Deltaj[j]/((float)BATCH_SIZE);
            }
        }
    }
    //assert(0);
    free(Deltaj);

    return 0;
}

