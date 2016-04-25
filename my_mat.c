/*****************************
 * Source of my math library *
 * By Jeng-Hau Lin           *
 *****************************/

#include "my_mat.h"

float act_pct(float x){
    return x;
}
float dact_pct(float x){
    return 1;
}

float sigmoid(float x){
    return 1/(1+exp(-x));
}
float dsigmoid(float x){
    return sigmoid(x)*(1-sigmoid(x));
}

float softmax(float x, float result[], const int n_c){

    float sum1 = 0.0;
    int i;
#pragma parallel for private(i)
    for(i=0;i!=n_c;++i){
        sum1 += exp(result[i]);
    }

    return exp(x)/sum1;
}

float crossEntropy(float target[], float result[], const int n_c){
    float sum1 = 0.0;
    int i;
#pragma parallel for private(i)
    for(i=0;i!=n_c;++i){
        sum1 += target[i]*log(result[i]);
    }

    return sum1;
}

float bnz(float x){
    return (x>=0)?(1.0):(-1.0);
}

float sbnz(float x){
    return (x>=0)?(1.0):(0.0);
}

int rndArr(float *arr, const size_t n_c){
    int i;
#pragma parallel for private(i)
    for(i=0;i!=n_c;++i){
        *(arr+i) = (rand()%100) / 10000.0;
    }
    return 0;
}

int rndMtx(float *mtx, const size_t n_r ,const size_t n_c){
    int r, c;
    //printf("n_r = %d\n", (int)n_r);
    //printf("n_c = %d\n", (int)n_c);
#pragma parallel for private(r,c)
    for(r=0;r!=n_r;++r){
        for(c=0;c!=n_c;++c){
            *(mtx + r*n_c + c) = (rand()%200 - 100.0) / 10000.0;
            //*(mtx + r*n_c + c) = (rand()%101) /10000.0;
        }
    }

    //prtMtx((float*)mtx, n_r, n_c);
    return 0;
}

int shufArr(int *arr, const size_t nc){
    //srand(time(NULL));
    int i;
    for(i=0;i!=nc;++i){
        int temp = arr[i];
        int randomIdx = rand()%nc;
        arr[i] = arr[randomIdx];
        arr[randomIdx] = temp;
    }

    return 0;
}

int checkMax(float * output, int nc, int ans_byte){

    float outMaxV = output[ans_byte];
    int i;
    for(i=0;i<nc;++i){
        if(i==ans_byte)
            continue;

        if(output[i]>=outMaxV)
            return 0;
    }

    return 1;
}

int prtArr(float arr[], const size_t n){
    int i;
    for(i=0;i!=n;++i){
        printf("%10f,", arr[i]);
    }
    printf("\n");

    return 0;
}

int prtMtx(float *mtx, const size_t n_r, const size_t n_c){
    int r, c;
    for(r=0;r!=n_r;++r){
        for(c=0;c!=n_c;++c){
            printf("%10f,", *(mtx + r*n_c + c));
        }
        printf("\n");
    }
    return 0;
}
