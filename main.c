/*****************************
 * Train a Neural Network    *
 * plan implementation       *
 * save weights in memory *
 * 5-layer:                  *
 *         1 input layer     *
 *         2 hidden layer    *
 *         3 hidden layer    *
 *         4 hidden layer    *
 *         5 output layer    *
 * By Jang-Hau Lin           *
 *****************************/

#include <assert.h>
#include "MLP_5Ls.h"

#define TRAIN_MODE 1

#define Ntrains 100

// batch normalization + Adamax
int main(int argc, char *argv[]){
    srand(time(NULL)*getpid());
// Declarations
    const float threshold = 0.0001; // The threshold controls when to skip back_propogate

    int t = 0;
    float alpha = ALPHA;
    // Weights
    float * p_w12 = malloc(sizeof(float)*N_2*N_1);
    float * p_w23 = malloc(sizeof(float)*N_3*N_2);
    float * p_w34 = malloc(sizeof(float)*N_4*N_3);
    // Record the inputs of layer 2 and 3 for different mini-batch
    float * p_in2 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_in3 = malloc(sizeof(float)*BATCH_SIZE*N_3);
    float * p_in4 = malloc(sizeof(float)*BATCH_SIZE*N_4);
    // Record the activations of layer 2 and 3 for different mini-batch
    float * p_act1 = malloc(sizeof(float)*BATCH_SIZE*N_1);
    float * p_act2 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_act3 = malloc(sizeof(float)*BATCH_SIZE*N_3);
    float * p_act4 = malloc(sizeof(float)*BATCH_SIZE*N_4);

    int idx;
    float * gb2 = malloc(sizeof(float)*2*N_2);
    for(idx=0;idx<N_2;++idx)
        gb2[idx] = GAMMA;
    for(idx=N_2;idx<2*N_2;++idx)
        gb2[idx] = BETA;
    float * gb3 = malloc(sizeof(float)*2*N_3);
    for(idx=0;idx<N_3;++idx)
        gb3[idx] = GAMMA;
    for(idx=N_3;idx<2*N_3;++idx)
        gb3[idx] = BETA;

    float * mgb2 = malloc(sizeof(float)*2*N_2);
    float * mgb3 = malloc(sizeof(float)*2*N_3);
    float * vgb2 = malloc(sizeof(float)*2*N_2);
    float * vgb3 = malloc(sizeof(float)*2*N_3);

    /// momentum for all weights
    float * p_m12 = malloc(sizeof(float)*N_2*N_1);
    float * p_m23 = malloc(sizeof(float)*N_3*N_2);
    float * p_m34 = malloc(sizeof(float)*N_4*N_3);
    /// inertia for all weights
    float * p_v12 = malloc(sizeof(float)*N_2*N_1);
    float * p_v23 = malloc(sizeof(float)*N_3*N_2);
    float * p_v34 = malloc(sizeof(float)*N_4*N_3);

    /// dl/dyk_i for each sample
    float * p_gyki12 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_gyki23 = malloc(sizeof(float)*BATCH_SIZE*N_3);
    /// dl/dxhk_i for each sample
    float * p_gxhki12 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_gxhki23 = malloc(sizeof(float)*BATCH_SIZE*N_3);

    /// dl/du for a mini-batch
    float * p_gu12 = malloc(sizeof(float)*N_2);
    float * p_gu23 = malloc(sizeof(float)*N_3);
    /// dl/dds for a mini-batch
    float * p_gds12 = malloc(sizeof(float)*N_2);
    float * p_gds23 = malloc(sizeof(float)*N_3);

    /// dl/dxhk_i for each sample
    float * p_gxki12 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_gxki23 = malloc(sizeof(float)*BATCH_SIZE*N_3);

    /// averaged dl/dgamma and dl/dbeta for a mini-batch
    float * p_gGB12 = malloc(sizeof(float)*2*N_2);
    float * p_gGB23 = malloc(sizeof(float)*2*N_3);

/// derivatives for batch normalization
    /// dl/dyk for a mini-batch
    float * p_gWk12 = malloc(sizeof(float)*N_2*N_1);
    float * p_gWk23 = malloc(sizeof(float)*N_3*N_2);
    float * p_gWk34 = malloc(sizeof(float)*N_4*N_3);

    // Record the normalized inputs of layer 2 and 3 for different mini-batch
    float * p_xh2 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_xh3 = malloc(sizeof(float)*BATCH_SIZE*N_3);
    /// Record the scaled and shifted inputs of layer 2 and 3 for different mini-batch
    float * p_ba2 = malloc(sizeof(float)*BATCH_SIZE*N_2);
    float * p_ba3 = malloc(sizeof(float)*BATCH_SIZE*N_3);
    // Remember the mean of inputs acrossa mini-batch
    float * p_u2 = malloc(sizeof(float)*N_2);
    float * p_u3 = malloc(sizeof(float)*N_3);
    // Remember the standard deviation of inputs acrossa mini-batch
    float * p_ds2 = malloc(sizeof(float)*N_2);
    float * p_ds3 = malloc(sizeof(float)*N_3);





    NNet my_NNet2 = {
        // Put weights in heap
        p_w12, // the weights between layer 1 and 2, size [N_2*N_1]
        p_w23, // the weights between layer 2 and 3, size [N_3*N_2]
        p_w34, // the weights between layer 2 and 3, size [N_4*N_3]

        p_in2, // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_2]
        p_in3, // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_3]
        p_in4, // the wighted sums of the 2nd nodes, size [BATCH_SIZE*N_4]
        p_act1, // the activations of the 1nd nodes, size [BATCH_SIZE*N_1]
        p_act2, // the activations of the 2nd nodes, size [BATCH_SIZE*N_2]
        p_act3, // the activations of the 3nd nodes, size [BATCH_SIZE*N_3]
        p_act4, // the activations of the 3nd nodes, size [BATCH_SIZE*N_4]

        gb2, // The array of gamma and beta
        gb3, // The array of gamma and beta

        p_xh2, // the normalized activations of the 2nd nodes, size [BATCH_SIZE*N_2]
        p_xh3, // the normalized activations of the 2nd nodes, size [BATCH_SIZE*N_3]
        p_ba2, // the scaled and shifted activations of the 2nd nodes, size [BATCH_SIZE*N_2]
        p_ba3, // the scaled and shifted activations of the 2nd nodes, size [BATCH_SIZE*N_3]

        p_u2, // average of in2 across a mini-batch, size [N_2]
        p_u3, // average of in3 across a mini-batch, size [N_3]
        p_ds2, // standard deviation of in2 across a mini-batch, size [N_2]
        p_ds3, // standard deviation of in3 across a mini-batch, size [N_3]

        p_gyki12, // d(l)/dyk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
        p_gyki23, // d(l)/dyk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]
        p_gxhki12, // d(l)/dxhk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
        p_gxhki23, // d(l)/dxhk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]
        p_gxki12, // d(l)/dxk of layer2 across a mini-batch, size [BATCH_SIZE*N_2]
        p_gxki23, // d(l)/dxk of layer3 across a mini-batch, size [BATCH_SIZE*N_3]

        p_gu12, // d(l)/du of layers, size [N_2]
        p_gu23, // d(l)/du of layers, size [N_3]
        p_gds12, // d(l)/dds of layers, size [N_2]
        p_gds23, // d(l)/dds of layers, size [N_3]

        p_m12, // the momentum of w12, size [N_2*N_1]
        p_m23, // the momentum of w12, size [N_3*N_2]
        p_m34, // the momentum of w12, size [N_4*N_3]
        p_v12, // the inertia of w12, size [N_2*N_1]
        p_v23, // the inertia of w12, size [N_3*N_2]
        p_v34, // the inertia of w12, size [N_4*N_3]

        p_gWk12, // d(l)/dSk of layer 2 across a mini-batch, size [N_2*N_1]
        p_gWk23, // d(l)/dSk of layer 3 across a mini-batch, size [N_3*N_2]
        p_gWk34, // d(l)/dSk of layer 3 across a mini-batch, size [N_4*N_3]

        mgb2, // moment of gamma and beta on layer 2, size [2*N_2]
        mgb3, // moment of gamma and beta on layer 3, size [2*N_3]

        vgb2, // moment of gamma and beta on layer 2, size [2*N_2]
        vgb3, // moment of gamma and beta on layer 3, size [2*N_3]

        p_gGB12, //averaged d(l)/dgamma and d(l)/dbeta of layer across a mini-batc2, size [2*N_2]
        p_gGB23, //averaged d(l)/dgamma and d(l)/dbeta of layer3 across a mini-batc, size [2*N_3]

        N_LAYERS, 
        {N_1, N_2, N_3, N_4}}; // numbers of nodes in layers (a N_LAYERS array)

    // Initialize the weights randomly
    if(TRAIN_MODE) rndMtx(((float *)my_NNet2.pW12), N_2, N_1);
    if(TRAIN_MODE) rndMtx(((float *)my_NNet2.pW23), N_3, N_2);
    if(TRAIN_MODE) rndMtx(((float *)my_NNet2.pW34), N_4, N_3);

    // Some pre-trained data for testing
    if(!TRAIN_MODE) readWeights("./LOG/weights.bin", &my_NNet2); // All training sets, 100 iterations, batch-size = 10

    char * trLfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-labels-idx1-ubyte";
    char * trIfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-images-idx3-ubyte";
    //char * teLfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-labels-idx1-ubyte";
    //char * teIfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-images-idx3-ubyte";
    char * teLfile = "/home/james/Codes/Cpp/MNIST/dataset/test/t10k-labels-idx1-ubyte";
    char * teIfile = "/home/james/Codes/Cpp/MNIST/dataset/test/t10k-images-idx3-ubyte";
    
    int fl, fi;
	if(TRAIN_MODE) fl = open(trLfile, O_RDONLY, 0);
    else fl = open(teLfile, O_RDONLY, 0);
	if(TRAIN_MODE) fi = open(trIfile, O_RDONLY, 0);
    else fi = open(teIfile, O_RDONLY, 0);
	u_int32_t ilen = 28*28;
    int c = 0;
    u_char *image = (u_char *)malloc(ilen);
	u_char byte = 0;
    // The index of the images
    int indexArr[N_SAMPLES];
    for(idx=0;idx<N_SAMPLES;++idx)
        indexArr[idx] = idx;

// Training over the same dataset for Ntrains times
    // Count errors
    int errCnt = 0;
    float b_cost =0.0;
    float a_cost;
    float target[10];
    for(idx=0;idx<10;++idx) // different system like mesl-exp has different initial value!!!
        target[idx] = 0.0;

    time_t timer1, timer2;
    double seconds;
    time(&timer1);
    int k,i;
    for(k=0;k!=Ntrains;++k){

        shufArr(indexArr, N_SAMPLES);

        i = 0;
        errCnt = 0;
        while(i<N_SAMPLES){
            int ib;
            int bytes[BATCH_SIZE] = {0};
            // Forward propogation
            for(ib=0;ib<BATCH_SIZE;++ib){
                lseek(fi, 16 + indexArr[i+ib]*ilen, SEEK_SET);
                lseek(fl, 8 + indexArr[i+ib]*1, SEEK_SET);
                c = read(fi, image, ilen);
                c = read(fl, &byte, sizeof(byte));

                bytes[ib] = byte;

                forwardIn(image, &my_NNet2, ib);

                if(BNZ) forwardBnz(my_NNet2.pA1, my_NNet2.pW12, my_NNet2.pI2, my_NNet2.pA2, N_1, N_2, ib);
                else       forward(my_NNet2.pA1, my_NNet2.pW12, my_NNet2.pI2, my_NNet2.pA2, N_1, N_2, ib);
            }
            
            // batch Normalization
            if(BN||BNZ) batchNml(&my_NNet2, 1);
            //printf("after 1st batch normalization\n");

            for(ib=0;ib<BATCH_SIZE;++ib){
                //forward23(& my_NNet2, ib);
                if(BNZ) forwardBnz(my_NNet2.pBa2, my_NNet2.pW23, my_NNet2.pI3, my_NNet2.pA3, N_2, N_3, ib);
                else if(!BNZ&&BN) forward(my_NNet2.pBa2, my_NNet2.pW23, my_NNet2.pI3, my_NNet2.pA3, N_2, N_3, ib);
                else       forward(my_NNet2.pA2, my_NNet2.pW23, my_NNet2.pI3, my_NNet2.pA3, N_2, N_3, ib);
            }

            // batch Normalization
            if(BN||BNZ) batchNml(&my_NNet2, 2);
            //printf("after 2nd batch normalization\n");

            //b_cost = 0.0;
            for(ib=0;ib<BATCH_SIZE;++ib){
                target[bytes[ib]] = 1.0;

                //forward34(& my_NNet2, ib);
                forward(my_NNet2.pBa3, my_NNet2.pW34, my_NNet2.pI4, my_NNet2.pA4, N_3, N_4, ib);

                if(!checkMax(my_NNet2.pA4+ib*N_4, N_4, bytes[ib])){
                    ++errCnt;
                }

                a_cost = cost1(target, my_NNet2.pA4+ib*N_4);
                b_cost += a_cost;
                target[bytes[ib]] = 0.0;
            }
            //printf("after forward34\n");
            i += BATCH_SIZE;

            // Backward propogation
            if(TRAIN_MODE){
                ++t;
                alpha *= LAMBDA;

/// Layer 4
                float * p_gs4 = malloc(sizeof(float)*BATCH_SIZE*N_4);
                backwardOut(bytes, target, &my_NNet2, my_NNet2.pA4, my_NNet2.pI4, my_NNet2.pBa3, my_NNet2.pGWk34, p_gs4);
/// Layer 3
                backwardGyb(N_3, N_4, my_NNet2.pGyki23, my_NNet2.pW34, p_gs4);
                free(p_gs4);

                float * p_gs3 = malloc(sizeof(float)*BATCH_SIZE*N_3);
                if(BN||BNZ){
                    backwardBN(N_2, N_3, my_NNet2.pGyki23, my_NNet2.pGB3, my_NNet2.pGxhki23, my_NNet2.pGds23, my_NNet2.pA3, 
                               my_NNet2.pU3, my_NNet2.pDs3, my_NNet2.pGu23, my_NNet2.pGxki23, my_NNet2.pGGB23, my_NNet2.pxh3);
                    backwardGW(N_2, N_3, p_gs3, my_NNet2.pI3, my_NNet2.pGxki23, my_NNet2.pA2, my_NNet2.pGWk23);
                }else{
                    backwardGW(N_2, N_3, p_gs3, my_NNet2.pI3, my_NNet2.pGyki23, my_NNet2.pA2, my_NNet2.pGWk23); // turn off BN
                }

/// Layer 2
                backwardGyb(N_2, N_3, my_NNet2.pGyki12, my_NNet2.pW23, p_gs3);
                free(p_gs3);

                float * p_gs2 = malloc(sizeof(float)*BATCH_SIZE*N_2);
                if(BN||BNZ){
                    backwardBN(N_1, N_2, my_NNet2.pGyki12, my_NNet2.pGB2, my_NNet2.pGxhki12, my_NNet2.pGds12, my_NNet2.pA2, 
                               my_NNet2.pU2, my_NNet2.pDs2, my_NNet2.pGu12, my_NNet2.pGxki12, my_NNet2.pGGB12, my_NNet2.pxh2);
                    backwardGW(N_1, N_2, p_gs2, my_NNet2.pI2, my_NNet2.pGxki12, my_NNet2.pA1, my_NNet2.pGWk12);
                }else{
                    backwardGW(N_1, N_2, p_gs2, my_NNet2.pI2, my_NNet2.pGyki12, my_NNet2.pA1, my_NNet2.pGWk12); // turn off BN
                }
                free(p_gs2);

                //backward(bytes, &my_NNet2);
                //printf("after backward\n");

                accumulate(&my_NNet2, alpha, t);
            }
        }

        if(Ntrains >=100 && (k+1)%(Ntrains/100)==0){
            // print out the percentage of calculation and loss averaged by a mini-batch
            //printf("%d,%f",(k+1)*100/Ntrains, b_cost/((float) BATCH_SIZE));
            printf("%d,%f",(k+1)*100/Ntrains, b_cost/((float) N_SAMPLES)*(Ntrains/100.0));
            printf(" %f, %f\n", gb2[101], gb2[N_2+101]);
            b_cost = 0.0;
            fflush(stdout);
        }

        if(!TRAIN_MODE) break;
    }
    close(fi);
    close(fl);

    time(&timer2);
    seconds = difftime(timer2,timer1);

    printf("\n");
    int a,b;
    for(a=0;a<28;++a){
        for(b=0;b<=28;++b){
            printf("%3d", *(image + a*28 +b));
        }
        printf("\n");
    }
    printf("\n");
    printf("the last target: %d\n", byte);
    printf("the last output:\n");
    //prtArr(output, N_4);
    prtArr(my_NNet2.pA4+(BATCH_SIZE-1)*N_4, N_4);
    printf("weights:\n");
    prtMtx((float *)my_NNet2.pW12, 1,4);
    printf("Time elapsed: %d\n", (int)seconds);

    printf("error = %f%%\n", (float)errCnt/((float)N_SAMPLES)*100.0);

    if(TRAIN_MODE) saveWeights("./LOG/weights.bin", &my_NNet2);

    free(p_w12);
    free(p_w23);
    free(p_w34);

    free(p_m12);
    free(p_m23);
    free(p_m34);

    free(p_v12);
    free(p_v23);
    free(p_v34);

    free(p_gyki12);
    free(p_gyki23);

    free(p_gxhki12);
    free(p_gxhki23);

    free(p_gu12);
    free(p_gu23);

    free(p_gds12);
    free(p_gds23);

    free(p_gxki12);
    free(p_gxki23);

    free(p_gGB12);
    free(p_gGB23);

    free(p_gWk12);
    free(p_gWk23);

    free(p_in2);
    free(p_in3);
    free(p_in4);

    free(p_xh2);
    free(p_xh3);

    free(p_ba2);
    free(p_ba3);

    free(p_u2);
    free(p_u3);

    free(p_ds2);
    free(p_ds3);

    free(p_act1);
    free(p_act2);
    free(p_act3);
    free(p_act4);

    return 0;
}

