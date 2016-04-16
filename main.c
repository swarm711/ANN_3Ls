/*****************************
 * Train a Neural Network    *
 * plan implementation       *
 * save weightings in memory *
 * 5-layer:                  *
 *         1 input layer     *
 *         2 hidden layer    *
 *         3 output layer    *
 * By Jang-Hau Lin           *
 *****************************/

#include "MLP_3Ls.h"

#define TRAIN_MODE 1 // 0 for test; 1 for training
#define Ntrains 100 // run the entire dataset for Ntrains times

int main(int argc, char *argv[]){
    srand(time(NULL)); // set random seed for shuffling the dataset
// Declarations
    int counter1 = 0; // The counter for plotting learning curve
    float alpha = 0.25; // Learning rate

    /* Weights */
    float * p_w12;
    p_w12 = (float *) malloc(sizeof(float)*N_2*N_1);
    float * p_w23;
    p_w23 = (float *) malloc(sizeof(float)*N_3*N_2);
    /* Record delta until minibatch size is reached */
    float * pDw12;
    pDw12 = (float *) malloc(sizeof(float)*N_2*N_1);
    float * pDw23;
    pDw23 = (float *) malloc(sizeof(float)*N_3*N_2);
    float * ppDw[] = {pDw12, pDw23};

    NNet my_NNet2 = {
        // Put weights in heap
        p_w12, // Weightings between layer 1 and 2 (a N_2 x N_1 matrix)
        p_w23, // Weightings between layer 2 and 3 (a N_3 x N_2 matrix)
       
        {0.0}, // weighted sum of layer 2 (a N_2 array)
        {0.0}, // weighted sum of layer 3 (a N_3 array)

        {0.0}, // activation of layer 1 (a N_1 array)
        {0.0}, // activation of layer 2 (a N_2 array)
        {0.0}, // activation of layer 3 (a N_3 array)
        N_LAYERS, 
        {N_1, N_2, N_3}}; // numbers of nodes in layers (a N_LAYERS array)
    if(TRAIN_MODE) rndMtx(((float *)my_NNet2.wt12), N_2, N_1);
    if(TRAIN_MODE) rndMtx(((float *)my_NNet2.wt23), N_3, N_2);

    if(!TRAIN_MODE) readWeights("./LOG/_weights_all_tr_b_10.bin", &my_NNet2); // All training sets, 100 iterations, batch-size = 10

    /* The training/test dataset*/
    //char * trLfile = "<the path to the training label dataset >/train-labels-idx1-ubyte";
    //char * trIfile = "<the path to the training image dataset >/train-images-idx3-ubyte";
    char * trLfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-labels-idx1-ubyte";
    char * trIfile = "/home/james/Codes/Cpp/MNIST/dataset/train/train-images-idx3-ubyte";
    char * teLfile = "<the path to the test label dataset >/t10k-labels-idx1-ubyte";
    char * teIfile = "<the path to the test image dataset >/t10k-images-idx3-ubyte";
    int fl, fi;
	if(TRAIN_MODE) fl = open(trLfile, O_RDONLY, 0);
    else fl = open(teLfile, O_RDONLY, 0);
	if(TRAIN_MODE) fi = open(trIfile, O_RDONLY, 0);
    else fi = open(teIfile, O_RDONLY, 0);
    int ilen = 28*28;
    int c = 0;
    u_char *image;
	u_char byte = 0;
	image = (u_char *)malloc(ilen);

    /* Skip the first few attributes */
    lseek(fi, 16, SEEK_SET);
    lseek(fl, 8, SEEK_SET);

    /* index array for shuffling */
    int indexArr[N_SAMPLES];
    int idx;
    for(idx=0;idx<N_SAMPLES;++idx)
        indexArr[idx] = idx;

// Training over the same dataset for Ntrains times
    int errCnt = 0; // Count the number of mis-classified outputs
    /* Measure the calculation time */
    time_t timer1, timer2;
    double seconds;
    time(&timer1);

    float acc_loss =0.0; // Accumulated loss for plotting of the learning curve
    float a_loss;    // Loss for a sample (label, image)
    float target[N_3] = {0.0};
    float output[N_3] = {0.0};

    int counter = 0; // Counter for mini-batch, if reach BATCH_SIZE, update the weights
    int k,i;
    for(k=0;k!=Ntrains;++k){
        /* If training, shuffle the dataset for stochastic mini-batch */
        if(TRAIN_MODE) shufArr(indexArr, N_SAMPLES);

        for(i=0;i!=N_SAMPLES;++i){
            ++counter;
            /* Read the training/test dataset, one pair of image and label at a time*/
            if(TRAIN_MODE) lseek(fi, 16 + indexArr[i]*ilen, SEEK_SET);
            if(TRAIN_MODE) lseek(fl, 8 + indexArr[i]*1, SEEK_SET);
            c = read(fi, image, ilen);
            c = read(fl, &byte, sizeof(byte));
            target[byte] = 1.0; // Set the target array corresponding to the label

            /* Forward propogation */
            forward(image, &my_NNet2, output);
            if(!TRAIN_MODE && output[byte]<0.9){
                ++errCnt;
            }

            /* Calculate the loss*/
            a_loss = cost1(target, output);
            acc_loss += a_loss; // Accumulate the loss

            // Backward propogation
            if(TRAIN_MODE)
                    backward(target, alpha,  &my_NNet2, ppDw, &counter);

            target[byte] = 0.0; // Reset the target array for the next sample
            
        }
        
        if(Ntrains >=100 && k%(Ntrains/100)==0){
            printf("%d%%(%.3f) ",k*100/Ntrains, acc_loss); // Accumulated loss
            acc_loss =0.0;
            fflush(stdout);
        }
        
        if(!TRAIN_MODE) break;
    }
	close(fi);
	close(fl);

    /* Finish the time interval */
    time(&timer2);
    seconds = difftime(timer2,timer1);

    /* Plot the last image and label for sanity check */
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
    prtArr(output, N_3);
    printf("Time elapsed: %d\n", (int)seconds);
    /* print out the training time */
    if(!TRAIN_MODE) printf("error = %f%%\n", (float)errCnt/((float)N_SAMPLES)*100.0);
    /* save the trained weights */
    if(TRAIN_MODE) saveWeights("./LOG/weights.bin", &my_NNet2);

    return 0;
}

