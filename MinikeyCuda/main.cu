
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

#include "Worker.cuh"

#include "lib/Int.h"
#include "lib/Math.cuh"
#include "lib/util.h"
#include "lib/SECP256k1.h"

#include "bloom_filter.hpp"

Secp256K1* secp;

using namespace std;

bool readArgs(int argc, char** argv);
void prepareAlphabet();
bool checkDevice();
void processCandidate(Int& toTest);
bool read_file(const std::string& file_name);
void incrementBase58(int count, int* key);
void printSpeed(double speed);
void saveStatus();
cudaError_t processCuda();

int DEVICE_NR = 0;
unsigned int BLOCK_THREADS = 0;
unsigned int BLOCK_NUMBER = 0;
unsigned int THREAD_STEPS = 5;
const int ALPHABET_LEN = 57;
char ALPHABET[ALPHABET_LEN];

string fileResult = "result.txt";
string fileStatus = "fileStatus.txt";
string fileInput = "";
int fileStatusInterval = 60;

const int KEY_LENGTH = 21;
int KEY_START_IX[KEY_LENGTH];
string KEY_START = "";
uint64_t outputSize;

bloom_parameters parameters;
bloom_filter filter;


int main(int argc, char** argv)
{
    prepareAlphabet();
    parameters.false_positive_probability = 0.000000001;
    parameters.projected_element_count = 5000;
    parameters.compute_optimal_parameters();
    filter = bloom_filter(parameters);

    readArgs(argc, argv);
    if (!checkDevice()) {
        return -1;
    }

    secp = new Secp256K1();
    secp->Init();

    auto time = std::chrono::system_clock::now();
    std::time_t s_time = std::chrono::system_clock::to_time_t(time);
    std::cout << "Work started at " << std::ctime(&s_time);

    cudaError_t cudaStatus = processCuda();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Device reset failed!");
        return 1;
    }
    
    return 0;
}

cudaError_t processCuda() {
    cudaError_t cudaStatus;
    int buffStart[KEY_LENGTH];
    int *dev_buffStart = new int[KEY_LENGTH];

    int COLLECTOR_SIZE = BLOCK_NUMBER;
    unsigned int* buffAlphabet = new unsigned int[ALPHABET_LEN];
    for (int i = 0; i < ALPHABET_LEN; i++) {
        buffAlphabet[i] = ALPHABET[i];
    }
    cudaStatus = loadAlphabet(buffAlphabet);

    unsigned int* buffDeviceResult = new unsigned int[outputSize * 8];
    unsigned int* dev_buffDeviceResult = new unsigned int[outputSize * 8];
    for (int i = 0; i < outputSize * 8; i++) {
        buffDeviceResult[i] = 0;
    }
    cudaStatus = cudaMalloc((void**)&dev_buffDeviceResult, outputSize * 8 * sizeof(unsigned int));
    cudaStatus = cudaMemcpyAsync(dev_buffDeviceResult, buffDeviceResult, outputSize * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int* buffResult = new unsigned int[COLLECTOR_SIZE * 8];
    unsigned int* dev_buffResult = new unsigned int[COLLECTOR_SIZE * 8];
    cudaStatus = cudaMalloc((void**)&dev_buffResult, COLLECTOR_SIZE * 8 * sizeof(unsigned int));
    cudaStatus = cudaMemcpy(dev_buffResult, buffResult, COLLECTOR_SIZE * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    bool* buffCollectorWork = new bool[1];
    buffCollectorWork[0] = false;
    bool* dev_buffCollectorWork = new bool[1];
    cudaStatus = cudaMalloc((void**)&dev_buffCollectorWork, 1 * sizeof(bool));
    cudaStatus = cudaMemcpy(dev_buffCollectorWork, buffCollectorWork, 1 * sizeof(bool), cudaMemcpyHostToDevice);

    cudaStatus = cudaMalloc((void**)&dev_buffStart, KEY_LENGTH * sizeof(int));

    uint64_t counter = 0;
    std::chrono::steady_clock::time_point beginCountHashrate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point beginCountStatus = std::chrono::steady_clock::now();

    while (true) {
        for (int i = 0; i < KEY_LENGTH; i++) {
            buffStart[i] = KEY_START_IX[i];
        }
        cudaStatus = cudaMemcpy(dev_buffStart, buffStart, KEY_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
        kernelMinikeys <<<BLOCK_NUMBER, BLOCK_THREADS >>> (dev_buffDeviceResult, dev_buffCollectorWork, dev_buffStart, THREAD_STEPS);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(buffCollectorWork, dev_buffCollectorWork, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
        bool anyResult = buffCollectorWork[0];
        buffCollectorWork[0] = false;
        cudaStatus = cudaMemcpyAsync(dev_buffCollectorWork, buffCollectorWork, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        while (anyResult) {
            resultCollector << <BLOCK_NUMBER, 1 >> > (dev_buffDeviceResult, dev_buffResult, THREAD_STEPS * BLOCK_THREADS);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
                goto Error;
            }            
            cudaStatus = cudaMemcpy(buffResult, dev_buffResult, COLLECTOR_SIZE * 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }
            anyResult = false;
            for (int resultIx = 0; resultIx < COLLECTOR_SIZE * 8;) {
                if (buffResult[resultIx] != 0x0 || buffResult[resultIx + 1] != 0x0 || buffResult[resultIx + 2] != 0x0 || buffResult[resultIx + 3] != 0x0 ||
                    buffResult[resultIx + 4] != 0x0 || buffResult[resultIx + 5] != 0x0 || buffResult[resultIx + 6] != 0x0 || buffResult[resultIx + 7] != 0x0) {
                    Int toTest = new Int();
                    for (int b = 0; b < 8; b++) {
                        toTest.bits[7-b] = buffResult[resultIx + b];
                    }
                    toTest.bits[8] = 0x0;
                    toTest.bits[9] = 0x0;
                    processCandidate(toTest);
                    anyResult = true;
                }
                resultIx += 8;
            }
        }
        incrementBase58(outputSize, KEY_START_IX);
        counter += outputSize;
        int64_t tHash = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountHashrate).count();
        int64_t tStatus = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginCountStatus).count();
        if (tHash > 5) {
            double speed = (double)((double)counter / tHash) / 1000000.0;
            printSpeed(speed);
            counter = 0;
            beginCountHashrate = std::chrono::steady_clock::now();
        }
        if (tStatus > fileStatusInterval) {
            saveStatus();
            beginCountStatus = std::chrono::steady_clock::now();
        }
    }
Error:
    cudaFree(dev_buffResult);
    cudaFree(dev_buffDeviceResult);
    cudaFree(dev_buffStart);
    cudaFree(dev_buffCollectorWork);
    return cudaStatus;
}

void processCandidate(Int& toTest) {
    FILE* keys;
    char rmdhash[21], address[50];
    Point publickey = secp->ComputePublicKey(&toTest);
    secp->GetHash160(P2PKH, false, publickey, (unsigned char*)rmdhash);
    addressToBase58(rmdhash, address);
    string a = address;
    if (filter.contains(a)) {
        printf("found: %s - %s\n", address, toTest.GetBase16().c_str());
        keys = fopen(fileResult.c_str(), "a+");
        fprintf(keys, "%s\n", address);
        fprintf(keys, "%s\n\n", toTest.GetBase16().c_str());
        fclose(keys);
    }
}

void prepareAlphabet() {
    string a = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (int i = 0; i < ALPHABET_LEN; i++) {
        ALPHABET[i] = a.at(i);
    }
}

bool readArgs(int argc, char** argv) {
    int a = 1;
    while (a < argc) {
        if (strcmp(argv[a], "-d") == 0) {
            a++;
            DEVICE_NR = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-t") == 0) {
            a++;
            BLOCK_THREADS = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-b") == 0) {
            a++;
            BLOCK_NUMBER = strtol(argv[a], NULL, 10);
        }
        else if (strcmp(argv[a], "-input") == 0) {
            a++;
            fileInput = string(argv[a]);
            read_file(fileInput);
        }
        else if (strcmp(argv[a], "-rangeStart") == 0) {
            a++;
            KEY_START= string(argv[a]);
            while (KEY_START.length() < KEY_LENGTH+1) {
                KEY_START.append("1");
            }
            for (int i = 1, ix=0; i < KEY_LENGTH+1; i++, ix++) {
                for (int c = 0; c < 58; c++) {
                    if (ALPHABET[c] == KEY_START.at(i)) {
                        KEY_START_IX[ix] = c;
                        break;
                    }
                }
            }
        }
        a++;
    }
    return true;
}

bool checkDevice() {
    cudaError_t cudaStatus = cudaSetDevice(DEVICE_NR);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "device %d failed!", DEVICE_NR);
        return false;
    }
    else {
        cudaDeviceProp props;
        cudaStatus = cudaGetDeviceProperties(&props, DEVICE_NR);
        printf("Using:\n");
        printf("%s (%2d procs)\n", props.name, props.multiProcessorCount);
        printf("maxThreadsPerBlock: %2d\n\n", props.maxThreadsPerBlock);
        if (BLOCK_NUMBER == 0) {
            BLOCK_NUMBER = props.multiProcessorCount * 8;
        }
        if (BLOCK_THREADS == 0) {
            BLOCK_THREADS = (props.maxThreadsPerBlock / 8) * 5;
        }
        outputSize = (uint64_t) BLOCK_NUMBER * BLOCK_THREADS * THREAD_STEPS;

    }
    return true;
}

bool read_file(const std::string& file_name) {
    std::ifstream stream(file_name.c_str());

    if (!stream)
    {
        std::cout << "Error: Failed to open file '" << file_name << "'" << std::endl;
        return false;
    }
    std::string buffer;

    while (std::getline(stream, buffer))
    {
        filter.insert(buffer);
    }
    printf("loaded: %d\n", filter.element_count());
}

void incrementBase58(int count, int* key) {    
    for (int c = 0; c < count; c++) {
        int i = 20;
        do {
            key[i] = (key[i] + 1) % 57;
        } while (key[i--] == 0 && i >= 0);
    }
}

void saveStatus() {
    FILE* stat = fopen(fileStatus.c_str(), "w");
    auto time = std::chrono::system_clock::now();
    std::time_t s_time = std::chrono::system_clock::to_time_t(time);
    fprintf(stat, "%s\n", std::ctime(&s_time));
    string key="";
    for (int i = 0; i < KEY_LENGTH; i++) {
        key+=ALPHABET[KEY_START_IX[i]];        
    }
    fprintf(stat, "-rangeStart=S%s\n", key.c_str());
    fclose(stat);
}

void printSpeed(double speed) {
    std::string speedStr;
    if (speed < 0.01) {
        speedStr = "< 0.01 MKey/s";
    }
    else {
        if (speed < 1000) {
            speedStr = formatDouble("%.3f", speed) + " MKey/s";
        }
        else {
            speed /= 1000;
            if (speed < 1000) {
                speedStr = formatDouble("%.3f", speed) + " GKey/s";
            }
            else {
                speed /= 1000;
                speedStr = formatDouble("%.3f", speed) + " TKey/s";
            }
        }
    }
    printf("\r %s   ", speedStr.c_str()); 
    fflush(stdout);
}

/*
* test : SkK5VPtmTm3mQKYaJQFRZP
* 1PzEGi7a6UEGCAXtGjZj8kBX2VEHcLMrqd - F30C1DDD12EA91BD35D5D1B83EAC611717D99DA826F207C3C3D4839E271648CB
*/