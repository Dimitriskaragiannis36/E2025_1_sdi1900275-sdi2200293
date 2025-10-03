#include "mnist.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

using namespace std;

static uint32_t read_uint32_big_endian(ifstream &f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  |
           (uint32_t(bytes[3]));
}

vector<vector<float>> load_mnist_images(const string &filename, int max_images) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Could not open MNIST file: " + filename);
    }

    uint32_t magic = read_uint32_big_endian(f);
    if (magic != 2051) {
        throw runtime_error("Invalid MNIST image file (wrong magic number)");
    }

    uint32_t num_images = read_uint32_big_endian(f);
    uint32_t num_rows   = read_uint32_big_endian(f);
    uint32_t num_cols   = read_uint32_big_endian(f);

    cout << "MNIST file: " << num_images << " images, "
         << num_rows << "x" << num_cols << endl;

    int dim = num_rows * num_cols;
    int count = (max_images > 0 && max_images < (int)num_images) ? max_images : num_images;

    vector<vector<float>> images(count, vector<float>(dim));

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < dim; j++) {
            unsigned char pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f; //normalize σε [0,1]
        }
    }

    return images;
}
