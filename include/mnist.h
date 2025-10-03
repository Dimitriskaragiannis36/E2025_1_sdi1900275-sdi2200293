#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <string>

using namespace std;

vector<vector<float>> load_mnist_images(const string &filename, int max_images = 1000);

#endif //MNIST_H
