#ifndef MNIST_H
#define MNIST_H

#include <vector>   //std::vector
#include <string>   //std::string

//Template με προεπιλογή: Normalize = false
template <bool Normalize = false>
//Φορτώνει εικόνες MNIST από το αρχείο filename
std::vector<std::vector<float>> load_mnist_images(const std::string& filename, int max_images = 1000);

#endif //MNIST_H
