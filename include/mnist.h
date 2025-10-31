#ifndef MNIST_H
#define MNIST_H

#include <vector>   //std::vector
#include <string>   //std::string
#include "helper.h"  //για το utils::MNIST_NORMALIZED

//Template με προεπιλογή: Normalize = falseNormalize = false
template <bool Normalize = utils::MNIST_NORMALIZED>
//Φορτώνει εικόνες MNIST από το αρχείο filename
std::vector<std::vector<float>> load_mnist_images(const std::string& filename, int max_images = 1000);

#endif //MNIST_H
