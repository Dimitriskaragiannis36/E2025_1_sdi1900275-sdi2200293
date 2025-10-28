#include "mnist.h" //παρέχει τη δήλωση της συνάρτησης load_mnist_images
#include <fstream>  //για std::ifstream
#include <iostream> //για std::cout
#include <stdexcept>   //για std::runtime_error
#include <cstdint>  //για uint32_t

static std::uint32_t read_uint32_big_endian(std::ifstream &f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (std::uint32_t(bytes[0]) << 24) |
           (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8)  |
           (std::uint32_t(bytes[3]));
}

template <bool Normalize>
std::vector<std::vector<float>> load_mnist_images(const std::string &filename, int max_images) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open MNIST file: " + filename);
    }

    std::uint32_t magic = read_uint32_big_endian(f);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file (wrong magic number)");
    }

    std::uint32_t num_images = read_uint32_big_endian(f);
    std::uint32_t num_rows   = read_uint32_big_endian(f);
    std::uint32_t num_cols   = read_uint32_big_endian(f);

    std::cout << "MNIST file: " << num_images << " images, "
         << num_rows << "x" << num_cols << std::endl;

    int dim = num_rows * num_cols;
    int count = (max_images < 0 || max_images > (int)num_images) ? num_images : max_images;

    std::vector<std::vector<float>> images(count, std::vector<float>(dim));

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < dim; j++) {
            unsigned char pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            if constexpr (Normalize)
                images[i][j] = static_cast<float>(pixel) / 255.0f;
            else
                images[i][j] = static_cast<float>(pixel);
        }
    }

    return images;
}

//ρητή εισαγωγή για τις δύο εκδοχές της συνάρτησης με template
template std::vector<std::vector<float>> load_mnist_images<false>(const std::string&, int);
template std::vector<std::vector<float>> load_mnist_images<true>(const std::string&, int);
