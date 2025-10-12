#include "hypercube.h"
#include <iostream>

namespace nn
{

    Hypercube::Hypercube(int, int, int, int, int, unsigned int, utils::DistanceFunc)
    {
        std::cout << "Hypercube ctor\n";
    }

    void Hypercube::build_index(const std::vector<std::vector<float>> &)
    {
        std::cout << "[Hypercube] build_index()\n";
    }

    std::vector<std::pair<int, float>> Hypercube::knn_query(const std::vector<float> &, int) const
    {
        return {};
    }

    std::vector<int> Hypercube::range_search(const std::vector<float> &, float) const
    {
        return {};
    }

    void Hypercube::clear_index()
    {
        std::cout << "[Hypercube] clear_index()\n";
    }

} /*namespace nn*/
