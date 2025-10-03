#include "lsh.h"
#include <iostream>
#include <vector>

int main() {
    //φτιάχνουμε LSH αντικείμενο
    nn::LSH lsh(3, 5, 4, 4.0f);

    //υποθετικό dataset
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f}
    };

    lsh.build_index(data);

    //υποθετικό query
    std::vector<float> query = {1.5f, 2.5f, 3.5f};
    auto neighbors = lsh.query(query, 2);

    std::cout << "Query returned " << neighbors.size() << " neighbors." << std::endl;

    lsh.clear_index();

    return 0;
}
