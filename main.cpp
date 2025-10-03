#include "lsh.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
    //φτιάχνουμε LSH αντικείμενο
    nn::LSH lsh(3, 5, 4, 4.0f);

    //υποθετικό dataset
    vector<vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f}
    };

    lsh.build_index(data);

    //υποθετικό query
    vector<float> query = {1.5f, 2.5f, 3.5f};
    auto neighbors = lsh.query(query, 2);

    cout << "Query returned " << neighbors.size() << " neighbors." << endl;

    lsh.clear_index();

    return 0;
}
