#include "lsh.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // Φτιάχνουμε LSH αντικείμενο: dim=3, L=5 πίνακες, k=4 hash functions, w=4.0
    nn::LSH lsh(3, 5, 4, 4.0f);

    // Υποθετικό dataset
    vector<vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f},
        {10.0f, 10.0f, 10.0f}
    };

    lsh.build_index(data);

    //υποθετικό query
    vector<float> query = {1.5f, 2.5f, 3.5f};
    auto neighbors = lsh.query(query, 3);

    cout << "Query returned " << neighbors.size() << " neighbors:" << endl;
    for (int idx : neighbors) {
        cout << " - index " << idx << ": [ ";
        for (float val : data[idx]) {
            cout << val << " ";
        }
        cout << "]" << endl;
    }

    lsh.clear_index();

    return 0;
}
