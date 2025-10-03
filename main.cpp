#include "lsh.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {
    //φτιάχνουμε LSH αντικείμενο: dim=3, L=5 πίνακες, k=4 hash functions, w=4.0
    nn::LSH lsh(3, 5, 4, 4.0f);

    //υποθετικό dataset
    vector<vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f},
        {10.0f, 10.0f, 10.0f}
    };

    lsh.build_index(data);

    //υποθετικό query
    vector<float> query = {1.5f, 2.5f, 3.5f};

    //(α) Nearest Neighbor
    int nn_idx = lsh.nn_query(query);
    cout << "Nearest Neighbor:" << endl;
    cout << " - index " << nn_idx << ": [ ";
    for (float val : data[nn_idx]) cout << val << " ";
    cout << "]" << endl << endl;

    //(β) k-Nearest Neighbors
    int k = 2;
    auto knn = lsh.knn_query(query, k);
    cout << k << " Nearest Neighbors:" << endl;
    for (int idx : knn) {
        cout << " - index " << idx << ": [ ";
        for (float val : data[idx]) cout << val << " ";
        cout << "]" << endl;
    }
    cout << endl;

    //(γ) Range Search
    float R = 5.0f;
    auto range = lsh.range_search(query, R);
    cout << "Range search (R=" << R << "):" << endl;
    for (int idx : range) {
        cout << " - index " << idx << ": [ ";
        for (float val : data[idx]) cout << val << " ";
        cout << "]" << endl;
    }

    lsh.clear_index();

    return 0;
}
