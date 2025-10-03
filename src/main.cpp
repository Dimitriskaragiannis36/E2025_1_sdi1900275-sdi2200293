#include "lsh.h"
#include "mnist.h"
#include <iostream>
using namespace std;

int main() {
    //φόρτωσε MNIST (π.χ. τις πρώτες 2000 εικόνες για ταχύτητα)
    string filename = "data/train-images.idx3-ubyte";
    auto data = load_mnist_images(filename, 2000);

    //φτιάχνουμε LSH αντικείμενο: dim=784 (28x28), L=5 πίνακες, k=4 hash functions, w=4.0
    nn::LSH lsh(784, 5, 4, 4.0f);

    //χτίσιμο index
    lsh.build_index(data);

    //υποθετικό query = η εικόνα με index 0
    vector<float> query = data[0];

    //(α)Nearest Neighbor
    int nn_idx = lsh.nn_query(query);
    if (nn_idx != -1) {
        cout << "Nearest Neighbor: index " << nn_idx << endl;
    } else {
        cout << "No neighbor found." << endl;
    }

    //(β)k-Nearest Neighbors
    int k = 5;
    auto knn = lsh.knn_query(query, k);
    cout << k << " Nearest Neighbors:" << endl;
    for (int idx : knn) {
        cout << " - index " << idx << endl;
    }
    cout << endl;

    //(γ)Range Search
    float R = 5.0f;
    auto range = lsh.range_search(query, R);
    cout << "Range search (R=" << R << "): found " << range.size() << " neighbors" << endl;

    lsh.clear_index();

    return 0;
}
