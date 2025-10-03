#ifndef LSH_H
#define LSH_H

#include <vector>
#include <string>
#include <unordered_map>
using namespace std;

namespace nn {

class LSH {
public:
    //κονστράκτορας με βασικές παραμέτρους
    LSH(int dim, int L = 5, int k = 4, float w = 4.0f, unsigned int seed = 12345);

    //χτίσιμο index πάνω σε dataset
    void build_index(const vector<vector<float>>& data);

    //query: επιστροφή approximate nearest neighbors
    vector<int> query(const vector<float>& q, int num_neighbors = 10) const;

    //εκκαθάριση index
    void clear_index();

private:
    int dim_;   //διάσταση
    int L_;     //αριθμός hash tables
    int k_;     //αριθμός hash functions ανά πίνακα
    float w_;   //μέγεθος παραθύρου
    unsigned int seed_;

    //πίνακες hash: key -> λίστα IDs
    vector<unordered_map<string, vector<int>>> tables_;

    //δείκτης στα δεδομένα
    const vector<vector<float>>* data_ptr_ = nullptr;
};

} //namespace nn

#endif //LSH_H

