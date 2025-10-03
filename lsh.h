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
    int L_;     //αριθμός hash tables L
    int k_;     //αριθμός hash functions ανά πίνακα k
    float w_;   //μέγεθος παραθύρου w
    unsigned int seed_;

    //πίνακες hash: key -> λίστα IDs
    vector<unordered_map<string, vector<int>>> tables_;

    //τυχαία διανύσματα v ∼ N(0,1)^d (L × k × d) 
    vector<vector<vector<float>>> v_;

    //μετατοπίσεις t ∼ U[0,w) (L × k) 
    vector<vector<float>> t_;

    //δείκτης στα δεδομένα
    const vector<vector<float>>* data_ptr_ = nullptr;

    //h(p) = floor((p·v + t) / w) για κάθε hash function 
    vector<long long> compute_hashes_for_table(const vector<float>& p, int table_idx) const;

    //βοηθητικές συναρτήσεις 
    string signature_to_string(const vector<long long>& hashes) const; 
    float euclidean_distance_sq(const vector<float>& x, const vector<float>& y) const;
};

} //namespace nn

#endif //LSH_H

