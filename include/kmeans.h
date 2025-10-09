#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cstddef>
#include <random>

namespace clustering {

//K-Means clustering
class KMeans {
public:
    enum class InitMethod { RANDOM, KMEANS_PLUS_PLUS };

    //κονστράκτορας
    KMeans(int k, int max_iters = 300, float tol = 1e-4f,
    InitMethod init = InitMethod::KMEANS_PLUS_PLUS,
    unsigned int seed = 1, bool verbose = false);

    //ταιριάζει το μοντέλο στα δεδομένα, επιστρέφει τον αριθμό των εκτελεσμένων επαναλήψεων
    int fit(const std::vector<std::vector<float>>& data);


    //προβλέπει την ετικέτα για ένα νέο σημείο
    int predict(const std::vector<float>& point) const;


    //προβλέπει τις ετικέτες για ένα σύνολο σημείων
    const std::vector<int>& labels() const { return labels_; }


    //επιστρέφει τα κεντροειδή
    const std::vector<std::vector<float>>& centroids() const { return centroids_; }


    //καθαρίζει το μοντέλο
    void clear();


    //getters
    int k() const { return k_; }
    int max_iters() const { return max_iters_; }
    float tol() const { return tol_; }


private:
    int k_; //αριθμός κλάσεων
    int max_iters_; //μέγιστος αριθμός επαναλήψεων
    float tol_; //κατώφλι σύγκλισης
    InitMethod init_method_; //μέθοδος αρχικοποίησης
    unsigned int seed_; //σπόρος για τυχαίους αριθμούς
    bool verbose_; //αν θα τυπώνει πληροφορίες


    std::vector<std::vector<float>> centroids_; // k x d
    std::vector<int> labels_; // n


    //τετραγωνική απόσταση μεταξύ δύο σημείων
    float squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    //τυχαίας αρχικοποίησης
    void init_random(const std::vector<std::vector<float>>& data, std::mt19937& rng);
    
    //k-means++
    void init_kmeans_pp(const std::vector<std::vector<float>>& data, std::mt19937& rng);
};


} //namespace clustering
#endif //KMEANS_H