#include "sift.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <limits>

/*έλεγχος endianness συστήματος (true αν little-endian)*/
static bool is_little_endian()
{
  const uint16_t x = 1;
  return *reinterpret_cast<const uint8_t *>(&x) == 1;
}

/*αντιστροφή byte-order για 32-bit*/
static inline uint32_t bswap32(uint32_t v)
{
  return ((v & 0x000000FFu) << 24) |
         ((v & 0x0000FF00u) << 8) |
         ((v & 0x00FF0000u) >> 8) |
         ((v & 0xFF000000u) >> 24);
}

namespace sift
{

  std::vector<std::vector<float>> load_sift_dat(const std::string &filename,
                                                int max_vectors,
                                                int expected_dim)
  {
    std::ifstream f(filename, std::ios::binary);
    if (!f)
      throw std::runtime_error("Cannot open SIFT file: " + filename);

    std::vector<std::vector<float>> data;
    data.reserve(10000); /*συντηρητική αρχικοποίηση, θα μεγαλώσει δυναμικά*/

    bool sys_le = is_little_endian();

    while (true)
    {
      /*1) διάσταση (int32 little-endian)*/
      int32_t dim_raw = 0;
      f.read(reinterpret_cast<char *>(&dim_raw), 4);
      if (!f)
        break; /*EOF ok*/

      if (!sys_le)
      {
        uint32_t u = bswap32(static_cast<uint32_t>(dim_raw));
        dim_raw = static_cast<int32_t>(u);
      }

      if (dim_raw != expected_dim)
      {
        throw std::runtime_error("SIFT record with unexpected dimension: " + std::to_string(dim_raw) +
                                 " (expected " + std::to_string(expected_dim) + ")");
      }

      /*2) 128 floats (little-endian)*/
      std::vector<float> v(expected_dim);
      f.read(reinterpret_cast<char *>(v.data()), sizeof(float) * expected_dim);
      if (!f)
        throw std::runtime_error("Truncated SIFT vector payload in file: " + filename);

      if (!sys_le)
      {
        /*αν το σύστημα είναι big-endian, αντιστρέφουμε κάθε float*/
        for (int i = 0; i < expected_dim; ++i)
        {
          uint32_t *pi = reinterpret_cast<uint32_t *>(&v[i]);
          *pi = bswap32(*pi);
        }
      }

      data.emplace_back(std::move(v));

      if (max_vectors > 0 && static_cast<int>(data.size()) >= max_vectors)
        break;
    }

    if (data.empty())
      std::cerr << "Warning: no vectors read from " << filename << "\n";

    return data;
  }

} /*namespace sift*/
