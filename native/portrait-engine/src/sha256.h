#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pe {

std::string sha256_hex(const uint8_t* data, size_t size);
std::string sha256_file(const std::string& utf8_path);

}  // namespace pe
