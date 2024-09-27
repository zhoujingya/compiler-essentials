#ifndef TINYLANG_BASIC_VERSION_H
#define TINYLANG_BASIC_VERSION_H

#include "Basic/Config.inc"
#include <string>

namespace tinylang {
std::string getTinylangVersion();

std::string getLLVMVersion();
} // namespace tinylang
#endif
