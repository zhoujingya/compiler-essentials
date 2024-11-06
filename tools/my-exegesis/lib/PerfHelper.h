#ifndef LLVM_TOOLS_MY_EXEGESIS_PERFHELPER_H
#define LLVM_TOOLS_MY_EXEGESIS_PERFHELPER_H

namespace myexegesis {
namespace pfm {

// Returns true on error.
bool pfmInitialize();
void pfmTerminate();

} // namespace pfm
} // namespace exegesis

#endif