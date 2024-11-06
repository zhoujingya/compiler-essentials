#ifndef LLVM_TOOLS_MY_EXEGESIS_BENCHMARKRESULT_H
#define LLVM_TOOLS_MY_EXEGESIS_BENCHMARKRESULT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstddef>
#include <string>
#include <vector>

using namespace llvm;

namespace myexegesis {
struct AsmTemplate {
  std::string Name;
};

struct BenchmarkMeasure {
  std::string Key;
  double Value;
  std::string DebugString;
};

struct InstructionBenchmark {
  AsmTemplate AsmTmpl;
  std::string CpuName;
  std::string LLVMTriple;
  size_t NumRepetitions = 0;
  std::vector<BenchmarkMeasure> Measurements;
  std::string Error;

  static InstructionBenchmark readYamlOrDie(StringRef FileName);

  // Unfortunately this function is non const because of YAML traits.
  void writeYamlOrDie(const llvm::StringRef Filename);
};
} // namespace myexegesis

#endif