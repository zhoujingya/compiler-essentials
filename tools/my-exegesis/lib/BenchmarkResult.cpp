#include "BenchmarkResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <cstring>

namespace llvm {
namespace yaml {
// std::vector<exegesis::Measure> will be rendered as a list.
template <> struct SequenceElementTraits<myexegesis::BenchmarkMeasure> {
  static const bool flow = false;
};

// exegesis::Measure is rendererd as a flow instead of a list.
// e.g. { "key": "the key", "value": 0123 }
template <> struct MappingTraits<myexegesis::BenchmarkMeasure> {
  static void mapping(IO &Io, myexegesis::BenchmarkMeasure &Obj) {
    Io.mapRequired("key", Obj.Key);
    Io.mapRequired("value", Obj.Value);
    Io.mapOptional("debug_string", Obj.DebugString);
  }
  static const bool flow = true;
};

template <> struct MappingTraits<myexegesis::AsmTemplate> {
  static void mapping(IO &Io, myexegesis::AsmTemplate &Obj) {
    Io.mapRequired("name", Obj.Name);
  }
};

template <> struct MappingTraits<myexegesis::InstructionBenchmark> {
  static void mapping(IO &Io, myexegesis::InstructionBenchmark &Obj) {
    Io.mapRequired("asm_template", Obj.AsmTmpl);
    Io.mapRequired("cpu_name", Obj.CpuName);
    Io.mapRequired("llvm_triple", Obj.LLVMTriple);
    Io.mapRequired("num_repetitions", Obj.NumRepetitions);
    Io.mapRequired("measurements", Obj.Measurements);
    Io.mapRequired("error", Obj.Error);
  }
};
} // namespace yaml
} // namespace llvm

using namespace llvm;

namespace myexegesis {
InstructionBenchmark InstructionBenchmark::readYamlOrDie(StringRef FileName) {
  std::unique_ptr<MemoryBuffer> MemBuffer =
      cantFail(errorOrToExpected(MemoryBuffer::getFile(FileName)));
  yaml::Input Yin(*MemBuffer);
  InstructionBenchmark Benchmark;
  Yin >> Benchmark;
  return Benchmark;
}

void InstructionBenchmark::writeYamlOrDie(const llvm::StringRef Filename) {
  if (Filename == "-") {
    yaml::Output Yout(outs());
    Yout << *this;
  } else {
    SmallString<1024> Buffer;
    raw_svector_ostream Ostr(Buffer);
    llvm::yaml::Output Yout(Ostr);
    Yout << *this;
    std::unique_ptr<FileOutputBuffer> File =
        cantFail(FileOutputBuffer::create(Filename, Buffer.size()));
    memcpy(File->getBufferStart(), Buffer.data(), Buffer.size());
    cantFail(File->commit());
  }
}

} // namespace myexegesis
