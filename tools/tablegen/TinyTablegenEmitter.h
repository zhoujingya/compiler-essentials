// Define a emitter class

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
namespace tinytblgen {
using namespace llvm;

class Emitter {
  raw_ostream &OS;
  RecordKeeper &Records;

public:
  Emitter(raw_ostream &OS, RecordKeeper &Records) : OS(OS), Records(Records) {}

  void EmitTokensAndKeywordFilter(raw_ostream &OS, RecordKeeper &Records);

};
} // namespace tinytblgen
