#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "X86.h"

namespace myexegesis {

static Error makeError(llvm::Twine Msg) {
  return make_error<llvm::StringError>(Msg, llvm::inconvertibleErrorCode());
}

X86Filter::~X86Filter() = default;

// Test whether we can generate a snippet for this instruction.
llvm::Error X86Filter::shouldRun(const LLVMState &State,
                                 const unsigned Opcode) const {
  const auto &InstrInfo = State.getInstrInfo();
  const MCInstrDesc &InstrDesc = InstrInfo.get(Opcode);
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return makeError("Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return makeError("Unsupported opcode: isCall/isReturn");
  const auto OpcodeName = InstrInfo.getName(Opcode);
  if (OpcodeName.starts_with("POPF") || OpcodeName.starts_with("PUSHF") ||
      OpcodeName.starts_with("ADJCALLSTACK")) {
    return makeError("Unsupported opcode: Push/Pop/AdjCallStack");
  }
  return llvm::ErrorSuccess();
}

} // namespace myexegesis
