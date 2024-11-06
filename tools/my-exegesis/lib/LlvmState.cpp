#include "LlvmState.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

namespace myexegesis {
LLVMState::LLVMState()
    : TheTriple(sys::getProcessTriple()), CpuName(sys::getHostCPUName().str()) {
  std::string Error;
  TheTarget = TargetRegistry::lookupTarget(getTripleString(), Error);
  assert(TheTarget && "unknown target for host");
  SubtargetInfo.reset(
      TheTarget->createMCSubtargetInfo(getTripleString(), CpuName, Features));
  InstrInfo.reset(TheTarget->createMCInstrInfo());
  RegInfo.reset(TheTarget->createMCRegInfo(getTripleString()));
  MCTargetOptions MCOptions;
  AsmInfo.reset(
      TheTarget->createMCAsmInfo(*RegInfo, getTripleString(), MCOptions));
}

std::unique_ptr<LLVMTargetMachine> LLVMState::createTargetMachine() const {
  const TargetOptions Options;
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      TheTarget->createTargetMachine(getTripleString(), CpuName, Features,
                                     Options, Reloc::Model::Static)));
}

bool LLVMState::canAssemble(const MCInst &Inst) const {
  MCObjectFileInfo ObjectFileInfo;
  SourceMgr SMgr;
  MCTargetOptions MCOptions;
  MCContext Context(getTriple(), AsmInfo.get(), RegInfo.get(),
                    SubtargetInfo.get(), &SMgr, &MCOptions, &ObjectFileInfo);
  std::unique_ptr<const MCCodeEmitter> CodeEmitter(
      TheTarget->createMCCodeEmitter(*InstrInfo, Context));
  SmallVector<char, 16> Tmp;
  SmallVector<MCFixup, 4> Fixups;
  CodeEmitter->encodeInstruction(Inst, Tmp, Fixups, *SubtargetInfo);
  return Tmp.size() > 0;
}
} // namespace myexegesis