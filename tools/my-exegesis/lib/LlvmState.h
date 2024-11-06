//===-- LlvmState.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_MY_EXEGESIS_LLVMSTATE_H
#define LLVM_TOOLS_MY_EXEGESIS_LLVMSTATE_H

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <string>

using namespace llvm;
namespace myexegesis {

// An object to initialize LLVM and prepare objects needed to run the
// measurements.
class LLVMState {
public:
  LLVMState();

  Triple getTriple() const { return TheTriple; }
  StringRef getTripleString() const { return TheTriple.getTriple(); }
  StringRef getCpuName() const { return CpuName; }
  StringRef getFeatures() const { return Features; }

  const MCInstrInfo &getInstrInfo() const { return *InstrInfo; }

  const MCRegisterInfo &getRegInfo() const { return *RegInfo; }

  const MCSubtargetInfo &getSubtargetInfo() const {
    return *SubtargetInfo;
  }

  std::unique_ptr<LLVMTargetMachine> createTargetMachine() const;

  bool canAssemble(const MCInst &mc_inst) const;

private:
  Triple TheTriple;
  StringRef CpuName;
  StringRef Features;
  const Target *TheTarget = nullptr;
  std::unique_ptr<const MCSubtargetInfo> SubtargetInfo;
  std::unique_ptr<const MCInstrInfo> InstrInfo;
  std::unique_ptr<const MCRegisterInfo> RegInfo;
  std::unique_ptr<const MCAsmInfo> AsmInfo;
};

} // namespace myexegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
