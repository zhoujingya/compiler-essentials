#ifndef LLVM_TOOLS_MY_EXEGESIS_INMEMORYASSEMBLER_H
#define LLVM_TOOLS_MY_EXEGESIS_INMEMORYASSEMBLER_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/ADT/ArrayRef.h>
#include <memory>

using namespace llvm;

namespace myexegesis {
// Consumable context for JitFunction below.
// This temporary object allows for retrieving MachineFunction properties before
// assembling it.
class JitFunctionContext {
public:
  explicit JitFunctionContext(std::unique_ptr<LLVMTargetMachine> TM);
  // Movable
  JitFunctionContext(JitFunctionContext &&) = default;
  JitFunctionContext &operator=(JitFunctionContext &&) = default;
  // Non copyable
  JitFunctionContext(const JitFunctionContext &) = delete;
  JitFunctionContext &operator=(const JitFunctionContext &) = delete;

  const llvm::BitVector &getReservedRegs() const { return ReservedRegs; }
private:
  friend class JitFunction;

  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<llvm::Module> Module;
  MachineFunction *MF = nullptr;
  BitVector ReservedRegs;
};

// Creates a void() function from a sequence of llvm::MCInst.
class JitFunction {
public:
  // Assembles Instructions into an executable function.
  JitFunction(JitFunctionContext &&Context, ArrayRef<MCInst> Instrutions);

  // Retrieves the function as an array of bytes.
  StringRef getFunctionBytes() const { return FunctionBytes; }

  // Retrieves the callable function.
  void operator()() const { ((void (*)())FunctionBytes.data())(); }

private:
  JitFunctionContext FunctionContext;
  std::unique_ptr<ExecutionEngine> ExecEngine;
  StringRef FunctionBytes;
};

} // namespace myexegesis

#endif