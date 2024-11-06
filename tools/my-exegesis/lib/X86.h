#ifndef LLVM_TOOLS_MY_EXEGESIS_X86_H
#define LLVM_TOOLS_MY_EXEGESIS_X86_H

#include "BenchmarkRunner.h"
#include "LlvmState.h"


namespace myexegesis {
class X86Filter : public BenchmarkRunner::InstructionFilter {
public:
    ~X86Filter() override;

    Error shouldRun(const LLVMState &State, unsigned Opcode) const override;
};
}

#endif