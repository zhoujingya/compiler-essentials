#ifndef LLVM_TOOLS_MY_EXEGESIS_INSTRUCTIONSNIPPETGENERATOR_H
#define LLVM_TOOLS_MY_EXEGESIS_INSTRUCTIONSNIPPETGENERATOR_H

#include "OperandGraph.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Timer.h"
#include <vector>

namespace myexegesis {

// A Variable represents a set of possible values that we need to choose from.
// It may represent one or more explicit operands that are tied together, or one
// implicit operand.
class Variable final {
public:
  bool IsUse = false;
  bool IsDef = false;
  bool IsReg = false;

  // Lists all the explicit operand indices that are tied to this variable.
  // Empty if Variable represents an implicit operand.
  SmallVector<size_t, 8> ExplicitOperands;

  // - In case of explicit operands, PossibleRegisters is the expansion of the
  // operands's RegClass registers. Please note that tied together explicit
  // operands share the same RegClass.
  // - In case of implicit operands, PossibleRegisters is a singleton MCPhysReg.
  SmallSetVector<MCPhysReg, 16> PossibleRegisters;

  // If RegInfo is null, register names won't get resolved.
  void print(raw_ostream &OS, const MCRegisterInfo *RegInfo) const;
};

// Builds a model of implicit and explicit operands for InstrDesc into
// Variables.
SmallVector<Variable, 8> getVariables(const MCRegisterInfo &RegInfo,
                                      const MCInstrDesc &InstrDesc,
                                      const BitVector &ReservedRegs);

// A simple object to represent a Variable assignement.
struct VariableAssignment {
  VariableAssignment(size_t VarIdx, MCPhysReg AssignedReg);

  size_t VarIdx;
  MCPhysReg AssignedReg;

  bool operator==(const VariableAssignment &) const;
  bool operator<(const VariableAssignment &) const;
};

// An AssignmentChain is a set of assignement realizing a dependency chain.
// We inherit from std::set to leverage uniqueness of elements.
using AssignmentChain = std::set<VariableAssignment>;

// Debug function to print an assignment chain.
void dumpAssignmentChain(const MCRegisterInfo &RegInfo,
                         const AssignmentChain &Chain);

// Inserts Variables into a graph representing register aliasing and finds all
// the possible dependency chains for this instruction, i.e. all the possible
// assignement of operands that would make execution of the instruction
// sequential.
std::vector<AssignmentChain>
computeSequentialAssignmentChains(const MCRegisterInfo &RegInfo,
                                  ArrayRef<Variable> Vars);

// Selects a random configuration leading to a dependency chain.
// The result is a vector of the same size as `Vars`.
// `random_index_for_size` is a functor giving a random value in [0, arg[.
std::vector<MCPhysReg>
getRandomAssignment(ArrayRef<Variable> Vars,
                    ArrayRef<AssignmentChain> Chains,
                    const std::function<size_t(size_t)> &RandomIndexForSize);

// Finds an assignment of registers to variables such that no two variables are
// assigned the same register.
// The result is a vector of the same size as `Vars`, or `{}` if the
// assignment is not feasible.
std::vector<MCPhysReg>
getExclusiveAssignment(ArrayRef<Variable> Vars);

// Finds a greedy assignment of registers to variables. Each variable gets
// assigned the first possible register that is not already assigned to a
// previous variable. If there is no such register, the variable gets assigned
// the first possible register.
// The result is a vector of the same size as `Vars`, or `{}` if the
// assignment is not feasible.
std::vector<MCPhysReg> getGreedyAssignment(ArrayRef<Variable> Vars);

// Generates an LLVM MCInst with the previously computed variables.
// Immediate values are set to 1.
MCInst generateMCInst(const MCInstrDesc &InstrDesc,
                            ArrayRef<Variable> Vars,
                            ArrayRef<MCPhysReg> VarRegs);

} // namespace myexegesis

#endif