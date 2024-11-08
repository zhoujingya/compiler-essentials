#include "InstructionSnippetGenerator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/MC/MCInstBuilder.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace myexegesis {
void Variable::print(raw_ostream &OS, const MCRegisterInfo *RegInfo) const {
  OS << "IsUse=" << IsUse << " IsDef=" << IsDef << " possible regs: {";
  for (const size_t Reg : PossibleRegisters) {
    if (RegInfo)
      OS << RegInfo->getName(Reg);
    else
      OS << Reg;
    OS << ",";
  }
  OS << "} ";
  if (ExplicitOperands.empty()) {
    OS << "implicit";
  } else {
    OS << "explicit ops: {";
    for (const size_t Op : ExplicitOperands)
      OS << Op << ",";
    OS << "}";
  }
  OS << "\n";
}

// Update the state of a Variable with an explicit operand.
static void updateExplicitOperandVariable(const MCRegisterInfo &RegInfo,
                                          const MCInstrDesc &InstrInfo,
                                          const size_t OpIndex,
                                          const BitVector &ReservedRegs,
                                          Variable &Var) {
  const bool IsDef = OpIndex < InstrInfo.getNumDefs();
  if (IsDef)
    Var.IsDef = true;
  if (!IsDef)
    Var.IsUse = true;
  Var.ExplicitOperands.push_back(OpIndex);
  const MCOperandInfo &OpInfo = InstrInfo.operands()[OpIndex];
  if (OpInfo.RegClass >= 0) {
    Var.IsReg = true;
    for (const MCPhysReg &Reg : RegInfo.getRegClass(OpInfo.RegClass)) {
      if (!ReservedRegs[Reg])
        Var.PossibleRegisters.insert(Reg);
    }
  }
}

static Variable &findVariableWithOperand(SmallVector<Variable, 8> &Vars,
                                         size_t OpIndex) {
  // Vars.size() is small (<10) so a linear scan is good enough.
  for (Variable &Var : Vars) {
    if (is_contained(Var.ExplicitOperands, OpIndex))
      return Var;
  }
  assert(false && "Illegal state");
  static Variable *const EmptyVariable = new Variable();
  return *EmptyVariable;
}

SmallVector<Variable, 8> getVariables(const MCRegisterInfo &RegInfo,
                                      const MCInstrDesc &InstrInfo,
                                      const BitVector &ReservedRegs) {
  SmallVector<Variable, 8> Vars;
  // For each operand, its "tied to" operand or -1.
  SmallVector<int, 10> TiedToMap;
  for (size_t I = 0, E = InstrInfo.getNumOperands(); I < E; ++I) {
    TiedToMap.push_back(InstrInfo.getOperandConstraint(I, MCOI::TIED_TO));
  }
  // Adding non tied operands.
  for (size_t I = 0, E = InstrInfo.getNumOperands(); I < E; ++I) {
    if (TiedToMap[I] >= 0)
      continue; // dropping tied ones.
    Vars.emplace_back();
    updateExplicitOperandVariable(RegInfo, InstrInfo, I, ReservedRegs,
                                  Vars.back());
  }
  // Adding tied operands to existing variables.
  for (size_t I = 0, E = InstrInfo.getNumOperands(); I < E; ++I) {
    if (TiedToMap[I] < 0)
      continue; // dropping non-tied ones.
    updateExplicitOperandVariable(RegInfo, InstrInfo, I, ReservedRegs,
                                  findVariableWithOperand(Vars, TiedToMap[I]));
  }
  // Adding implicit defs.
  for (size_t I = 0, E = InstrInfo.NumImplicitDefs; I < E; ++I) {
    Vars.emplace_back();
    Variable &Var = Vars.back();
    const MCPhysReg Reg = InstrInfo.implicit_defs()[I];
    assert(!ReservedRegs[Reg] && "implicit def of reserved register");
    Var.PossibleRegisters.insert(Reg);
    Var.IsDef = true;
    Var.IsReg = true;
  }
  // Adding implicit uses.
  for (size_t I = 0, E = InstrInfo.NumImplicitUses; I < E; ++I) {
    Vars.emplace_back();
    Variable &Var = Vars.back();
    const MCPhysReg Reg = InstrInfo.implicit_uses()[I];
    assert(!ReservedRegs[Reg] && "implicit use of reserved register");
    Var.PossibleRegisters.insert(Reg);
    Var.IsUse = true;
    Var.IsReg = true;
  }

  return Vars;
}

VariableAssignment::VariableAssignment(size_t VarIdx,
                                       llvm::MCPhysReg AssignedReg)
    : VarIdx(VarIdx), AssignedReg(AssignedReg) {}

bool VariableAssignment::operator==(const VariableAssignment &Other) const {
  return std::tie(VarIdx, AssignedReg) ==
         std::tie(Other.VarIdx, Other.AssignedReg);
}

bool VariableAssignment::operator<(const VariableAssignment &Other) const {
  return std::tie(VarIdx, AssignedReg) <
         std::tie(Other.VarIdx, Other.AssignedReg);
}

void dumpAssignmentChain(const MCRegisterInfo &RegInfo,
                         const AssignmentChain &Chain) {
  for (const VariableAssignment &Assignment : Chain) {
    outs() << format("(%d %s) ", Assignment.VarIdx,
                     RegInfo.getName(Assignment.AssignedReg));
  }
  outs() << "\n";
}

std::vector<AssignmentChain>
computeSequentialAssignmentChains(const MCRegisterInfo &RegInfo,
                                  ArrayRef<Variable> Vars) {
  using graph::Node;
  graph::Graph Graph;

  // Add register aliasing to the graph.
  setupRegisterAliasing(RegInfo, Graph);

  // Adding variables to the graph.
  for (size_t I = 0, E = Vars.size(); I < E; ++I) {
    const Variable &Var = Vars[I];
    const Node N = Node::Var(I);
    if (Var.IsDef) {
      Graph.connect(Node::In(), N);
      for (const size_t Reg : Var.PossibleRegisters)
        Graph.connect(N, Node::Reg(Reg));
    }
    if (Var.IsUse) {
      Graph.connect(N, Node::Out());
      for (const size_t Reg : Var.PossibleRegisters)
        Graph.connect(Node::Reg(Reg), N);
    }
  }

  // Find all possible dependency chains (aka all possible paths from In to Out
  // node).
  std::vector<AssignmentChain> AllChains;
  for (;;) {
    const auto Path = Graph.getPathFrom(Node::In(), Node::Out());
    if (Path.empty())
      break;
    switch (Path.size()) {
    case 0:
    case 1:
    case 2:
    case 4:
      assert(false && "Illegal state");
      break;
    case 3: { // IN -> variable -> OUT
      const size_t VarIdx = Path[1].varValue();
      for (size_t Reg : Vars[VarIdx].PossibleRegisters) {
        AllChains.emplace_back();
        AllChains.back().emplace(VarIdx, Reg);
      }
      Graph.disconnect(Path[0], Path[1]); // IN -> variable
      Graph.disconnect(Path[1], Path[2]); // variable -> OUT
      break;
    }
    default: { // IN -> var1 -> Reg[...] -> var2 -> OUT
      const size_t Last = Path.size() - 1;
      const size_t Var1 = Path[1].varValue();
      const MCPhysReg Reg1 = Path[2].regValue();
      const MCPhysReg Reg2 = Path[Last - 2].regValue();
      const size_t Var2 = Path[Last - 1].varValue();
      AllChains.emplace_back();
      AllChains.back().emplace(Var1, Reg1);
      AllChains.back().emplace(Var2, Reg2);
      Graph.disconnect(Path[1], Path[2]); // Var1 -> Reg[0]
      break;
    }
    }
  }

  return AllChains;
}

std::vector<MCPhysReg>
getRandomAssignment(ArrayRef<Variable> Vars, ArrayRef<AssignmentChain> Chains,
                    const std::function<size_t(size_t)> &RandomIndexForSize) {
  // Registers are initialized with 0 (aka NoRegister).
  std::vector<MCPhysReg> Registers(Vars.size(), 0);
  if (Chains.empty())
    return Registers;
  // Pick one of the chains and set Registers that are fully constrained (have
  // no degrees of freedom).
  const size_t ChainIndex = RandomIndexForSize(Chains.size());
  for (const VariableAssignment Assignment : Chains[ChainIndex])
    Registers[Assignment.VarIdx] = Assignment.AssignedReg;
  // Registers with remaining degrees of freedom are assigned randomly.
  for (size_t I = 0, E = Vars.size(); I < E; ++I) {
    MCPhysReg &Reg = Registers[I];
    const Variable &Var = Vars[I];
    const auto &PossibleRegisters = Var.PossibleRegisters;
    if (Reg > 0 || PossibleRegisters.empty())
      continue;
    Reg = PossibleRegisters[RandomIndexForSize(PossibleRegisters.size())];
  }
  return Registers;
}

// Finds a matching register `reg` for variable `VarIdx` and sets
// `RegAssignments[r]` to `VarIdx`. Returns false if no matching can be found.
// `seen.count(r)` is 1 if register `reg` has been processed.
static bool
findMatchingRegister(ArrayRef<Variable> Vars, const size_t VarIdx,
                     std::unordered_set<MCPhysReg> &Seen,
                     std::unordered_map<MCPhysReg, size_t> &RegAssignments) {
  for (const MCPhysReg Reg : Vars[VarIdx].PossibleRegisters) {
    if (!Seen.count(Reg)) {
      Seen.insert(Reg); // Mark `Reg` as seen.
      // If `Reg` is not assigned to a variable, or if `Reg` was assigned to a
      // variable which has an alternate possible register, assign `Reg` to
      // variable `VarIdx`. Since `Reg` is marked as assigned in the above line,
      // `RegAssignments[r]` in the following recursive call will not get
      // assigned `Reg` again.
      const auto AssignedVarIt = RegAssignments.find(Reg);
      if (AssignedVarIt == RegAssignments.end() ||
          findMatchingRegister(Vars, AssignedVarIt->second, Seen,
                               RegAssignments)) {
        RegAssignments[Reg] = VarIdx;
        return true;
      }
    }
  }
  return false;
}

// This is actually a maximum bipartite matching problem:
//   https://en.wikipedia.org/wiki/Matching_(graph_theory)#Bipartite_matching
// The graph has variables on the left and registers on the right, with an edge
// between variable `I` and register `Reg` iff
// `Vars[I].PossibleRegisters.count(A)`.
// Note that a greedy approach won't work for cases like:
//   Vars[0] PossibleRegisters={C,B}
//   Vars[1] PossibleRegisters={A,B}
//   Vars[2] PossibleRegisters={A,C}
// There is a feasible solution {0->B, 1->A, 2->C}, but the greedy solution is
// {0->C, 1->A, oops}.
std::vector<MCPhysReg> getExclusiveAssignment(ArrayRef<Variable> Vars) {
  // `RegAssignments[r]` is the variable id that was assigned register `Reg`.
  std::unordered_map<MCPhysReg, size_t> RegAssignments;

  for (size_t VarIdx = 0, E = Vars.size(); VarIdx < E; ++VarIdx) {
    if (!Vars[VarIdx].IsReg)
      continue;
    std::unordered_set<MCPhysReg> Seen;
    if (!findMatchingRegister(Vars, VarIdx, Seen, RegAssignments))
      return {}; // Infeasible.
  }

  std::vector<MCPhysReg> Registers(Vars.size(), 0);
  for (const auto &RegVarIdx : RegAssignments)
    Registers[RegVarIdx.second] = RegVarIdx.first;
  return Registers;
}

std::vector<MCPhysReg> getGreedyAssignment(ArrayRef<Variable> Vars) {
  std::vector<MCPhysReg> Registers(Vars.size(), 0);
  SmallSet<MCPhysReg, 8> Assigned;
  for (size_t VarIdx = 0, E = Vars.size(); VarIdx < E; ++VarIdx) {
    const auto &Var = Vars[VarIdx];
    if (!Var.IsReg)
      continue;
    if (Var.PossibleRegisters.empty())
      return {};
    // Try possible registers until an unassigned one is found.
    for (const auto Reg : Var.PossibleRegisters) {
      if (Assigned.insert(Reg).second) {
        Registers[VarIdx] = Reg;
        break;
      }
    }
    // Fallback to first possible register.
    if (Registers[VarIdx] == 0)
      Registers[VarIdx] = Var.PossibleRegisters[0];
  }
  return Registers;
}

MCInst generateMCInst(const MCInstrDesc &InstrInfo, ArrayRef<Variable> Vars,
                      ArrayRef<MCPhysReg> VarRegs) {
  const size_t NumOperands = InstrInfo.getNumOperands();
  SmallVector<MCPhysReg, 16> OperandToRegister(NumOperands, 0);

  // We browse the variable and for each explicit operands we set the selected
  // register in the OperandToRegister array.
  for (size_t I = 0, E = Vars.size(); I < E; ++I) {
    for (const size_t OpIndex : Vars[I].ExplicitOperands) {
      OperandToRegister[OpIndex] = VarRegs[I];
    }
  }

  // Building the instruction.
  MCInstBuilder Builder(InstrInfo.getOpcode());
  for (size_t I = 0, E = InstrInfo.getNumOperands(); I < E; ++I) {
    const MCOperandInfo &OpInfo = InstrInfo.operands()[I];
    switch (OpInfo.OperandType) {
    case MCOI::OperandType::OPERAND_REGISTER:
      Builder.addReg(OperandToRegister[I]);
      break;
    case MCOI::OperandType::OPERAND_IMMEDIATE:
      Builder.addImm(1);
      break;
    default:
      Builder.addOperand(MCOperand());
    }
  }

  return Builder;
}

} // namespace myexegesis