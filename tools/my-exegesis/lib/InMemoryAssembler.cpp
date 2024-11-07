#include "InMemoryAssembler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cstddef>
#include <memory>

using namespace llvm;

namespace myexegesis {
static constexpr const char ModuleID[] = "ExegesisInfoTest";
static constexpr const char FunctionID[] = "foo";

// Small utility function to add named passes.
static bool addPass(PassManagerBase &PM, StringRef PassName,
                    TargetPassConfig &TPC) {
  const PassRegistry *PR = PassRegistry::getPassRegistry();
  const PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    errs() << " run-pass " << PassName << " is not registered.\n";
    return true;
  }

  if (!PI->getNormalCtor()) {
    errs() << " cannot create pass: " << PassName << PI->getPassName() << "\n";
    return true;
  }
  Pass *P = PI->getNormalCtor()();
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  PM.add(P);
  TPC.printAndVerify(Banner);

  return false;
}

// Creates a void MachineFunction with no argument.
static MachineFunction &createVoidVoidMachineFunction(StringRef FunctionID,
                                                      Module *Module,
                                                      MachineModuleInfo *MMI) {
  Type *const ReturnType = Type::getInt32Ty(Module->getContext());
  FunctionType *FunctionType = FunctionType::get(ReturnType, false);
  Function *const F = Function::Create(
      FunctionType, GlobalValue::InternalLinkage, FunctionID, Module);
  // Making sure we can create a MachineFunction out of this Function even if it
  // contains no IR.
  F->setIsMaterializable(true);
  return MMI->getOrCreateMachineFunction(*F);
}

static object::OwningBinary<object::ObjectFile>
assemble(Module *Module, std::unique_ptr<MachineModuleInfo> MMI,
         LLVMTargetMachine *TM) {
  legacy::PassManager PM;
  MCContext &Context = MMI->getContext();
  auto MMIWP = std::make_unique<MachineModuleInfoWrapperPass>(TM);

  TargetLibraryInfoImpl TLII(Triple(Module->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  TargetPassConfig *TPC = TM->createPassConfig(PM);
  PM.add(TPC);
  PM.add(MMIWP.release());
  TPC->printAndVerify("MachineFunctionGenerator::assemble");
  // Adding the following passes:
  // - machineverifier: checks that the MachineFunction is well formed.
  // - prologepilog: saves and restore callee saved registers.
  for (const char *PassName : {"machineverifier", "prologepilog"})
    if (addPass(PM, PassName, *TPC))
      report_fatal_error("Unable to add a mandatory pass");
  TPC->setInitialized();

  SmallVector<char, 4096> AsmBuffer;
  raw_svector_ostream AsmStream(AsmBuffer);
  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, AsmStream, nullptr, CodeGenFileType::ObjectFile,
                        Context))
    report_fatal_error("Cannot add AsmPrinter passes");

  PM.run(*Module);

  // Storing the generated assembly into a MemoryBuffer that owns the memory.
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(AsmStream.str());
  // Create the ObjectFile from the MemoryBuffer.
  std::unique_ptr<object::ObjectFile> Obj =
      cantFail(object::ObjectFile::createObjectFile(Buffer->getMemBufferRef()));
  // Returning both the MemoryBuffer and the ObjectFile.
  return object::OwningBinary<object::ObjectFile>(std::move(Obj),
                                                  std::move(Buffer));
}

static void fillMachineFunction(MachineFunction &MF,
                                ArrayRef<MCInst> Instructions) {
  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  const MCInstrInfo *MCII = MF.getTarget().getMCInstrInfo();
  const DebugLoc DL;
  for (const MCInst &Inst : Instructions) {
    const unsigned Opcode = Inst.getOpcode();
    const MCInstrDesc &MCID = MCII->get(Opcode);
    MachineInstrBuilder Builder = BuildMI(MBB, DL, MCID);
    for (unsigned OpIndex = 0, E = Inst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      const MCOperand &Op = Inst.getOperand(OpIndex);
      if (Op.isReg()) {
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef())
          Flags |= RegState::Define;
        Builder.addReg(Op.getReg(), Flags);
      } else if (Op.isImm()) {
        Builder.addImm(Op.getImm());
      } else {
        llvm_unreachable("Not yet implemented");
      }
    }
  }
  // Adding the Return Opcode.
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(MBB, DL, TII->get(TII->getReturnOpcode()));
}

namespace {

// Implementation of this class relies on the fact that a single object with a
// single function will be loaded into memory.
class TrackingSectionMemoryManager : public SectionMemoryManager {
public:
  explicit TrackingSectionMemoryManager(uintptr_t *CodeSize)
      : CodeSize(CodeSize) {}

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    *CodeSize = Size;
    return SectionMemoryManager::allocateCodeSection(Size, Alignment, SectionID,
                                                     SectionName);
  }

private:
  uintptr_t *const CodeSize = nullptr;
};

} // namespace

JitFunctionContext::JitFunctionContext(std::unique_ptr<LLVMTargetMachine> TheTM)
    : Context(std::make_unique<LLVMContext>()), TM(std::move(TheTM)),
      MMI(std::make_unique<MachineModuleInfo>(TM.get())),
      Module(std::make_unique<llvm::Module>(ModuleID, *Context)) {
  Module->setDataLayout(TM->createDataLayout());
  MF = &createVoidVoidMachineFunction(FunctionID, Module.get(), MMI.get());
  // We need to instruct the passes that we're done with SSA and virtual
  // registers.
  auto &Properties = MF->getProperties();
  Properties.set(MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(MachineFunctionProperties::Property::IsSSA);
  Properties.reset(MachineFunctionProperties::Property::TracksLiveness);
  // prologue/epilogue pass needs the reserved registers to be frozen, this is
  // usually done by the SelectionDAGISel pass.
  MF->getRegInfo().freezeReservedRegs(*MF);
  // Saving reserved registers for client.
  ReservedRegs = MF->getSubtarget().getRegisterInfo()->getReservedRegs(*MF);
}

JitFunction::JitFunction(JitFunctionContext &&Context,
                         ArrayRef<MCInst> Instructions)
    : FunctionContext(std::move(Context)) {
  fillMachineFunction(*FunctionContext.MF, Instructions);
  // We create the pass manager, run the passes and returns the produced
  // ObjectFile.
  object::OwningBinary<object::ObjectFile> ObjHolder =
      assemble(FunctionContext.Module.get(), std::move(FunctionContext.MMI),
               FunctionContext.TM.get());
  assert(ObjHolder.getBinary() && "cannot create object file");
  // Initializing the execution engine.
  // We need to use the JIT EngineKind to be able to add an object file.
  LLVMLinkInMCJIT();
  uintptr_t CodeSize = 0;
  std::string Error;
  ExecEngine.reset(
      EngineBuilder(std::move(FunctionContext.Module))
          .setErrorStr(&Error)
          .setMCPU(FunctionContext.TM->getTargetCPU())
          .setEngineKind(EngineKind::JIT)
          .setMCJITMemoryManager(
              std::make_unique<TrackingSectionMemoryManager>(&CodeSize))
          .create(FunctionContext.TM.release()));
  if (!ExecEngine)
    report_fatal_error(StringRef(Error));
  // Adding the generated object file containing the assembled function.
  // The ExecutionEngine makes sure the object file is copied into an
  // executable page.
  ExecEngine->addObjectFile(ObjHolder.takeBinary().first);
  // Setting function
  FunctionBytes = StringRef(reinterpret_cast<const char *>(
                                ExecEngine->getFunctionAddress(FunctionID)),
                            CodeSize);
}

} // namespace myexegesis