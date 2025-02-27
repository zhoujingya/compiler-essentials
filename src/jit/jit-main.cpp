#include "jit.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

static cl::opt<std::string>
    InputFile(cl::Positional, cl::Required,
              cl::desc("<input-file>"));

std::unique_ptr<Module>
loadModule(StringRef Filename, LLVMContext &Ctx,
           const char *ProgName) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod =
      parseIRFile(Filename, Err, Ctx);
  if (!Mod.get()) {
    Err.print(ProgName, errs());
    exit(-1);
  }
  return Mod;
}

Error jitmain(std::unique_ptr<Module> M,
              std::unique_ptr<LLVMContext> Ctx,
              int argc, char *argv[]) {
  auto JIT = JIT::create();
  if (!JIT)
    return JIT.takeError();

  if (auto Err = (*JIT)->addIRModule(
          orc::ThreadSafeModule(std::move(M),
                                std::move(Ctx))))
    return Err;

  auto MainSym = (*JIT)->lookup("main");
  if (!MainSym)
    return MainSym.takeError();

  llvm::orc::ExecutorAddr MainExecutorAddr = MainSym->getAddress();
  auto *Main = MainExecutorAddr.toPtr<int(int, char**)>();

  (void)Main(argc, argv);
  return Error::success();
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  cl::ParseCommandLineOptions(argc, argv,
                              "jitty\n");

  auto Ctx = std::make_unique<LLVMContext>();

  std::unique_ptr<Module> M =
      loadModule(InputFile, *Ctx, argv[0]);
  // Setup banner: error message will be prefixed with the program name
  ExitOnError ExitOnErr(std::string(argv[0]) +
                        ": ");
  ExitOnErr(jitmain(std::move(M), std::move(Ctx),
                    argc, argv));

  return 0;
}
