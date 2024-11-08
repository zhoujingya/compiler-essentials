#include "StandalonePasses.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "StandaloneDialect.h"
#include "StandaloneOps.h"
#include "StandalonePasses.h"
#include "llvm/ADT/Sequence.h"
#include "StandaloneDialect.h"


struct MyAttribute {
  static mlir::StringAttr get(mlir::MLIRContext *context) {
    return mlir::StringAttr::get(context, "my_string_value");
  }
};

namespace {
class SimpleAttrPass : public mlir::PassWrapper<SimpleAttrPass, mlir::OperationPass<>> {
  void runOnOperation() override;
};
} // namespace

void SimpleAttrPass::runOnOperation() {
  auto ops = getOperation();
  auto &context = getContext();
  ops->walk([&](mlir::standalone::WorldOp WorldOp) {
    // Create the custom attribute.
    mlir::StringAttr attr = MyAttribute::get(&context);

    // Set the custom attribute on the `scf.for` operation.
    WorldOp->setAttr("my_attribute", attr);
    WorldOp->dump();
  });
}

std::unique_ptr<mlir::Pass> standalone::createSimpleAttr() {
  return std::make_unique<SimpleAttrPass>();
}
