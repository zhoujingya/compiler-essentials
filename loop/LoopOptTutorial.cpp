//===-------- LoopOptTutorial.cpp - Loop Opt Tutorial Pass ------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a small loop pass to be used to illustrate several
/// aspects about writing a loop optimization. It was developed as part of the
/// "Writing a Loop Optimization" tutorial, presented at LLVM Devepeloper's
/// Conference, 2019.
//===----------------------------------------------------------------------===

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "LoopOptTutorial.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "loop-opt-tutorial"

bool LoopSplit::run(Loop &L) const {

  LLVM_DEBUG(dbgs() << "Entering " << __func__ << "\n");

  LLVM_DEBUG(dbgs() << "TODO: Need to check if Loop is a valid candidate\n");

  return false;
}

PreservedAnalyses LoopOptTutorialPass::run(Loop &L, LoopAnalysisManager &LAM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &U) {
  outs() << "Entering LoopOptTutorialPass::run " << "\n";
  outs() << "Loop: " << L << "\n";

  bool Changed = LoopSplit(AR.LI).run(L);

  if (!Changed)
    return PreservedAnalyses::all();

  return llvm::getLoopPassPreservedAnalyses();
}

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getLoopOptTutorialPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopOptTutorial", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, LoopPassManager &LPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "LoopOptTutorial") {
                    LPM.addPass(LoopOptTutorialPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize HelloWorld when added to the pass pipeline on the
// command line, i.e. via '-passes=hello-world'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLoopOptTutorialPluginInfo();
}
