# -*- Python -*-

# Configuration file for the 'lit' test runner.

import platform, os

import lit.formats

# Global instance of LLVMConfig provided by lit
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
# (config is an instance of TestingConfig created when discovering tests)
config.name = "Compiler-essentials"

# testFormat: The test format to use to interpret tests.
# As per shtest.py (my formatting):
#   ShTest is a format with one file per test. This is the primary format for
#   regression tests (...)
# I couldn't find any more documentation on this, but it seems to be exactly
# what we want here.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".ll", ".c", ".cpp", ".tiny", ".mlir", ".toy", ".td"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs"]

# On Mac OS, 'clang' installed via HomeBrew (or build from sources) won't know
# where to look for standard headers (e.g. 'stdlib.h'). This is a workaround.
if platform.system() == "Darwin":
    tool_substitutions = [
        ToolSubst(
            "%clang",
            "clang",
            extra_args=[
                "-isysroot",
                # http://lists.llvm.org/pipermail/cfe-dev/2016-July/049868.html
                "`xcrun --show-sdk-path`",
                # https://github.com/Homebrew/homebrew-core/issues/52461
                "-mlinker-version=0",
            ],
        ),
    ]
else:
    tool_substitutions = [
        ToolSubst(
            "%clang",
            "clang",
        )
    ]
llvm_config.add_tool_substitutions(tool_substitutions)

# The list of tools required for testing - prepend them with the path specified
# during configuration (i.e. LT_LLVM_TOOLS_DIR/bin)
tools = ["opt", "lli", "not", "FileCheck", "clang"]
llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)
llvm_config.add_tool_substitutions(
    [
        "commandline",
        "tinylang-lexer",
        "jit",
        "tinylang-parser",
        "tiny-tblgen",
        "mlir-toy",
    ],
    config.bin_dir,
)
llvm_config.add_tool_substitutions(
    ["qemu-riscv64", "riscv64-unknown-linux-gnu-gcc"], config.rv_bin_dir
)
# The LIT variable to hold the file extension for shared libraries (this is
# platform dependent)
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
# The LIT variable to hold the location of plugins/libraries
config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))

# Add the 'utils' directory to the environment path
utils_path = os.path.join(os.path.dirname(__file__), "..", "utils")
config.environment["PATH"] = os.pathsep.join(
    (utils_path, config.environment.get("PATH", ""))
)
# Define a substitution for the utils path
config.substitutions.append(("%utils_path", utils_path))

if config.enable_small_cc:
    llvm_config.add_tool_substitutions(["smallcc"], config.bin_dir)
    config.available_features.add("smallcc")
