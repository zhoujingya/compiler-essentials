## Some important concepts

* ODS: Operation definition specification
* DRR:  Declarative Rewrite Rules

## MLIR Toy chapter 3

Optimize Transpose using C++ style pattern-match and rewrite

有两种办法可以实现模式匹配的转换
1: Imperative: 命令式, c++模式匹配与转换
2: Declarative: 声明式, 基于table-gen工具的模式匹配与转换

需要注意的是**DRR**要求提前定义好**ODS**规则

感觉有点像LLVM后端tablegen的指令匹配过程

那么练习的时候可以分为两步

>功能描述

* 完全使用C++写这个pattern:
* 使用tablegen匹配这个功能:

