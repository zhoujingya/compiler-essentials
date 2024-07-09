# llvm-essential

Learning llvm essentials


## mem2reg

```llvm
define i32 @main() #0 {
entry:
%retval = alloca i32, align 4
%a = alloca i32, align 4
%b = alloca i32, align 4
store i32 0, ptr %retval
store i32 5, ptr %a, align 4
store i32 3, ptr %b, align 4
%0 = load i32, ptr %a, align 4
%1 = load i32, ptr %b, align 4
%sub = sub nsw i32 %0, %1
ret i32 %sub
}

```

`opt --passes=mem2reg -S -o mem2reg.ll mem2reg.ll`

Refer to this : `llvm/lib/Transforms/Utils/PromoteMemoryToRegister.cpp`
