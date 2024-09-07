; RUN: opt -load-pass-plugin=%shlibdir/libHelloWorld%shlibext  -S -passes=hello-world %s -o - | FileCheck %s

; CHECK: (llvm-essential) Hello from: test
; CHECK: (llvm-essential) number of arguments: 3
define void @test(i32 %a, i16 %b, i8 %c) {
  ret void
}
