; RUN: opt -load-pass-plugin=%shlibdir/libMem2Reg%shlibext  -S -passes=mem2reg1 %s -o - | FileCheck %s

; CHECK: define i32 @test5(i32 %n) {
; CHECK-NEXT:   %sum = add i32 10, 20
; CHECK-NEXT:   ret i32 %sum
; CHECK-NEXT: }

define i32 @test5(i32 %n) {
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 10, i32* %x
  store i32 20, i32* %y
  %val1 = load i32, i32* %x
  %val2 = load i32, i32* %y
  %sum = add i32 %val1, %val2
  ret i32 %sum
}


