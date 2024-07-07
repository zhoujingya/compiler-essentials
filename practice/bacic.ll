@.str = private unnamed_addr constant [13 x i8] c"Hello world\0A\00", align 1
@gg = global ptr @.str, align 8

define i32 @add(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @main() {
  %3 = call i32 @add(i32 12, i32 121)
  call i32(ptr, ...) @printf(ptr @.str)
  ret i32 %3
}


declare i32 @printf(ptr noundef, ...)
