// RUN: tiny-tblgen -gen-tokens %s | FileCheck %s

//CHECK:{
//CHECK-NEXT:  "tokens": [
//CHECK-NEXT:    {
//CHECK-NEXT:      "name": "DD"
//CHECK-NEXT:    }
//CHECK-NEXT:  ]
//CHECK-NEXT:}

class MyClass {
  int x = 3;
}
def DD : MyClass;
