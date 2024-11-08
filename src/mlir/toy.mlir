  Module:
    Function 
      Proto 'multiply_transpose  Module:
    Function 
      Proto 'multiply_transpose' @test/toy.mlir:2:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @test/toy.mlir:3:25
            Call 'transpose' [ @test/toy.mlir:3:10
              var: a @test/toy.mlir:3:20
' @test/toy.mlir:2:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @test/toy.mlir:3:25
            Call 'transpose' [ @test/toy.mlir:3:10
              var: a @test/toy.mlir:3:20
            ]
            Call 'transpose' [             ]
            Call 'transpose' [ @test/toy.mlir:3:25
              var: b @test/toy.mlir:3:25
              var: b @test/toy.mlir:3:35
            ]
      } // Block
    Function 
  @test/toy.mlir:3:35
            ]
      } // Block
    Function 
      Proto 'main' @test/toy.mlir:6:1
      Params: []
        Proto 'main' @test/toy.mlir:6:1
      Params: []
      Block {
        VarDecl a<2, 3>  Block {
        VarDecl a<2, 3> @test/toy.mlir:7:3
          Literal: <2, 3 @test/toy.mlir:7:3
          Literal: <2, 3>[ <3>[ >[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/toy.mlir:7:17<3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/toy.mlir:7:17
        VarDecl b<2, 3> @test/toy.mlir:8:3

        VarDecl b<2, 3> @test/toy.mlir:8:3
          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/toy.mlir:8:17
      , 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/toy.mlir:8:17
        VarDecl c<> @test/toy.mlir:9:3
          Call 'multiply_transpose' [   VarDecl c<> @test/toy.mlir:9:3
          Call 'multiply_transpose' [ @test/toy.mlir:9:11
            var: a @test/toy.mlir:9:30
@test/toy.mlir:9:11
            var: a @test/toy.mlir:9:30
            var: b @test/toy.mlir:9:33
                  var: b @test/toy.mlir:9:33
          ]
        VarDecl d<> @test/toy.mlir:10:3
    ]
        VarDecl d<> @test/toy.mlir:10:3
          Call 'multiply_transpose' [ @test/toy.mlir:10:11
                Call 'multiply_transpose' [ @test/toy.mlir:10:11
            var: b @test/toy.mlir:10:30
                  var: b @test/toy.mlir:10:30
            var: a @test/toy.mlir:10:33
          ]
      var: a @test/toy.mlir:10:33
          ]
        Print [ @test/toy.mlir:11:3
          var: d @test/toy.mlir:11:9
  Print [ @test/toy.mlir:11:3
          var: d @test/toy.mlir:11:9
        ]
      } // Block
        ]
      } // Block
