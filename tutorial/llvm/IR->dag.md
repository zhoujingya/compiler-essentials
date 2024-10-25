## Convert IR to selectionDAG

* Black arrows mean data flow dependency

* Red arrows mean glue dependency

* Blue dashed arrows mean chain dependency

Glue prevents the two nodes from being broken up during scheduling. Chain dependenciesprevent nodes with side effects. A data dependency indicates when an instruction dependson the result of a previous instruction.
