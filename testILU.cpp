#include "matrix.h"
#include "preconditioner.h"
#include <stdio.h>

int main(){

  SparseMatrix* A = new SparseMatrix(4,4,11);

  // lower triangular
  A->newline(0);
  A->append(0,0,1);
  A->append(0,2,3);
  A->append(0,3,4);
  A->newline(1);
  A->append(1,0,2);
  A->append(1,1,1);
  A->append(1,3,6);
  A->newline(2);
  A->append(2,2,2);
  A->append(2,3,1);
  A->newline(3);
  A->append(3,0,3);
  A->append(3,2,5);
  A->append(3,3,3);  
  A->finalize();

  A->display();

  ILUPrec* P = new ILUPrec(A);
  P->computeILU(); 

}
