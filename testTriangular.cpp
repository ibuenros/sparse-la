#include "matrix.h"
#include "stdio.h"

int main(){

  TriangularMatrix* A = new TriangularMatrix(4,4,7,1);

/*
  // upper triangular
  A->newline(0);
  A->append(0,0,1);
  A->append(0,2,5);
  A->newline(1);
  A->append(1,1,2);
  A->append(1,3,6);
  A->newline(2);
  A->append(2,2,3);
  A->append(2,3,7);
  A->newline(3);
  A->append(3,3,4);
  printf("A:\n");
  A->finalize();
*/

  // lower triangular
  A->newline(0);
  A->append(0,0,1);
  A->newline(1);
  A->append(1,1,2);
  A->newline(2);
  A->append(2,0,5);
  A->append(2,2,3);
  A->newline(3);
  A->append(3,1,6);
  A->append(3,2,7);
  A->append(3,3,4);  
  A->finalize();

  A->display();

  Vector* b = new Vector(4,false);
  b->set(0,1);
  b->set(1,5);
  b->set(2,1);
  b->set(3,3);
  printf("b:\n");
  b->display();

  A->solveLinearSystem(b,b);
  printf("result:\n");
  b->display();

}
