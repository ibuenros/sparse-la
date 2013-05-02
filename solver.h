#ifndef _LINSOLVER_
#define _LINSOLVER_

#include "matrix.h"
#include "vector.h"
#include "preconditioner.h"

class LinearSolver{
 protected:
  SparseMatrix* A;
  Preconditioner* P;

 public:
  LinearSolver(SparseMatrix* Ain, Preconditioner* Pin){ A=Ain; P=Pin;}
  virtual double solve(Vector* in, Vector* out)=0;
};

class CGSolver: public LinearSolver{
 
 public:
  CGSolver(SparseMatrix* Ain, Preconditioner* Pin):LinearSolver(Ain,Pin){}
  double solve(Vector* in, Vector* out);

};





#endif
