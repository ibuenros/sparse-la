#ifndef _PRECONDITIONER_
#define _PRECONDITIONER_

#include "matrix.h"

class Preconditioner{
 protected:
  SparseMatrix* A;
  int* AI;
  int* AJ;
  double* Aval;
  bool alreadysetup;

 public:
  Preconditioner(SparseMatrix* Ain); 
  virtual void setup()=0;
  virtual void applyInverse(Vector* out, Vector* in)=0; 

};

class JacobiPrec: public Preconditioner{

 protected:
  double* inverseDiagonal;
  int sizeMat;

 public:
  JacobiPrec(SparseMatrix* Ain);
  ~JacobiPrec();
  void setup();
  void applyInverse(Vector* out, Vector* in);   

};

class ILUPrec: public Preconditioner{

 protected:
  TriangularMatrix* L;
  TriangularMatrix* U;
  int sizeMat;

 public:
  ILUPrec(SparseMatrix* Ain);
  ~ILUPrec();
  void setup();
  void applyInverse(Vector* out, Vector* in);

};

#endif
