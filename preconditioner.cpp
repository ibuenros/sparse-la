#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
using namespace std;

#include "matrix.h"
#include "preconditioner.h"

Preconditioner::Preconditioner(SparseMatrix* Ain){
  A=Ain;
  alreadysetup=false;
}

JacobiPrec::JacobiPrec(SparseMatrix* Ain)
  :Preconditioner(Ain){
  
  sizeMat=Ain->sizesquare();
  if(sizeMat==-1) {printf("ILU preconditioner is only implemented for square matrices.\n"); exit(-1);}
  inverseDiagonal = new double[sizeMat];

}

JacobiPrec::~JacobiPrec(){
  delete[] inverseDiagonal;
}

ILUPrec::ILUPrec(SparseMatrix* Ain)
  :Preconditioner(Ain){

  sizeMat=Ain->sizesquare();
  if(sizeMat==-1) {printf("ILU preconditioner is only implemented for square matrices.\n"); exit(-1);}
  U = new TriangularMatrix(Ain->sizesquare(),Ain->sizesquare(),(Ain->getNonZero()+1)/2+Ain->sizesquare(),0);
  L = new TriangularMatrix(Ain->sizesquare(),Ain->sizesquare(),(Ain->getNonZero()+1)/2+Ain->sizesquare(),1);

}

ILUPrec::~ILUPrec(){
  delete U;
  delete L;
}

//==========================================================
// ILU Preconditioner
//==========================================================

void JacobiPrec::setup(){  

  double diagentry;
  for(int i=0; i<sizeMat; i++){
    diagentry = A->getEntry(i,i);
#ifndef NDEBUG
    if (diagentry == 0.0) {printf("Error: Jacobi preconditioner found a zero in the diagonal.\n"); exit(-1);}
#endif
    inverseDiagonal[i]=1/diagentry;
  }

  alreadysetup=true;
}

void JacobiPrec::applyInverse(Vector *out, Vector *in){

  if(!alreadysetup){ printf("The preconditioner has not been setup. Its inverse cannot be applied.\n"); exit(-1);}

  for(int i=0; i<sizeMat; i++)
    out->set(i,in->get(i)*inverseDiagonal[i]);
}

// Computes the ILU factorization of A
void ILUPrec::setup(){
  A->computeILU(L,U);

  printf("L:\n");
  //L->display();
  printf("U:\n");
  //U->display();
  alreadysetup=true;
}

void ILUPrec::applyInverse(Vector *out, Vector *in){

  if(!alreadysetup){ printf("The preconditioner has not been setup. Its inverse cannot be applied.\n"); exit(-1);}
  L->solveLinearSystem(out,in);
  U->solveLinearSystem(out,out);

}
