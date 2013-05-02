#include<cstdlib>
#include<stdio.h>
#include<cmath>

#include "solver.h"

// Solve this*x=b by conjugate gradient, assuming this is positive definite
double CGSolver::solve(Vector* b, Vector* x){
  int sizex=A->sizesquare();
  int sizey=sizex;
  if (sizex==-1) {printf("Matrix is not square. Cannot solve.\n"); exit(-1);}
  Vector* r=new Vector(sizex,false);
  Vector* d=new Vector(sizex,false);
  Vector* ek=new Vector(sizey,false);
  Vector* z;
  if(P) z=new Vector(sizex,false);
  double rho, rhoprev, beta,alpha, res, res0;
  int k=0;
  double tol=1e-12;

#ifdef USEOPENMP
  printf("Performing CG using MPI with up to %i threads available in %i processors\n",omp_get_max_threads(),omp_get_num_procs());
#endif


  A->daxpy(x,b,r,-1,-1);
  if(P){
    P->applyInverse(z,r);
    rho = r->dot(z);
    res = r->dot(r);
  } else{
    rho=r->dot(r);
    res=rho;
  }
  res0=res;

  //double time=MPI_Wtime();
  while(sqrt(res)>tol*sqrt(res0)){
    printf("CG iteration: Res=%e, target=%e\n",sqrt(res),tol*sqrt(res0));
    k++;
    if(k==1) {
      if(P) d->copy(z);
      else d->copy(r);
    }
    else{
      beta=rho/rhoprev;
      if(P) d->capdy(z,beta,1);
      else d->capdy(r,beta,1);
    }
    A->daxpy(d,NULL,ek,1,1);
    alpha=rho/ek->dot(d);
    x->capdy(d,1,alpha);
    r->capdy(ek,1,-1*alpha);
    rhoprev=rho;
    if(P){
      P->applyInverse(z,r);
      rho = r->dot(z);
      res = r->dot(r);
    } else {
      rho=r->dot(r);
      res=rho;
    }
  }
  printf("CG Converged in %i iterations.\n",k);
  //printf("CG time: %f\n",MPI_Wtime()-time);
  
  return(sqrt(rho));
}
