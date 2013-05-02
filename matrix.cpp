#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#ifdef USEMPI
#include <mpi.h>
#endif
#ifdef USEOPENMP
#include <omp.h>
#endif
#ifndef NDEBUG
//#include <google/profiler.h>
#endif
using namespace std;

#include "matrix.h"

// SparseMatrix constructor
SparseMatrix::SparseMatrix(int sizeyin, int sizexin, int nnzin){

  nnz=nnzin;
  val=new (nothrow) double[nnz];
  J=new (nothrow) int[nnz];
  I=new (nothrow) int[sizeyin+1];
  if (val==NULL || J==NULL || I==NULL){ printf("Could not allocate sparse matrix.\n"); exit(-1); }

#ifndef NDEBUG
  printf("I'm debugging!\n");
#endif

  //#ifndef NDEBUG
  sizex=sizexin;
  sizey=sizeyin;
  finalized=false;
  lastrow=-1;
  lastcolumn=-1;
  lastwritten=-1;
  nnzmax=nnz;
  //#endif

  return;
}

SparseMatrix::~SparseMatrix(){

  delete[] val;
  delete[] J;
  delete[] I;

  if(ilu) delete[] ilu;

  return;
}

// Triangular Matrix constructor
TriangularMatrix::TriangularMatrix(int sizeyin, int sizexin, int nnz, int typein)
  :SparseMatrix(sizeyin,sizexin,nnz), type(typein) {
#ifndef NDEBUG
  if (type!=0 && type !=1) {printf("Invalid triangular matrix type: %i.\n",type); exit(-1);}
#endif
}

TriangularMatrix::~TriangularMatrix(){}

//========================================================================
// Linear Algebra Functions
//========================================================================

// append an entry to the matrix
void SparseMatrix::append(int y, int x, double value){
#ifndef NDEBUG
  if (x<=lastcolumn){ 
    printf("Attempted to append entry (%i,%i) after entry (%i,%i).\n",y,x,lastrow,lastcolumn); 
    exit(-1);
  }
  if (y!=lastrow){ printf("Attempted to write line %i, when current line was %i.\n",y,lastrow); exit(-1);}
  if (y>=sizey || y<0 || x>=sizex || x<0){
    printf("Attempted to write entry (%i,%i) of a %i-by-%i matrix.\n",y+1,x+1,sizey,sizex); 
    exit(-1);
  }
  lastcolumn=x;
  if (lastwritten>=nnzmax){
    printf("Attempted to write %i-th entry. Only %i entries allocated.\n",lastwritten+1,nnzmax);
    exit(-1);
  }
  if (finalized){printf("Matrix finalized, cannot append any new entries.\n"); exit(-1);}
#endif

  val[++lastwritten]=value;
  J[lastwritten]=x;
  return;
}

// Start row lineno in the matrix
void SparseMatrix::newline(int lineno){
#ifndef NDEBUG
  if(lineno<=lastrow){printf("Attempted to start row %i, when last row written was %i. lineno must be larger than last row.\n",lineno,lastrow); exit(-1);}
  lastcolumn=-1;
  if (finalized){ printf("Matrix finalized, cannot start new lines.\n"); exit(-1);}
#endif
  for(int i=lastrow; i<lineno; i++){
    I[i+1]=lastwritten+1;
  }
  lastrow=lineno;
  return;
}

// Fill up I with all the rows
void SparseMatrix::finalize(){

#ifndef NDEBUG
  //printf("Finalizing matrix\n");
  finalized=true;
#endif

  for(int i=lastrow; i<sizey; i++){
    I[i+1]=lastwritten+1;
  }
  return;
}

// Add a scalar to an entry. Returns a non-negative if success, -1 if the entry didn't exist
int SparseMatrix::addToEntry(int y, int x, double value){
  int startrow = I[y];
  int endrow = I[y+1];
  int xindex = -1;
  for (int i=startrow; i<endrow; i++){
    if (J[i]==x) {xindex = i; break;}
  }
  if (xindex > -1) val[xindex] += value;
  return xindex;
}

// get entry if it exists
double SparseMatrix::getEntry(int y, int x){
  int startrow = I[y];
  int endrow = I[y+1];
  int xindex = -1;
  for (int i=startrow; i<endrow; i++){
    if (J[i]==x) {xindex = i; break;}
  }
  if (xindex > -1) return(val[xindex]);
  else return 0;

}

// Multiply this by vecin and store in vecout
void SparseMatrix::vectorMult(Vector* vecin, Vector* vecout){

#ifndef NDEBUG
  if(!finalized){printf("Matrix not finalized, cannot multiply by a vector.\n"); exit(-1);}
#endif
  
  double rowresult;
  int j;
#ifdef USEOPENMP
#pragma omp parallel for private(rowresult,j)
#endif
  for (int i=0; i<sizey; i++){
    rowresult=0;
    for (j=I[i]; j<I[i+1]; j++){
      rowresult += val[j]*vecin->get(J[j]);
    }
    vecout->set(i,rowresult);
  }

  return;
}

// Perform Ax+y and store in vecout. If y is NULL, y=0. op=1 is add, op=-1 is subtract, then multiply by the scalar coeff
void SparseMatrix::daxpy(Vector* x, Vector* y, Vector* vecout, short op, double coeff){

#ifndef NDEBUG
  if(!finalized){printf("Matrix not finalized, cannot multiply by a vector.\n"); exit(-1);}
#endif

  double* initaddress;
  if(y!=NULL) initaddress=y->getInitAddress();
  double* outaddress=vecout->getInitAddress();
  double* xaddress=x->getInitAddress();
  double rowresult;
  int j;
#ifdef USEOPENMP
#pragma omp parallel for private(rowresult,j)
#endif
  for (int i=0; i<sizey; i++){
    rowresult=0;
    for (j=I[i]; j<I[i+1]; j++){
#ifdef NDEBUG
      rowresult += val[j]*xaddress[J[j]];
#else
      rowresult += val[j]*x->get(J[j]);
#endif
    }
#ifndef NDEBUG
    if (y==NULL) vecout->set(i,coeff*rowresult);
    else if (op==1) vecout->set(i,coeff*(rowresult+y->get(i)));
    else if (op==-1) vecout->set(i,coeff*(rowresult-y->get(i)));
    else {printf("Operation not recognnized in daxpy.\n"); exit(-1); }
#else
    if (y==NULL) outaddress[i]=coeff*rowresult;
    else if (op==1) outaddress[i]=coeff*(rowresult+initaddress[i]);
    else if (op==-1) outaddress[i]=coeff*(rowresult-initaddress[i]);
    else {printf("Operation not recognnized in daxpy.\n"); exit(-1); }
#endif
  }

  return;
}

// Perform Ax+y and store in vecout. If y is NULL, y=0. op=1 is add, op=-1 is subtract, then multiply by the scalar coeff
// Allows to give an offset for y, i.e. start from entry offset
void SparseMatrix::daxpyOffset(Vector* x, Vector* y, Vector* vecout, short op, double coeff, int offset){

#ifndef NDEBUG
  if(!finalized){printf("Matrix not finalized, cannot multiply by a vector.\n"); exit(-1);}
#endif
  
  double* initaddress;
  if(y!=NULL) initaddress=y->getInitAddress();
  double* outaddress=vecout->getInitAddress();
  double* xaddress=x->getInitAddress();
  double rowresult;
  int j;
#ifdef USEOPENMP
#pragma omp parallel for private(rowresult,j)
#endif
  for (int i=0; i<sizey; i++){
    rowresult=0;
    for (j=I[i]; j<I[i+1]; j++){
      rowresult += val[j]*xaddress[J[j]];
      //rowresult += val[j]*x->get(J[j]);
    }
    //if (y==NULL) vecout->set(i,coeff*rowresult);
    //else if (op==1) vecout->set(i,coeff*(rowresult+y->get(i+offset)));
    //else if (op==-1) vecout->set(i,coeff*(rowresult-y->get(i+offset)));
    //else {printf("Operation not recognnized in daxpy.\n"); exit(-1); }
    //printf("get: %f, direct: %f\n",coeff*(rowresult-y->get(i+offset)),coeff*(rowresult-initaddress[i+offset]));
    if (y==NULL) outaddress[i]=coeff*rowresult;
    else if (op==1) outaddress[i]=coeff*(rowresult+initaddress[i+offset]);
    else if (op==-1) outaddress[i]=coeff*(rowresult-initaddress[i+offset]);
    else {printf("Operation not recognnized in daxpy.\n"); exit(-1); }
    //printf("%f\n",outaddress[i]);
  }

  return;
}

// Compute ILU factorization
void SparseMatrix::computeILU(TriangularMatrix* L, TriangularMatrix* U){
  
  int startrow;
  int endrow;
  int innerstartrow, innerendrow;
  int k,j, diagindex, innerindex;
  int *diag = new int[sizex];

  ilu = new double[nnz];
  memcpy(ilu,val,sizeof(double)*nnz);
//printf("ilu=\n");
//for(int i=0; i<nnz; i++) printf("%f ",ilu[i]);
//printf("\n");
  for (int i=0; i<sizey; i++){
    startrow=I[i];
    endrow=I[i+1];
    diag[i]=-1;
    for (int nnzk=startrow; nnzk<endrow; nnzk++){ // Find the diagonal term in this row
      if(J[nnzk]==i) {diag[i]=nnzk; break;}
    }
//printf("Row %i, diagindex %i, value %f\n",i,diagindex,ilu[diagindex]);
    if (diag[i]==-1) {printf("ILU: Could not find diagonal entry at row %i. Matrix could be singular.\n",i); exit(-1);}
    for (int nnzk=startrow; nnzk<endrow; nnzk++){
      k=J[nnzk];
      innerstartrow=I[k];
      innerendrow=I[k+1];
      if(k >= i) break;
      ilu[nnzk] /= ilu[diag[k]];
      for (int nnzj=nnzk+1; nnzj<endrow; nnzj++){
        j=J[nnzj];
        innerindex=-1;
        for (int searchidx=innerstartrow; searchidx<innerendrow; searchidx++){
          if(J[searchidx]==j) {innerindex=searchidx; break;}
        }
        if (innerindex >= 0) ilu[nnzj] -= ilu[nnzk]*ilu[innerindex];
      }
    }
  }

  int col;
  if(U && L){
    for (int row=0; row<sizey; row++){
      U->newline(row);
      L->newline(row);
      for (int i=I[row]; i<I[row+1]; i++){
        col=J[i];
        if(col<row) L->append(row,col,ilu[i]);
        else U->append(row,col,ilu[i]);
      }
      L->append(row,row,1.0);
    }
    U->finalize();
    L->finalize();
    delete[] ilu;
    ilu=NULL;
  }
}

// appends in a triangular matrix. If in debugging mode, checks valid entries.
void TriangularMatrix::append(int y, int x, double value){
#ifndef NDEBUG
  if (type==0 && (x<y) ) {
    printf("Tried to write invalid entry (%i,%i) of upper triangular matrix.\n",y,x);
    exit(-1);
  }
  if (type==1 && (x>y) ) {
    printf("Tried to write invalid entry (%i,%i) of lower triangular matrix.\n",y,x);
    exit(-1);
  }
#endif
  SparseMatrix::append(y,x,value);
}

// Solve linear system with triangular matrix directly
void TriangularMatrix::solveLinearSystem(Vector* x, Vector* b){
  int rowstart, rowend;
  double tempvalue;

#ifndef NDEBUG
  if(!finalized) {printf("Matrix not finalized. Cannot invert.\n"); exit(-1);}
#endif

  if (type==0){ //upper triangular matrix
    for (int row=sizey-1; row>-1; row--){
      tempvalue = b->get(row);
      rowstart = I[row];
      rowend = I[row+1];
#ifndef NDEBUG
      if(val[rowstart]==0.0) {printf("Tried to invert singular matrix.\n"); exit(-1);}
#endif
      for (int i=rowstart+1; i<rowend; i++){
        tempvalue -= val[i]*x->get(J[i]);
      }
      x->set(row,tempvalue/val[rowstart]);
    }     
  }
  if (type==1){ //upper triangular matrix
    for (int row=0; row<sizey; row++){
      tempvalue = b->get(row);
      rowstart = I[row];
      rowend = I[row+1];
#ifndef NDEBUG
      if(val[rowend-1]==0.0) {printf("Tried to invert singular matrix.\n"); exit(-1);}
#endif
     for (int i=rowstart; i<rowend-1; i++){
        tempvalue -= val[i]*x->get(J[i]);
      }
      x->set(row,tempvalue/val[rowend-1]);
    }     
  }

}

#ifdef USEMPI
//============================================================================
// Parallel routines (Specific to iterative methods with square matrices)
//============================================================================

// Pre-process the matrix at each processor to know exactly which processors it needs to commuincate with for Matvec multiplications
// The idea here is that not all of the right hand side vector needs to be updated at every iteration, as the matrix is sparse, and
// if there is any structure at all, we don't want to waste time in communication
//
// This function will find any structure if there exists any in terms of reducing communication costs. (Note we are not assuming any
// particular structure a priori, but it is nice to reduce the commuincation cost). The function takes less processing than a single
// CG iteration, so if there's no structure, we're not losing much, but we have the potential to improve communication time a lot
void SparseMatrix::preProcess(int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world){

  //(*need_proc)=new (nothrow) short(numproc);
  if (need_proc==NULL) {printf("need_proc has not been initialized\n"); exit(-1);}
  for (int i=0; i<numproc; i++) need_proc[i]=0;
  
  int shortproc=sizex/numproc; // how many lines do processors with the least lines have
  int longproc=shortproc+1; // how many lines do processors with the most lines have
  int num_long=sizex%numproc; // how many processors have the most lines
  int whichproc;
  for (int entry=0; entry<I[sizey]; entry++){
    // we look at J[entry], which gives us the column where the entry-th entry of the matrix is located.
    // first we check if its column is less than num_long*longproc, which means it is located in one of the processors with an extra line
    // if it is there, then it is in processor (int)column/longproc
    // otherwise, it is located in a processor with less lines, so we see the leftover lines after all long processors finish,
    // and check which number of short processor it ends up in
    if (J[entry]<num_long*longproc) whichproc=J[entry]/longproc;
    else whichproc=(J[entry]-num_long*longproc)/shortproc+num_long;
    need_proc[whichproc]=1;  // then we say that this entry needs info from whichproc, so we should keep this part of the vector updated
  }
  
  //(*send_proc)=new (nothrow) short(numproc);
  if (send_proc==NULL){printf("send_proc has not been initialized.\n"); exit(-1);}
  MPI_Alltoall(need_proc,1,MPI_SHORT,send_proc,1,MPI_SHORT,comm_world);

  return;
}

// parallel matrix vector multiplication
// performs outall=this*in in parallel. outall will store the full matrix vector multiplication, outlocal stores the result of only the rows in this processor
// times the vector, and requires no commuincation. rowdist contains the first row stored in each processor, need_proc and send_proc are produced in
// preprocess, and store the processors that need to commuincate with this one. requests is an array of MPI requests used to determine whether communication has finished.
// the frist activerequests[0] elements are send requests, and outlocal should not be modified until all these requests have cleared. The next activerequests[1] elements
// are receive requests, and outall will not be ready untill all these requests clear.
void SparseMatrix::parVectorMult(Vector* in, Vector* outall, Vector* outlocal, int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world, MPI_Request* request, int* activerequests){

  // Perform local matrix vector multiplication
  vectorMult(in, outlocal);
  double* sendaddress=outlocal->getInitAddress();
  double* recvaddress=outall->getInitAddress();
  activerequests[0]=0;
  activerequests[1]=0;

  // copy outlocal to outall for this processor
  //printf("What i will send: %f, %f to %i, this many els: %i\n",sendaddress[0],sendaddress[1],rowdist[rank],rowdist[rank+1]-rowdist[rank]);
  memcpy(recvaddress+rowdist[rank],sendaddress,sizeof(double)*(rowdist[rank+1]-rowdist[rank]));
  //printf("I see: %f,%f,%f,%f,%f\n",recvaddress[0],recvaddress[1],recvaddress[2],recvaddress[3],recvaddress[4]);

  // Commuincate to the necessary processors
  // These calls are non-blocking, the function will return control to the parent function without waiting for them to finish
  // It is the responsibility of the parent function to check the requests to make sure commuincations have finished before using the data
  //printf("numproc: %i\n",numproc);
  for(int i=0; i<numproc; i++){
    if (send_proc[i]==0 || i==rank) continue;
    //printf("sending from %i to %i\n",rank,i);
    MPI_Isend(sendaddress,sizey,MPI_DOUBLE,i,2*i,comm_world,request+(activerequests[0]++));
  }
  for(int i=0; i<numproc; i++){
    if (need_proc[i]==0 || i==rank) continue;
    MPI_Irecv(recvaddress+rowdist[i],rowdist[i+1]-rowdist[i],MPI_DOUBLE,i,2*rank,comm_world,request+activerequests[0]+(activerequests[1]++));
  }

  return;
}

// Parallel daxpy... pretty much the same as matrix vector multiplication, but adds y as well (and can multiply by a coefficient at the end)
void SparseMatrix::pardaxpy(Vector* x, Vector* y, short op, double coeff, Vector* outall, Vector* outlocal, int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world, MPI_Request* request, int* activerequests){

  // Perform local matrix vector multiplication
  daxpyOffset(x, y, outlocal, op, coeff, rowdist[rank]);
  double* sendaddress=outlocal->getInitAddress();
  double* recvaddress=outall->getInitAddress();
  activerequests[0]=0;
  activerequests[1]=0;

  // copy outlocal to outall for this processor
  //printf("What i will send: %f, %f to %i, this many els: %i\n",sendaddress[0],sendaddress[1],rowdist[rank],rowdist[rank+1]-rowdist[rank]);
  memcpy(recvaddress+rowdist[rank],sendaddress,sizeof(double)*(rowdist[rank+1]-rowdist[rank]));
  //printf("I see: %f,%f,%f,%f,%f\n",recvaddress[0],recvaddress[1],recvaddress[2],recvaddress[3],recvaddress[4]);

  // Commuincate to the necessary processors
  // These calls are non-blocking, the function will return control to the parent function without waiting for them to finish
  // It is the responsibility of the parent function to check the requests to make sure commuincations have finished before using the data
  //printf("numproc: %i\n",numproc);
  for(int i=0; i<numproc; i++){
    if (send_proc[i]==0 || i==rank) continue;
    //printf("sending from %i to %i\n",rank,i);
    MPI_Isend(sendaddress,sizey,MPI_DOUBLE,i,2*i,comm_world,request+(activerequests[0]++));
  }
  for(int i=0; i<numproc; i++){
    if (need_proc[i]==0 || i==rank) continue;
    MPI_Irecv(recvaddress+rowdist[i],rowdist[i+1]-rowdist[i],MPI_DOUBLE,i,2*rank,comm_world,request+activerequests[0]+(activerequests[1]++));
  }

  return;
}

// CG in parallel
// This may be kind of complicated... we are not going to keep complete vectors updated, the only vectors that need to be updated beyond the locally
// stored rows are those that are multiplied by a matrix, and whatever they depend on. The only vectors that are multiplied by A are x and r, but
// x is only multiplied initially, so we will never communicate x. r is multiplied at each iteration, so it needs to be kept updated at each iteration
// (only the rows that are relevant for the vector multiplication). r depends on q, which depends on s, so these will also be updated. Any dot product
// is computed with the local section and the reduced.
// At the end, we will assemble the solution vector with the correct pieces of x
double SparseMatrix::parCG(Vector* b, Vector* x,int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world){
  
  Vector* r=new (nothrow) Vector(sizex,true);
  Vector* rlocal=new (nothrow) Vector(sizey,true);
  Vector* s=new (nothrow) Vector(sizex,true);
  Vector* slocal=new (nothrow) Vector(sizey,true);
  Vector* q=new (nothrow) Vector(sizex,true);
  Vector* qlocal=new (nothrow) Vector(sizey,true);
  Vector* p=new (nothrow) Vector(sizex,true);
  double rholocal, rhoprev, rho0,alpha, mulocal;
  double beta=0;
  double rho=0;
  double mu=0;
  int k=0;
  double tol=1e-12;
  MPI_Request* requests=new (nothrow) MPI_Request[numproc];
  int activerequests[2];
  MPI_Status status;

  // r=b-Ax
  //printf("r=b-Ax\n");
  pardaxpy(x,b,-1,-1,r,rlocal,numproc,rank,rowdist,need_proc,send_proc,comm_world,requests,activerequests);
  rholocal=rlocal->dot(rlocal);
  // reduce rho
  MPI_Allreduce(&rholocal,&rho,1,MPI_DOUBLE,MPI_SUM,comm_world);
  rho0=rho;
  // Now we will compute s, which depends on r, so we must check r is fully updated
  // We only need receives to have completed, but later we will need sends to complete. To avoid creating a new requests array for s
  // we check the sends now as well (because we only ever multiply by r once, this doesn't affect timing much
  for (int i=0; i<activerequests[0]; i++) MPI_Wait(requests+i,&status); //check sends
  for (int i=0; i<activerequests[1]; i++) MPI_Wait(requests+activerequests[0]+i,&status); //check receives
  // s=Ar
  //printf("s=Ar\n");
  pardaxpy(r,NULL,1,1,s,slocal,numproc,rank,rowdist,need_proc,send_proc,comm_world,requests,activerequests);
  // mu=r'*s
  mulocal=rlocal->dot(slocal);
  MPI_Allreduce(&mulocal,&mu,1,MPI_DOUBLE,MPI_SUM,comm_world);
  // alpha=rho/mu
  alpha=rho/mu;

#ifndef NDEBUG
  ProfilerStart("./output.pprof");
#endif

  double time=MPI_Wtime();
  while(true){
    k++;
    p->parcapdy(r,beta,1,numproc,rank,rowdist,need_proc); //p=r+beta*p
    // We will use s in here, so check if it has received all the information
    for (int i=0; i<activerequests[1]; i++) MPI_Wait(requests+activerequests[0]+i,&status); //check receives
    q->parcapdy(s,beta,1,numproc,rank,rowdist,need_proc); //q=s+beta*q
    x->parcapdy(p,1,alpha,numproc,rank,rowdist,need_proc); //x=x+alpha*p
    r->parcapdy(q,1,-alpha,numproc,rank,rowdist,need_proc); //r=r-alpha*q
    // update rlocal to dot against s
    memcpy(rlocal->getInitAddress(),r->getInitAddress()+rowdist[rank],sizeof(double)*(rowdist[rank+1]-rowdist[rank]));
    if(sqrt(rho)<tol*sqrt(rho0)) break;
    // We will rewrite s in here, so check if all its sends have completed
    for (int i=0; i<activerequests[0]; i++) MPI_Wait(requests+i,&status); //check sends
    pardaxpy(r,NULL,1,1,s,slocal,numproc,rank,rowdist,need_proc,send_proc,comm_world,requests,activerequests); // s=Ar
    rhoprev=rho;
    // rho=r'*r , mu=s'*r
    rholocal=rlocal->dot(rlocal);
    mulocal=rlocal->dot(slocal);
    MPI_Allreduce(&rholocal,&rho,1,MPI_DOUBLE,MPI_SUM,comm_world);
    MPI_Allreduce(&mulocal,&mu,1,MPI_DOUBLE,MPI_SUM,comm_world);

    beta=rho/rhoprev;
    alpha=rho/(mu-rho*(beta/alpha));
  }
  if (rank==0){
    printf("Final error squared: %e\n",rho);
    printf("CG Converged in %i iterations.\n",k);
    printf("CG time: %f\n",MPI_Wtime()-time);
  }
  
  delete r;
  delete rlocal;
  delete s;
  delete slocal;
  delete q;
  delete qlocal;

#ifndef NDEBUG
  ProfilerStop();
#endif

  // Collect x into every processor. Not all of x is correct locally, but at least the rows corresponding to this processor are,
  // so we do an allgather with the rows corresponding to each processor
  // Can sendbuf and recvbuf intersect? If they can, these can be made more efficient(skip the copy), but for safety I implement it a bit differently
  int* recvcounts=new(nothrow) int[numproc];
  for(int i=0; i<numproc; i++) recvcounts[i]=rowdist[i+1]-rowdist[i];
  p->copy(x); // we don't use p's memory anymore, so copy x into it
  MPI_Allgatherv(p->getInitAddress()+rowdist[rank],rowdist[rank+1]-rowdist[rank],MPI_DOUBLE,x->getInitAddress(),recvcounts,rowdist,MPI_DOUBLE,comm_world);

  delete p;

  return(sqrt(rho));
}
#endif

//============================================================================
// Other useless routines
//============================================================================

// Just for testing, display the matrix
void SparseMatrix::display(){

  printf("I=");
  for (int i=0; i<sizey+1; i++) printf("%i ",I[i]);
  printf("\n");

  printf("J=");
  for (int i=0; i<nnzmax; i++) printf("%i ",J[i]);
  printf("\n");

  printf("val=");
  for (int i=0; i<nnzmax; i++) printf("%f ",val[i]);
  printf("\n");

  printf("Matrix=\n");
  for (int y=0; y<sizey; y++){
    if (I[y+1]-I[y]==0){
      for (int x=0; x<sizex; x++) printf("0 ");
    }else{
      for (int x=0; x<J[I[y]]; x++) printf("0 ");
      for (int xout=I[y]; xout<I[y+1]; xout++){
	printf("%1.2f ",val[xout]);
	if (xout!=I[y+1]-1){
	  for (int x=J[xout]+1; x<J[xout+1]; x++) printf("0 ");
	}else{
	  for (int x=J[xout]+1; x<sizex; x++) printf("0 ");
	}
      }
    }
    printf("\n");
  }

  return;
}
