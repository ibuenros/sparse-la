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

#include "vector.h"

// Vector constructor (if setzeros initialize the vector to all 0)
Vector::Vector(int size, bool setzeros){

  vecsize=size;
  val=new (nothrow) double[size];
  if(val==NULL){ printf("Could not allocate vector.\n"); exit(-1);}

#ifndef NDEBUG
  entryset=new (nothrow) bool[size];
  if (entryset==NULL){ printf("Could not allocate entryset in vector.\n"); exit(-1);}
  for (int i=0; i<size; i++){
    entryset[i]=false;
  }
#endif

  if (setzeros){ //if the vector should be filled with zeros (unnecessary if the program will fill it either way)
    for (int i=0; i<size; i++){
      val[i]=0;
#ifndef NDEBUG
      entryset[i]=true;
#endif
    }
  }

  return;
}

Vector::~Vector(){

  delete[] val;

  return;
}

//======================================================================
// Basic Functions
//======================================================================

// this is weird... if NDEBUG then get and set are defined inline in matrix.h... don't know if this makes things any faster
#ifndef NDEBUG
// Set the value of entry idx to value
void Vector::set(int idx, double value){

  entryset[idx]=true;
  if (idx >= vecsize){printf("Attempted to write entry %i in vector of length %i.\n",idx,vecsize); exit(-1); }

  val[idx]=value;

  return;
}

// Get the value of entry idx
double Vector::get(int idx){
  
  if (!entryset[idx]){printf("Attempted to read entry %i in vector, which has not been set.\n",idx); exit(-1);}

  return(val[idx]);
}
#endif

// length
int Vector::length(){
  return vecsize;
}

// Compute the dot product of this with other
double Vector::dot(Vector* other){

  double result=0;
#ifndef NDEBUG
  for (int idx=0; idx<vecsize; idx++){
    if (!entryset[idx]){printf("Attempted to read entry %i in vector for a dot product, which has not been set.\n",idx); exit(-1);}
  }
  if (other->length() != vecsize){printf("Attempted dot product between vectors of different sizes: %i and %i.\n",vecsize,other->length()); exit(-1);}
#endif

  double* initaddress=other->getInitAddress();
#ifdef USEOPENMP
#pragma omp parallel for reduction(+:result)
#endif
  for (int idx=0; idx<vecsize; idx++){
    result += val[idx]*initaddress[idx];
  }
  return result;
}

// Do c*this+d*y
void Vector::capdy(Vector* y, double c, double d){
#ifndef NDEBUG
  for (int idx=0; idx<vecsize; idx++){
    if (!entryset[idx]){printf("Attempted to read entry %i in vector for a vector sum, which has not been set.\n",idx); exit(-1);}
  }
  if (y->length() != vecsize){printf("Attempted to add vectors of different sizes: %i and %i.\n",vecsize,y->length()); exit(-1);}
#endif

  /*
  //Old code
  for (int idx=0; idx<vecsize; idx++){
    if(y==NULL) val[idx] = c*val[idx];
    else val[idx] = c*val[idx]+d*y->get(idx);
  }
  */

  // Attemp to accelerate c*a+d*y by bypassing get
  double* initaddress;
  if(y!=NULL) initaddress=y->getInitAddress();
#ifdef USEOPENMP
#pragma omp parallel for
#endif
  for (int idx=0; idx<vecsize; idx++){
    if(y==NULL) val[idx] = c*val[idx];
    else val[idx] = c*val[idx]+d*initaddress[idx];
  }

  return;
}

//Copy vector other into vector this
void Vector::copy(Vector* other){

#ifndef NDEBUG
  //for (int idx=0; idx<vecsize; idx++){
  //  if (!entryset[idx]){printf("Attempted to read entry %i in vector for a vector copy, which has not been set.\n",idx); exit(-1);}
  //}
  if (other->length() != vecsize){printf("Attempted to copy vectors of different sizes: %i and %i.\n",vecsize,other->length()); exit(-1);}
#endif

  for (int idx=0; idx<vecsize; idx++){
    set(idx,other->get(idx));
  }
  return;
}

double* Vector::getInitAddress(){
  return val;
}

#ifdef USEMPI
//======================================================================
// Parallel routines
//======================================================================

// Faster version of c*a+d*y, only computes the sum of relevatn parts of the vector
// The original c*a+d*y was consuming most of the time... this should improve a bit with
// large numbers of processors
void Vector::parcapdy(Vector* y,double c, double d,int numproc, int rank, int* rowdist, short* need_proc){
#ifndef NDEBUG
  //for (int idx=0; idx<vecsize; idx++){
  //  if (!entryset[idx]){printf("Attempted to read entry %i in vector for a vector sum, which has not been set.\n",idx); exit(-1);}
  //}
  //if (y->length() != vecsize){printf("Attempted to add vectors of different sizes: %i and %i.\n",vecsize,y->length()); exit(-1);}
#endif

  double* initaddress;
  if(y!=NULL) initaddress=y->getInitAddress();
  for (int proc=0; proc<numproc; proc++){
    if (need_proc[proc]==1 || rank==proc){
      for (int idx=rowdist[proc]; idx<rowdist[proc+1]; idx++){
	if(y==NULL) val[idx] = c*val[idx];
	else val[idx] = c*val[idx]+d*initaddress[idx];
      }
    }
  }

  return;
}
#endif

//=====================================================================
// Miscellaneous
//=====================================================================

void Vector::display(){

  printf("Vec=\n");
  for (int i=0; i<vecsize; i++) printf("%f\n",val[i]);

  return;
}
