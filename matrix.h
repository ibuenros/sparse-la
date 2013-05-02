#ifndef _MATRIX_
#define _MATRIX_

#include "vector.h"

class TriangularMatrix;

class SparseMatrix{

 protected:
  int nnz;
  int sizex, sizey;
  bool finalized;
  int lastrow, lastcolumn, lastwritten, nnzmax;
  int* J;
  int* I;
  double* val;
  double* ilu;

 public:
  SparseMatrix(int sizeyin, int sizexin, int nnzin);
  ~SparseMatrix();

  void newline(int lineno);
  void append(int y,int x, double value);
  void finalize();
  int addToEntry(int y, int x, double value);
  double getEntry(int y, int x);

  void vectorMult(Vector* in, Vector* out);
  void daxpy(Vector* x, Vector* y, Vector* out, short op, double coeff);
  void daxpyOffset(Vector* x, Vector* y, Vector* vecout, short op, double coeff, int offset);
  void computeILU(TriangularMatrix* L, TriangularMatrix* U);

  void display();
  int sizesquare() {if(sizex==sizey) return sizex; else return -1;}
  int getNonZero() {return nnz;}
  double CG(Vector* in, Vector* out);

#ifdef USEMPI
  double parCG(Vector* b, Vector* x,int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world);
  void pardaxpy(Vector* x, Vector* y, short op, double coeff, Vector* outall, Vector* outlocal, int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world, MPI_Request* request, int* activerequests);
  void preProcess(int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world);
  void parVectorMult(Vector* in, Vector* outall, Vector* outlocal, int numproc, int rank, int* rowdist, short* need_proc, short* send_proc, MPI_Comm comm_world, MPI_Request* request, int* activerequests);
#endif
};

class TriangularMatrix: public SparseMatrix{

  private:
    int type; // 0 upper, 1 lower

  public:
    TriangularMatrix(int sizeyin, int sizexin, int nnz, int typein);
    ~TriangularMatrix();
    void append(int y, int x, double value);
    void solveLinearSystem(Vector* x, Vector* b);
};

#endif
