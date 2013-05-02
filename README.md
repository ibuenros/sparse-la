Basic library for sparse linear algebra routines.

Parallel capabilities through OpenMP and MPI (use flags USEMPI, USEOPENMP at compilation).

Debug mode checks vectors and matrices before performing any operations (eg. correct sizes, protects accessing out-of-bounds entries, etc.). Use flag NDEBUG to skip all checks: faster, but doesn't check anything.

I made this library from scratch for class projects and for fun. Obviously not the best option out there, and possibly buggy.

Included classes and methods:

Vector:
-set, get, copy, display routines.
-vector addition and scalar multiplication, serial and parallel versions

SparseMatrix:
-build using newline, append, and finalize
-change non-zero entries after finalized(slow),(zero entries can't be changed)
-get entries(slow)
-matrix-vector multiplication and daxpy, serial and parallel versions
-preprocess matrix to increase efficiency of parallel operations
-ConjugateGradient solver, serial and parallel versions, preconditioned or non-preconditioned

TriangularMatrix:
-direct linear solver

Preconditioner:
-compute Jacobi or ILU preconditioners
