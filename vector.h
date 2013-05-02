#ifndef _VECTOR_
#define _VECTOR_

class Vector{

 private:
  //double* val;
  int vecsize;
#ifndef NDEBUG
  bool* entryset;
#endif

 public:
  double* val;

 public:
  Vector(int size, bool setzeros);
  ~Vector();
#ifndef NDEBUG
  void set(int idx, double value);
  double get(int idx);
#endif
#ifdef NDEBUG
  inline void set(int idx, double value){val[idx]=value;}
  inline double get(int idx){return(val[idx]);}
#endif
  void increase(int idx, double value){val[idx] += value;}
  double dot(Vector* other);
  void display();
  void capdy(Vector* y, double c, double d); //c*a+d*y
  void parcapdy(Vector* y,double c, double d,int numproc, int rank, int* rowdist, short* need_proc);
  int length();
  void copy(Vector* other);
  double* getInitAddress();
};

#endif
