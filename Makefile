CC = g++
CFLAGS = -I.
OBJ = vector.o matrix.o preconditioner.o solver.o
INCLUDES = vector.h matrix.h preconditioner.h solver.h

.PHONY: $(INCLUDES) clean

archive: $(OBJ) $(INCLUDES)
	rm -f ../include/sparse.h
	ln -s $(CURDIR)/sparse.h ../include/sparse.h
	ar ru libsparse.a $(OBJ)
	rm -f ../lib/libsparse.a
	ln -s $(CURDIR)/libsparse.a ../lib/libsparse.a
	
$(INCLUDES):
	rm -f ../include/Sparse/$@
	ln -s $(CURDIR)/$@ ../include/Sparse/$@

%.o: %.cpp $(INCLUDES)
	$(CC) -c -o $@ $< $(CFLAGS)

tests: $(OBJ)
	g++ -o testTriangular testTriangular.cpp matrix.cpp 
	g++ -o testILU testILU.cpp preconditioner.cpp matrix.cpp

clean:
	rm -f *.o *.a testTriangular testILU
