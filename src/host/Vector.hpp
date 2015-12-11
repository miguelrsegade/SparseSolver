#pragma once
#include <iostream>

class Vector
{
friend class VectorGPU;
private:
   double* mData; // data stored in vector
   int mSize; // size of vector
public:
   Vector(const Vector& otherVector);
   Vector(int size);
   Vector(int size, double value); // Initialize vector to value
   ~Vector();
   int GetSize() const;
   double& operator[](int i); // zero-based indexing
   // read-only zero-based indexing 
   double Read(int i) const;
   double& operator()(int i); // one-based indexing
   // assignment
   Vector& operator=(const Vector& otherVector);
   Vector operator-() const; // unary -
   Vector operator+(const Vector& v1) const; // binary +
   Vector operator-(const Vector& v1) const; // binary -
   // scalar multiplication
   Vector operator*(double a) const;
   // p-norm method
   double Norm(int p=2) const;
   // Dot product
   double operator*(const Vector& v1) const;
   //Output operator
   friend std::ostream& operator<<(std::ostream& os, const Vector& v);
   // Left scalar multiplication
   friend Vector operator*(double a, const Vector& v);
};


//Prototypes for friend functions
std::ostream& operator<<(std::ostream& os, const Vector& v);
