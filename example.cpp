/******************************************

  Example of ode45 solver application 

  Try to solve simple harmonic equation

	x'' = -x

  Transform into vector: x1 = x, x2 = x'
  Then x1' = x2, x2' = -x1

  Initial conditions: x1 = 0, x2 = 1 (sine)

*******************************************/
#include <iostream>
#include "ode45.h"

using namespace std;
using namespace Eigen;

#define DIM 2   // number of equations

typedef Matrix<double,DIM,1> Vec2;    // can be used "standard" Vector2d

// Differential equation
Vec2 equation(double time, Vec2& y);

// Condition of maximum
bool isMaximum(const vector<double>& t, const vector<Vec2>& y);

// 
int main() {
   // solver
   ode45<DIM> de(0.01);
   // initial conditions
   Vec2 y0;
   y0 << 0, 1;
   // time interval
   Vector2d time(0,M_PI);

   // solve
   MatrixXd res = de.solve(equation, y0, time); 
   // result
   int last = res.rows()-1;
   cout << "Number of points: " << (last+1) << endl;
   cout << "	Time	x1	x2" << endl;
   cout << res.row(last) << endl;
   // save to file
   de.toCSV("sin.csv", res);

   // find first maximum
   res = de.solve(equation, y0, time, isMaximum);
   // result
   last = res.rows()-1;
   cout << "Number of points: " << (last+1) << endl;
   cout << "	Time	x1	x2" << endl;
   cout << res.row(last) << endl;

   return 0;
}

// ODE
Vec2 equation(double time, Vec2& y)
{
   Vec2 res;
   res(0) = y(1);
   res(1) = -y(0);

   return res;
}

// In maximum left derivative is positive and left is negative
bool isMaximum(const vector<double>& t, const vector<Vec2>& y)
{
   int n = y.size()-1;

   return n > 0 && y[n](1) <= 0 && y[n-1](1) >= 0;
}
