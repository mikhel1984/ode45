/****************************************************

>  Simple ODE solver based on Runge-Kutta approach.

>  2019, Stanislav Mikhel

*****************************************************/
#ifndef RUNGE_KUTTA_METHOD
#define RUNGE_KUTTA_METHOD

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

template<unsigned int R>
class ode45 {
public:
   // Vector name
   typedef Eigen::Matrix<double,R,1> VectorR;
   // ODE function
   typedef VectorR (*Equation) (double, VectorR&);
   // Break condition
   typedef bool (*Exit) (const std::vector<double>&,const std::vector<VectorR>&);

   // Constructor
   //	timeStep - independent variable step
   ode45(double timeStep=1E-3) : dt(timeStep) {}

   // Find solution for time interval
   //	fn - differential equation as a function: func(time,state) -> first derivative
   //	y0 - initial state
   //	time - time interval (as a vector)
   //	condition - break condition (optional): func(time_list,state_list) -> bool
   // Return:
   //	matrix, where each row has form [t, x1, x2, ...]
   Eigen::MatrixXd solve(Equation fn, VectorR& y0, Eigen::Vector2d& time, Exit condition=0);

   // Find solution for predefined time stamps
   //	fn - differential equation as a function: func(time,state) -> first derivative
   //	y0 - initial state
   //	time - list of time steps
   //	condition - break condition (optional): func(time_list,state_list) -> bool
   // Return:
   //	matrix, where each row has form [t, x1, x2, ...]
   Eigen::MatrixXd solve(Equation fn, VectorR& y0, Eigen::VectorXd& time, Exit condition=0);

   // Return:
   //	current time step
   double getTimeStep() { return dt; }

   // Set new time step
   //	h - new step value
   void setTimeStep(double h) { dt = h; }

   // Export matrix to csv file
   //	fname - file name
   //	m - matrix of results
   //	sep - separator (optional, default is coma)
   void toCSV(const char* fname, Eigen::MatrixXd& m, const char* sep=",");

private:
   // Find next state
   //	fn - ode
   //	t - current time
   //	y - current state
   // Return:
   //	new state vector
   VectorR rk(Equation fn, double t, VectorR& y);

   // Temporal vectors
   VectorR k1, k2, k3, k4, tmp;

   // time step
   double dt;

}; // ode45

// Find next state
template<unsigned int R>
Eigen::Matrix<double,R,1> ode45<R>::rk(ode45<R>::Equation fn, double t, ode45<R>::VectorR& y)
{
   double dt2 = 0.5*dt;
   k1 = fn(t, y);       tmp = y + dt2*k1;
   k2 = fn(t+dt2, tmp); tmp = y + dt2*k2;
   k3 = fn(t+dt2, tmp); tmp = y + dt*k3;
   k4 = fn(t+dt, tmp);

   return y + (dt/6)*(k1 + 2*(k2 + k3) + k4);
}

// Find solution for time interval
template<unsigned int R>
Eigen::MatrixXd ode45<R>::solve(ode45<R>::Equation fn, ode45<R>::VectorR& y0, Eigen::Vector2d& time, ode45<R>::Exit condition)
{
   double t = time(0), tend = time(1);
   assert(t < tend);
   int N = (int) (tend-t)/dt; // estimate capacity

   // collect time
   std::vector<double> tacc;
   tacc.reserve(N);
   tacc.push_back(t);
   // collect results
   std::vector<VectorR> acc;
   acc.reserve(N);
   acc.push_back(y0);

   // evaluate
   for(t = t+dt; t < tend; t += dt) {
      tacc.push_back(t);
      acc.push_back( rk(fn, t, acc.back()) ); 
      if( condition && condition(tacc, acc) ) break;
   }
   
   // to matrix of results
   Eigen::MatrixXd res(acc.size(),R+1);
   for(unsigned int r = 0; r < res.rows(); r++) {
      res(r,0) = tacc[r];
      for(unsigned int c = 0; c < R; c++) {
         res(r,c+1) = acc[r](c);
      }
   }

   return res;
}

// Find solution for predefined time stamps
template<unsigned int R>
Eigen::MatrixXd ode45<R>::solve(ode45<R>::Equation fn, ode45<R>::VectorR& y0, Eigen::VectorXd& time, ode45<R>::Exit condition)
{

   // collect time (for compatibility with "condition")
   std::vector<double> tacc;
   tacc.reserve(time.rows());
   tacc.push_back(time(0));
   // collect results
   std::vector<VectorR> acc;
   acc.reserve(time.rows());
   acc.push_back(y0);

   // evaluate
   for(int i = 0; i < time.rows(); i++) {
      double t = time(i);
      acc.push_back( rk(fn, t, acc.back()) ); 
      if( condition && condition(tacc, acc) ) break;
   }
   
   // to matrix of results
   Eigen::MatrixXd res(acc.size(),R+1);
   for(int r = 0; r < res.rows(); r++) {
      res(r,0) = time(r);
      for(int c = 0; c < R; c++) {
         res(r,c+1) = acc[r](c);
      }
   }

   return res;
}

// Save to csv file
template<unsigned int R>
void ode45<R>::toCSV(const char* fname, Eigen::MatrixXd& m, const char* sep)
{
   std::ofstream file;
   file.open(fname);

   if(!file.is_open()) return;

   // export 
   for(int r = 0; r < m.rows(); r++) {
      int c = 0;
      for(; c < m.cols()-1; c++) { file << m(r,c) << sep; }
      file << m(r,c) << std::endl;
   }
   
   file.close();
}

#endif // RUNGE_KUTTA_METHOD
