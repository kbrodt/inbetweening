#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/LU>
using namespace Eigen;
struct SymDir;
//runs gradient descent on the Symmetric Dirichlet energy with L2 loss
double optimizeDeformation(
  Ref<VectorXd> Vx,
  Ref<VectorXd> Vy,
  MatrixXd V0,
  MatrixXi F,
  const std::vector<int>& constrainedVerts,
  MatrixXd constrainedPos,
  double wConstraints,
  int maxIter,
  double xTol
);
double myF(const MatrixXd& newV, MatrixXd& grad, SymDir& symDir, double wConstraints, bool computeGrad);
