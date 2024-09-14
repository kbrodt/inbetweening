#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace Eigen;
struct SymDir
{
	MatrixXd triangleToMat(const MatrixXd& V, int i);

	SymDir(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd d0, const std::vector<int>& constrainedVertices);//pass by value for now, change to const ref later
	//computes Symmetric Dirichlet energy of a mesh
	double fastSymDir(const MatrixXd& newV, MatrixXd& grad, bool computeGrad=false);
	double l2Loss(const MatrixXd& newV, MatrixXd& grad, bool computeGrad);
private:
	Eigen::VectorXd areas;
	Eigen::VectorXd Ainv11, Ainv12, Ainv21, Ainv22;
	std::vector<Eigen::Matrix2d> Ainv;
	Eigen::MatrixXi F_;
	Eigen::MatrixXd d0_;//constrained vertex positions
	std::vector<int> constrainedVertices_;
};
