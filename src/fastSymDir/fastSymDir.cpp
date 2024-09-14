#include "fastSymDir.h"

MatrixXd SymDir::triangleToMat(const MatrixXd& V, int i)
{
	Eigen::Vector2d v1 = V.row(F_(i, 0));
	Eigen::Vector2d v2 = V.row(F_(i, 1));
	Eigen::Vector2d v3 = V.row(F_(i, 2));
	Eigen::Matrix2d A;
	A << v2(0) - v1(0), v3(0) - v1(0),
		v2(1) - v1(1), v3(1) - v1(1);
	return A;
}

SymDir::SymDir(MatrixXd V, MatrixXi F, Eigen::MatrixXd d0, const std::vector<int>& constrainedVertices)
	: F_(F), d0_(d0), constrainedVertices_(constrainedVertices)
{
	areas = Eigen::VectorXd(F.rows());
	Ainv11 = Eigen::VectorXd(F.rows());
	Ainv12 = Eigen::VectorXd(F.rows());
	Ainv21 = Eigen::VectorXd(F.rows());
	Ainv22 = Eigen::VectorXd(F.rows());
	for (int i = 0; i < F.rows(); i++) {
		Eigen::Vector3d v1(V(F(i, 0),0),V(F(i, 0),1),0.0);
		Eigen::Vector3d v2(V(F(i, 1),0),V(F(i, 1),1),0.0);
		Eigen::Vector3d v3(V(F(i, 2),0),V(F(i, 2),1), 0.0);
		areas(i) = 0.5 * (v2 - v1).cross(v3 - v1).norm();
	}
	
	areas /= areas.sum();
	Ainv.resize(F.rows());
	for (int i = 0; i < F.rows(); i++) {
		Eigen::Matrix2d A = triangleToMat(V, i);
		Ainv[i] = A.inverse();
		Ainv11[i] = Ainv[i](0, 0);
		Ainv12[i] = Ainv[i](0, 1);
		Ainv21[i] = Ainv[i](1, 0);
		Ainv22[i] = Ainv[i](1, 1);

		//if (std::isnan(Ainv[i](0,0)) || std::isinf(Ainv[i](0, 0)))
		//	std::cout << A << std::endl;
	}
}

//computes L2 norm of the difference between the new and the constrained vertex positions
double SymDir::l2Loss(const MatrixXd& newV, MatrixXd& grad, bool computeGrad)
{
	double result = (newV(constrainedVertices_, all) - d0_).squaredNorm()/ constrainedVertices_.size();
	if (computeGrad)
	{
		int nVerts = newV.rows();
		grad = MatrixXd::Zero(nVerts,2);
		grad(constrainedVertices_, all) = 2 * (newV(constrainedVertices_, all) - d0_)/ constrainedVertices_.size();
	}
	return result;
}

//computes Symmetric Dirichlet energy of a mesh
double SymDir::fastSymDir(const MatrixXd& newV, MatrixXd& grad, bool computeGrad) {
	int nTri = F_.rows();
	MatrixXd v1 = newV(F_.col(0), all);
	MatrixXd v2 = newV(F_.col(1), all);
	MatrixXd v3 = newV(F_.col(2), all);
	MatrixXd v12 = v2 - v1;
	MatrixXd v13 = v3 - v1;
	VectorXd J11 = v12.col(0).array()*Ainv11.array() + v13.col(0).array()*Ainv21.array();
	VectorXd J12 = v12.col(0).array()*Ainv12.array() + v13.col(0).array()*Ainv22.array();
	VectorXd J21 = v12.col(1).array()*Ainv11.array() + v13.col(1).array()*Ainv21.array();
	VectorXd J22 = v12.col(1).array()*Ainv12.array() + v13.col(1).array()*Ainv22.array();
	VectorXd detJ = J11.array()*J22.array() - J12.array()*J21.array();
	VectorXd sqrDetJ = detJ.array()*detJ.array();
	VectorXd abcd2 = J11.array()*J11.array() + J12.array()*J12.array() + J21.array()*J21.array() + J22.array()*J22.array();
	ArrayXd invSqrDetJ = 1 / sqrDetJ.array();
	VectorXd energy = areas.array() * abcd2.array() * (1 *invSqrDetJ + 1);
	if (computeGrad)
	{
		grad = MatrixXd::Zero(newV.rows(), newV.cols());
		ArrayXd invCubeDetJ = invSqrDetJ / detJ.array();
		//denote Q = (J  * (1 / sqrDetJ + 1) - abcd2 * J.inverse().transpose() / sqrDetJ)
		ArrayXd Q11 = J11.array() * (invSqrDetJ + 1) - abcd2.array() * J22.array() * invCubeDetJ;
		ArrayXd Q12 = J12.array() * (invSqrDetJ + 1) + abcd2.array() * J21.array() * invCubeDetJ;
		ArrayXd Q21 = J21.array() * (invSqrDetJ + 1) + abcd2.array() * J12.array() * invCubeDetJ;
		ArrayXd Q22 = J22.array() * (invSqrDetJ + 1) - abcd2.array() * J11.array() * invCubeDetJ;
		VectorXd dEdB00 = 2 * areas.array() * (Q11 * Ainv11.array() + Q12 * Ainv12.array());
		VectorXd dEdB01 = 2 * areas.array() * (Q11 * Ainv21.array() + Q12 * Ainv22.array());
		VectorXd dEdB10 = 2 * areas.array() * (Q21 * Ainv11.array() + Q22 * Ainv12.array());
		VectorXd dEdB11 = 2 * areas.array() * (Q21 * Ainv21.array() + Q22 * Ainv22.array());
		grad(F_(all, 0), 0) -= dEdB00 + dEdB01;
		grad(F_(all, 1), 0) += dEdB00;
		grad(F_(all, 2), 0) += dEdB01;
		grad(F_(all, 0), 1) -= dEdB10 + dEdB11;
		grad(F_(all, 1), 1) += dEdB10;
		grad(F_(all, 2), 1) += dEdB11;
	}
	return energy.sum();
}
