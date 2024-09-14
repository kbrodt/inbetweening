#include "optimize.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include "fastSymDir.h"
#include "igl/flip_avoiding_line_search.h"
#include "igl/cotmatrix.h"
#include "igl/writeOBJ.h"
namespace py = pybind11;

double myF(const MatrixXd& newV, MatrixXd& grad, SymDir& symDir, double wConstraints, bool computeGrad)
{
	double result = 0;
	result = symDir.fastSymDir(newV, grad, computeGrad);
	MatrixXd l2Grad = MatrixXd::Zero(newV.rows(), 2);
	result += wConstraints*symDir.l2Loss(newV, l2Grad, computeGrad);

	if (computeGrad)
		grad += wConstraints* l2Grad;
	return result;
}

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
)
{
	/*MatrixXd V0_3d(V0.rows(), 3);
	V0_3d << V0, MatrixXd::Zero(V0.rows(), 1);
	igl::writeOBJ("initialMesh.obj", V0_3d, F);*/
	const double rho = 0.6;
	SymDir symDir(V0, F, constrainedPos, constrainedVerts);
	MatrixXd X = V0;
	int nv = V0.rows();
	MatrixXd grad = MatrixXd::Zero(nv, 2);
	auto fun = [&symDir, &wConstraints](const MatrixXd& newV, MatrixXd& grad, bool computeGrad) {
		return myF(newV, grad, symDir, wConstraints, computeGrad);
	};

	double f=std::numeric_limits<double>::max();
	
	//stuff for preconditioning
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> Lsolver; //Cholesky decomposition for Laplacian
	Eigen::SparseMatrix<double> Lcot;
	igl::cotmatrix(V0, F, Lcot);
	Eigen::SparseMatrix<double> Lcot_no_last_column = Lcot.leftCols(Lcot.cols() - 1);
	Eigen::SparseMatrix<double, Eigen::RowMajor> Lcot_no_last_column_row_major(Lcot_no_last_column);
	Eigen::SparseMatrix<double, Eigen::RowMajor> Lcot_final(Lcot.cols() - 1, Lcot.cols() - 1);
	Lcot_final = Lcot_no_last_column_row_major.topRows(Lcot.cols() - 1);
	Eigen::SparseMatrix<double> Lapl = -Lcot_final;
	Lsolver.compute(Lapl);
	bool usePreconditioner = true;
	if (Lsolver.info() != Eigen::Success) {
		// decomposition failed
		std::cout << "Incorrect (not SPD) laplacian. Skipping preconditioning" << std::endl;
		usePreconditioner = true;
	}
	//end of stuff for preconditioning

	for (int i = 0; i < maxIter; i++) {
		f = fun(X, grad, true);

		if (grad.norm() < xTol)
		{
			std::cout << "Converged in " << i << " iterations" << std::endl;
			break;
		}
		//std::cout << "It: " << i << ", f = " << f << std::endl;
		//std::cout << f << ", ";
		
		//Laplacian preconditioning
		if (usePreconditioner)
		{
			Eigen::Vector2d translation;
			Eigen::VectorXd onesVector(nv); onesVector.setConstant(1.0);
			for (int col = 0; col < 2; ++col)
			{
				translation(col) = grad.col(col).sum() / nv;
				grad.col(col) -= onesVector * translation(col);
			}

			for (int col = 0; col < 2; col++)
			{
				Eigen::VectorXd incompleteCol = grad.col(col).head(nv - 1);
				incompleteCol = Lsolver.solve(incompleteCol);
				grad.col(col).head(nv - 1) = incompleteCol;
				grad.col(col)(nv - 1) = 0;
				grad.col(col) -= onesVector * grad.col(col).sum() / nv;
				grad.col(col) += onesVector * translation(col);
			}
		}

		MatrixXd d = -grad; //search direction, no preconditioning for now
		double min_positive_root = igl::flip_avoiding::compute_max_step_from_singularities(X, F, d);
		double max_step_size = std::min(1.0, 0.999* min_positive_root);
		std::function<double(Eigen::MatrixXd&)> compute_energy = [&](Eigen::MatrixXd& aaa) { return fun(aaa, grad, false); };
		MatrixXd Xprev = X;
		f = igl::line_search(X, d, max_step_size, compute_energy, f);
		if ((X - Xprev).norm() < 1e-10)
		{
			std::cout << "Stuck in a local minimum" << std::endl;
			break;
		}
	}
	Vx = X.col(0);
	Vy = X.col(1);

	/*MatrixXd result_3d(X.rows(), 3);
	result_3d << X, MatrixXd::Zero(X.rows(), 1);
	igl::writeOBJ("optimizedMesh.obj", result_3d, F);*/
	return f;
}

namespace py = pybind11;
PYBIND11_MODULE(fastSymDir, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("optimizeDeformation", &optimizeDeformation);

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}
