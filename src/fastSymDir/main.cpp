#include "optimize.h"
#include "fastSymDir.h"
#include <iomanip> 
#include <time.h>

int main()
{
	int nVerts = 1000;
	int nTri = 1000;
	MatrixXd V(nVerts, 2);
	V.setRandom();
	MatrixXi F(nTri, 3);
	for(int i = 0; i < nTri; i++) {
		F(i, 0) = rand() % nVerts;
		F(i, 1) = rand() % nVerts;
		F(i, 2) = rand() % nVerts;
		while (F(i, 0) == F(i, 1) || F(i, 0) == F(i, 2) || F(i, 1) == F(i, 2)) {
			F(i, 1) = rand() % nVerts;
			F(i, 2) = rand() % nVerts;
		}
	}
	std::vector<int> constrainedVertices;
	int nConstrained = 40;
	for (int i = 0; i < nConstrained; i++) {
		constrainedVertices.push_back(rand() % nVerts);
	}
	std::sort(constrainedVertices.begin(), constrainedVertices.end());
	auto it = std::unique(constrainedVertices.begin(), constrainedVertices.end());
	constrainedVertices.erase(it, constrainedVertices.end());//remove duplicates
	Eigen::MatrixXd constrainedPositions(constrainedVertices.size(), 2);
	constrainedPositions.setRandom();

	MatrixXd newV(nVerts,2);
	newV.setRandom();

	Eigen::MatrixXd grad(nVerts, 2);
	clock_t begin = clock();
	VectorXd outX(nVerts), outY(nVerts);
	double wConstraints = 10;
	optimizeDeformation(outX, outY, V, F, constrainedVertices, constrainedPositions, wConstraints, 10000, 1e-6);

	clock_t end = clock();
	std::cout << double(end - begin) / CLOCKS_PER_SEC << " seconds" << std::endl;

	SymDir symDir(V, F, constrainedPositions, constrainedVertices);
	myF(newV, grad, symDir, wConstraints, true);
	//fd gradient
	std::cout << std::endl;
	MatrixXd fdGrad(nVerts, 2);
	fdGrad.setZero();
	for (int i = 0; i < nVerts; i++) {
		for (int j = 0; j < 2; j++) {
			double eps = 1e-6;
			MatrixXd newV2 = newV;
			MatrixXd newV3 = newV;
			newV2(i, j) += eps;
			newV3(i, j) -= eps;
			fdGrad(i,j) = (myF(newV2,fdGrad, symDir, wConstraints,false) - myF(newV3,fdGrad, symDir, wConstraints,false)) / (2*eps);
		}
	}
	std::cout << grad << std::endl << std::endl;
	std::cout << fdGrad << std::endl;

	bool error = false;
	for (int i=0; i<nVerts; ++i)
		for (int j = 0; j < 2; ++j)
		{
			if (std::abs(fdGrad(i, j) - grad(i, j)) > 1e-2)
			{
				std::cout << std::setprecision(9) << "Possible error at " << i << ", " << j << ": " << fdGrad(i, j) << " vs " << grad(i, j) << ", diff = " << std::abs(fdGrad(i, j) - grad(i, j)) << std::endl;
				error = true;
			}
		}

	if (!error)
	{
		std::cout << "Gradients okay" << std::endl;
	}
	return 0;
}