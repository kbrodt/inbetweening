#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/readPLY.h>
#include <igl/readTGF.h>
#include <igl/verbose.h>
#include <igl/EPS.h>
#include <igl/bbw.h>
#include <igl/lbs_matrix.h>
#include <igl/writeDMAT.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


int selected = 0;
const Eigen::RowVector3d sea_green(70./255.,252./255.,167./255.);
Eigen::MatrixXd V, C, bc, W, M;
Eigen::MatrixXi Ele, BE, CE;
Eigen::VectorXi P, b;


bool custom_boundary_conditions(
  const Eigen::MatrixXd & V  ,
  const Eigen::VectorXi & P  ,
  const Eigen::MatrixXi & BE ,
  const Eigen::MatrixXi & CE ,
  Eigen::VectorXi &       b  ,
  Eigen::MatrixXd &       bc )
{
  using namespace Eigen;
  using namespace std;

  if(P.size()+BE.rows() == 0)
  {
    igl::verbose("^%s: Error: no handles found\n",__FUNCTION__);
    return false;
  }

  vector<int> bci;
  vector<int> bcj;
  vector<double> bcv;

  for(int p = 0;p<P.size();p++) {
    bci.push_back(P(p));
    bcj.push_back(p);
    bcv.push_back(1.0);
  }
  for(int e = 0;e<BE.rows();e++) {
    bci.push_back(BE(e, 0));
    bcj.push_back(P.size() + e);
    bcv.push_back(1.0);
    bci.push_back(BE(e, 1));
    bcj.push_back(P.size() + e);
    bcv.push_back(1.0);
  }
  for(int e = 0;e<CE.rows();e++) {
    for (int j = 0; j < CE.cols(); j++) {
      bci.push_back(CE(e, j));
      bcj.push_back(P.size() + e);
      bcv.push_back(1.0);
    }
  }

  // find unique boundary indices
  vector<int> vb = bci;
  sort(vb.begin(),vb.end());
  vb.erase(unique(vb.begin(), vb.end()), vb.end());

  b.resize(vb.size());
  bc = MatrixXd::Zero(vb.size(),P.size()+BE.rows());
  // Map from boundary index to index in boundary
  map<int,int> bim;
  int i = 0;
  // Also fill in b
  for(vector<int>::iterator bit = vb.begin();bit != vb.end();bit++)
  {
    b(i) = *bit;
    bim[*bit] = i;
    i++;
  }

  // Build BC
  for(i = 0;i < (int)bci.size();i++)
  {
    assert(bim.find(bci[i]) != bim.end());
    bc(bim[bci[i]],bcj[i]) = bcv[i];
  }

  // Normalize across rows so that conditions sum to one
  for(i = 0;i<bc.rows();i++)
  {
    double sum = bc.row(i).sum();
    assert(sum != 0 && "Some boundary vertex getting all zero BCs");
    bc.row(i).array() /= sum;
  }

  if(bc.size() == 0)
  {
    igl::verbose("^%s: Error: boundary conditions are empty.\n",__FUNCTION__);
    return false;
  }

  // If there's only a single boundary condition, the following tests
  // are overzealous.
  if(bc.cols() == 1)
  {
    // If there is only one weight function,
    // then we expect that there is only one handle.
    assert(P.rows() + BE.rows() == 1);
    return true;
  }

  // Check that every Weight function has at least one boundary value of 1 and
  // one value of 0
  for(i = 0;i<bc.cols();i++)
  {
    double min_abs_c = bc.col(i).array().abs().minCoeff();
    double max_c = bc.col(i).maxCoeff();
    if(min_abs_c > igl::FLOAT_EPS)
    {
      igl::verbose("^%s: Error: handle %d does not receive 0 weight\n",__FUNCTION__,i);
      return false;
    }
    if(max_c< (1-igl::FLOAT_EPS))
    {
      igl::verbose("^%s: Error: handle %d does not receive 1 weight\n",__FUNCTION__,i);
      return false;
    }
  }

  return true;
}


bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
      break;
    case '.':
      selected++;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      viewer.data().set_data(W.col(selected));
      break;
    case ',':
      selected--;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      viewer.data().set_data(W.col(selected));
      break;
  }
  return true;
}

Eigen::MatrixXd biskin(
  Eigen::MatrixXd V,
  Eigen::MatrixXi Ele,
  Eigen::MatrixXi BE,  // [n_bones, 2] 0-based
  Eigen::MatrixXi CE,  // [n_bones, n_sampled_joints] 0-based
  Eigen::VectorXi P
) {
  Eigen::MatrixXd bc;
  Eigen::VectorXi b;
  custom_boundary_conditions(V, P, BE, CE, b, bc);

  igl::BBWData data;
  Eigen::MatrixXd W;
  igl::bbw(V, Ele, b, bc, data, W);
  W = (W.array().colwise() / W.array().rowwise().sum()).eval();

  return W;
}

namespace py = pybind11;
PYBIND11_MODULE(biskin, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("biskin", &biskin);

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}

// BE = []
// for x, y in SKELETON:
//    if x in inds_to_map:
//        x = mapping[x]
//    if y in inds_to_map:
//        y = mapping[y]
//    BE.append((x, y))
// BE = np.array(BE, dtype="int")
// CE = np.array(sverts_inds, dtype="int")
// W = np.ndarray((len(JOINTS) + len(SKELETON), len(V)), dtype="float64")
// biskin(W, V, T, BE, CE, len(JOINTS))
// assert np.allclose(W.sum(0), 1)

// int main(int argc, char *argv[]) {
//   igl::readPLY(argv[1], V, Ele);
//   std::cout << V.rows() << " " << Ele.rows() << std::endl;
//   igl::readTGF(argv[2], C, BE);
//   P.setLinSpaced(C.rows(), 0, C.rows() - 1);
//   std::ifstream file(argv[3]);
//   if (file.is_open()) {
//     for (int i = 0; i < C.rows(); ++i) {
//       int item;
//       file >> item;
//       P(i) = item - 1;
//     }
//   }
//   std::ifstream file2(argv[4]);
//   if (file2.is_open()) {
//     int n;
//     file2 >> n;
//     CE.resize(BE.rows(), n);
//     for (int i = 0; i < BE.rows(); ++i) {
//       for (int j = 0; j < n; ++j) {
//         int item;
//         file2 >> item;
//         CE(i, j) = item - 1;
//       }
//     }
//   }
// 
//   std::cout 
//     << "C: "<< C.rows()
//     << " " << C.cols()
//     << std::endl
//     << "BE: " << BE.rows()
//     << " " << BE.cols()
//     << std::endl
//     << "CE: " << CE.rows()
//     << " " << CE.cols()
//     << std::endl;
//   // igl::boundary_conditions(V, Ele, C, P, BE, CE, b, bc);
//   //igl::boundary_conditions(V, Ele, C, P, BE, CE, CF, b, bc);
//   custom_boundary_conditions(V, P, BE, CE, b, bc);
//   std::cout << bc.rows() << " " << bc.cols() << std::endl;
//   // std::cout << b << std::endl << bc << std::endl;
//   // std::cout << b.size() << std::endl << bc.size() << std::endl;
// 
//   igl::BBWData data;
//   // data.active_set_params.max_iter = 10000000000;
//   //data.verbosity = 2;
//   igl::bbw(V, Ele, b, bc, data, W);
//   //igl::normalize_row_sums(W, W);
//   W  = (W.array().colwise() / W.array().rowwise().sum()).eval();
//   std::cout << W.rows() << " " << W.cols() << std::endl;
// 
//   igl::lbs_matrix(V, W, M);
//   std::cout << M.rows() << " " << M.cols() << std::endl;
// 
//   igl::writeDMAT(argv[5], W);
//   igl::writeDMAT(argv[6], M);
// 
//   if (argc > 7) {
//     igl::opengl::glfw::Viewer viewer;
//   viewer.data().set_mesh(V, Ele);
//     viewer.data().set_data(W.col(selected));
//     viewer.data().set_edges(C, BE, sea_green);
//     viewer.data().show_lines = false;
//     viewer.data().show_overlay_depth = false;
//     viewer.data().line_width = 1;
//     //viewer.callback_pre_draw = &pre_draw;
//     viewer.callback_key_down = &key_down;
//     viewer.core().is_animating = false;
//     viewer.core().animation_max_fps = 30.;
//     std::cout <<
//       "Press '.' to show next weight function." << std::endl <<
//       "Press ',' to show previous weight function." << std::endl <<
//       "Press [space] to toggle animation." << std::endl;
//     viewer.launch();
//   }
// 
//   return 0;
// }
