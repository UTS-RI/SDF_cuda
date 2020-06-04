#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "IO/readPLY.h"
#include "IO/writePLY.h"

#include "compute_sdf.h"
#include "print_tensor.h"
#include "polyscope/polyscope.h"

int main() {
    bool visualization = true;
    int grid_resolution = 100;    // The code goes nuts if the resolution is too high, this is probably a memory issue
    double bounding_box_scale = 1;

    // IO: load files
    std::cout << "Progress: load data\n";
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd N;
    Eigen::MatrixXi RGB;
    readPLY("../data/Lucy100k.ply", V, F, N, RGB);
    //readPLY("../data/david.ply", V, F, N, RGB);
    std::cout << "Progress: data loaded\n";

    // put it into rowWise
    Eigen::MatrixXf Vf;
    Eigen::MatrixXi Ff;
    Vf = V.transpose().cast <float> ();
    Ff = F.transpose();
    std::cout << "size of V: [" << Vf.rows() << ", " << Vf.cols() << "]\n";
    std::cout << "size of F: [" << Ff.rows() << ", " << Ff.cols() << "]\n";

    // get the sdf
    SDF sdf(Vf.data(), Vf.rows(), Ff.data(), Ff.rows(), grid_resolution);

    std::cout << "Progress: class created\n";

    int Dx, Dy, Dz;
    sdf.get_tensor_size(Dx, Dy, Dz);
    std::cout << "Tensor size: \n";
    std::cout << "Dx: " << Dx << "\n";
    std::cout << "Dy: " << Dy << "\n";
    std::cout << "Dz: " << Dz << "\n";

    float* sdf_array = (float*) malloc( Dx * Dy * Dz * sizeof(float));
    sdf.get_tensor(sdf_array);

    std::cout << "Progress: array collected\n";
    
    // transform back to eigen tensor (this is not the smartest way to do it, probably some cast thingy would be better)
    Eigen::Tensor<float, 3> sdf_tensor;
    sdf_tensor.resize(Dx, Dy, Dz);
    for (int x = 0; x < Dx; x++)
        for (int y = 0; y < Dy; y++)
            for (int z = 0; z < Dz; z++)
                sdf_tensor(x, y, z) = sdf_array[x + y*Dx + z*Dx*Dy];
    
    print_to_folder("../data/image_stack/", sdf_tensor);

    free(sdf_array);
    sdf.clear_memory();

}
