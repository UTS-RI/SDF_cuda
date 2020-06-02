/*
*   print a tensor into a stack of images
*   by R. Falque
*   07/02/2020
*/

#ifndef PRINT_TENSOR_H
#define PRINT_TENSOR_H


#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "IO/writePNG.h"

#include "IO/stb_image_write.h"
#include "IO/process_folder.h"

inline bool print_to_folder(std::string folder_name, Eigen::Tensor<float, 3> tensor)
{


    bool folder_exist = does_folder_exist(folder_name);
    if (!folder_exist) {
        std::cout << "Error: the folder does not exist\n";
        create_folder(folder_name);
    }
    
    empty_folder(folder_name);


    std::cout << "tensor.dimension(0): " << tensor.dimension(0) << "\n";
    std::cout << "tensor.dimension(1): " << tensor.dimension(1) << "\n";
    std::cout << "tensor.dimension(2): " << tensor.dimension(2) << "\n";

    Eigen::MatrixXf R, G, B;
    R = Eigen::MatrixXf::Zero(tensor.dimension(0), tensor.dimension(1));
    G = R;
    B = R;

    for (int i=0; i<tensor.dimension(2); i++) {  
    
        for (size_t j = 0; j < tensor.dimension(0); j++)
        {
            for (size_t k = 0; k < tensor.dimension(1); k++)
            {
                R(j, k) = tensor(j,k,i);
                G(j, k) = -tensor(j,k,i);
            }
        }

        R = R.cwiseMax(0);
        R /= R.maxCoeff();

        G = G.cwiseMax(0);
        G /= G.maxCoeff();
        B = G;
       
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << i;
        std::string s = ss.str();

        Eigen::MatrixXd Rd, Gd, Bd;
        Rd = R.cast<double>();
        Gd = G.cast<double>();
        Bd = B.cast<double>();

        std::string file_name = folder_name + s + ".png";
        writePNG(Rd, Gd, Bd, file_name);

    
    }

    std::cout << "Progress: Stack of images written in :" << folder_name << std::endl;

    return true;
};

#endif