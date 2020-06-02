#ifndef IO_WRITEPLY_H
#define IO_WRITEPLY_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <Eigen/Core>

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "structures.h"

//using namespace tinyply;

void writePLY(const std::string & filepath,
              Eigen::MatrixXd &V,
              Eigen::MatrixXi &F,
              Eigen::MatrixXd &N,
              Eigen::MatrixXi &RGB,
              bool write_in_ascii)
{
    bool verbose = false;

    try
	{
        tinyply::PlyFile object_to_write;
        tinyply::geometry cube;

        if (V.cols() != 0) {
            for (int i=0; i<V.cols(); i++)
                cube.vertices.push_back({float(V(0,i)), float(V(1,i)), float(V(2,i))});
            object_to_write.add_properties_to_element("vertex", { "x", "y", "z" }, 
                tinyply::Type::FLOAT32, cube.vertices.size(), reinterpret_cast<uint8_t*>(cube.vertices.data()), tinyply::Type::INVALID, 0);
        } else {
            throw std::invalid_argument( "asked to write a set of empty vertices" );
        }
        
        // add normals
        if (N.cols() == V.cols()) {
            for (int i=0; i<N.cols(); i++)
                cube.normals.push_back({float(N(0,i)), float(N(1,i)), float(N(2,i))});

            object_to_write.add_properties_to_element("vertex", { "nx", "ny", "nz" },
                tinyply::Type::FLOAT32, cube.normals.size(), reinterpret_cast<uint8_t*>(cube.normals.data()), tinyply::Type::INVALID, 0);
        }

        // add color on vertices
        if (RGB.cols() == V.cols()) {
            for (int i=0; i<RGB.cols(); i++)
                cube.rgb.push_back({uint8_t(RGB(0,i)), uint8_t(RGB(1,i)), uint8_t(RGB(2,i))});

            object_to_write.add_properties_to_element("vertex", { "red", "green", "blue" },
                tinyply::Type::UINT8, cube.rgb.size() , reinterpret_cast<uint8_t*>(cube.rgb.data()), tinyply::Type::INVALID, 0);
        }

        // add faces as an output
        if (F.cols() != 0) {
            for (int i=0; i<F.cols(); i++)
                cube.triangles.push_back({uint32_t(F(0,i)), uint32_t(F(1,i)), uint32_t(F(2,i))});
            
            object_to_write.add_properties_to_element("face", { "vertex_indices" },
                tinyply::Type::UINT32, cube.triangles.size(), reinterpret_cast<uint8_t*>(cube.triangles.data()), tinyply::Type::UINT8, 3);
        }

        // add comment
        object_to_write.get_comments().push_back("generated by tinyply 2.2");

        if (write_in_ascii) {
            std::filebuf fb_ascii;
            fb_ascii.open(filepath, std::ios::out);
            std::ostream outstream_ascii(&fb_ascii);
            if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + filepath);

            object_to_write.write(outstream_ascii, false);
        } else {
            std::filebuf fb_binary;
            fb_binary.open(filepath, std::ios::out | std::ios::binary);
            std::ostream outstream_binary(&fb_binary);
            if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filepath);
            
            object_to_write.write(outstream_binary, true);
        }

	}
	catch (const std::exception & e)
	{
		std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
	}
}

void writePLY(const std::string & filepath,
             Eigen::MatrixXd &V,
             Eigen::MatrixXi &F,
             Eigen::MatrixXd &N,
             bool write_in_ascii)
{
    Eigen::MatrixXi RGB;
    writePLY(filepath, V, F, N, RGB, write_in_ascii);
}

void writePLY(const std::string & filepath,
             Eigen::MatrixXd &V,
             Eigen::MatrixXi &F,
             bool write_in_ascii)
{
    Eigen::MatrixXd N;
    Eigen::MatrixXi RGB;
    writePLY(filepath, V, F, N, RGB, write_in_ascii);
}

void writePLY(const std::string & filepath,
             Eigen::MatrixXd &V,
             Eigen::MatrixXd &N,
             bool write_in_ascii)
{
    Eigen::MatrixXi F;
    Eigen::MatrixXi RGB;
    writePLY(filepath, V, F, N, RGB,write_in_ascii);
}

#endif