/*
*   compute a SDF grid on a GPU
*   by R. Falque
*   09/03/2020
*/

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_GLOBAL
#endif


#ifndef COMPUTE_SDF_H
#define COMPUTE_SDF_H

#include<vector>
#include "cudaTools/cutil_math.h"

class SDF
{
    private:
        // inputs
        float *vertices;
        int   *faces;
        int   number_of_vertices;
        int   number_of_faces;
        int   grid_resolution;

        // store the normals as well
        float3 *normal_faces;
        float3 *normal_vertices;
        float3 *normal_edges;

        // variables for internal computations
        float *min_corner;
        float *max_corner;
        float grid_size;

        // tensor for SDF
        float *sdf;
        int   D1, D2, D3;

        // private functions
        void init();
        void set_bounding_box_corners();
        void kernel_entry_to_compute_sdf(int blockSize, int numBlocks); // <- this is the 

    public:
        SDF(float *V, int V_size, int *F, int F_size, int grid_res);
        
        ~SDF() {
        };

        void get_tensor_size (int &Dx, int &Dy, int &Dz);

        void get_tensor(float *tensor);

        void clear_memory();

};

#endif