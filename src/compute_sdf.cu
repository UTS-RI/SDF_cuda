#include <iostream>
#include <math.h>
#include <algorithm>
#include <initializer_list> // for std::max of 3 numbers
#include "compute_sdf.h"

#include "cudaTools/cutil_math.h"


__global__
void compute_sdf(float *V, int V_size, 
                 int *F, int F_size,
                 float *sdf, int D1, int D2, int D3, float grid_size, float *min_corner);

__device__
float udTriangle( float3 p, float3 a, float3 b, float3 c );


SDF::SDF(float *V, int V_size, int *F, int F_size, int grid_res) {
    this->number_of_vertices = V_size;
    this->number_of_faces = F_size;
    this->grid_resolution = grid_res;

    std::cout << "Progress: Copy the mesh into memory\n";
    cudaMallocManaged(&this->vertices, V_size*3*sizeof(float));
    cudaMemcpy(this->vertices, V, V_size*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaMallocManaged(&this->faces, F_size*3*sizeof(int));
    cudaMemcpy(this->faces,    F, F_size*3*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    std::cout << "Progress: Compute the bounding box info\n";
    cudaMallocManaged(&this->min_corner, 3*sizeof(float));
    cudaMallocManaged(&this->max_corner, 3*sizeof(float));

    init();

    // this is the part that define the parallelization parameters
    int blockSize = 256;
    int numBlocks = (this->D1*this->D2*this->D3 + blockSize -1) / blockSize;
    kernel_entry_to_compute_sdf(blockSize, numBlocks);
    
    cudaDeviceSynchronize();
}


void SDF::init() {
    set_bounding_box_corners();

    float bounding_box_size = std::max( {this->max_corner[0] - this->min_corner[0], 
                                         this->max_corner[1] - this->min_corner[1], 
                                         this->max_corner[2] - this->min_corner[2]} );
    
    this->grid_size = bounding_box_size / (float(this->grid_resolution)-1);

    this->D1 = floor(this->max_corner[0] / this->grid_size) - floor(this->min_corner[0] / this->grid_size) +1;
    this->D2 = floor(this->max_corner[1] / this->grid_size) - floor(this->min_corner[1] / this->grid_size) +1;
    this->D3 = floor(this->max_corner[2] / this->grid_size) - floor(this->min_corner[2] / this->grid_size) +1;

    cudaMallocManaged(&this->sdf, D1 * D2 * D3 * sizeof(float));

}


void SDF::set_bounding_box_corners() {
    // lazy initialization, set V.row(0) as min and max
    min_corner[0] = this->vertices[0 + 0*this->number_of_vertices];
    min_corner[1] = this->vertices[0 + 1*this->number_of_vertices];
    min_corner[2] = this->vertices[0 + 2*this->number_of_vertices];
    
    max_corner[0] = this->vertices[0 + 0*this->number_of_vertices];
    max_corner[1] = this->vertices[0 + 1*this->number_of_vertices];
    max_corner[2] = this->vertices[0 + 2*this->number_of_vertices];
    
    // loop through Vertices
    for (size_t i = 0; i < this->number_of_vertices; i++) {
        min_corner[0] = min(min_corner[0], this->vertices[i]);
        max_corner[0] = max(max_corner[0], this->vertices[i]);
    
        min_corner[1] = min(min_corner[1], this->vertices[i + this->number_of_vertices]);
        max_corner[1] = max(max_corner[1], this->vertices[i + this->number_of_vertices]);
    
        min_corner[2] = min(min_corner[2], this->vertices[i + 2*this->number_of_vertices]);
        max_corner[2] = max(max_corner[2], this->vertices[i + 2*this->number_of_vertices]);
    }

}


void SDF::kernel_entry_to_compute_sdf(int blockSize, int numBlocks) {
    compute_sdf<<<numBlocks, blockSize>>>(vertices, number_of_vertices, 
                                          faces, number_of_faces, 
                                          sdf, D1, D2, D3, grid_size, min_corner);
    cudaDeviceSynchronize();
}


// public functions
void SDF::get_tensor_size (int &Dx, int &Dy, int &Dz) {
    Dx = this->D1;
    Dy = this->D2;
    Dz = this->D3;
}


void SDF::get_tensor(float *sdf_out) {
    cudaMemcpy(sdf_out, this->sdf, this->D1*this->D2*this->D3*sizeof(float), cudaMemcpyDeviceToHost);
}


void SDF::clear_memory() {
    cudaFree(this->vertices);
    cudaFree(this->faces);
    cudaFree(this->min_corner);
    cudaFree(this->max_corner);
    cudaFree(this->sdf);
}

__global__
void compute_sdf(float *V, int V_size, 
                 int *F, int F_size,
                 float *sdf, int D1, int D2, int D3, float grid_size, float *min_corner) {
    float x, y, z;

    int index = blockIdx.x * blockDim.x + threadIdx.x ;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < D1 * D2 * D3; i+=stride)
    {
        int x_id = i % D1;
        int y_id = (i/D1) % D2;
        int z_id = (i/D1/D2) % D3;

        float distance = 9999999999999; // start from a big distance
        for (size_t j = 0; j < F_size; j++)
        {
            x = float(x_id) * grid_size + min_corner[0];
            y = float(y_id) * grid_size + min_corner[1];
            z = float(z_id) * grid_size + min_corner[2];

            float3 p = make_float3(x, y, z);

            int3 f = make_int3 ( F[j], F[j + F_size], F[j + 2*F_size] );

            float3 a = make_float3(V[f.x], V[f.x + V_size], V[f.x + 2*V_size]);
            float3 b = make_float3(V[f.y], V[f.y + V_size], V[f.y + 2*V_size]);
            float3 c = make_float3(V[f.z], V[f.z + V_size], V[f.z + 2*V_size]);


            float distance_to_face;
            distance_to_face = udTriangle(p, a, b, c );

            if (abs(distance_to_face) < abs(distance) )
                distance = distance_to_face;

        }
        
        sdf[i] = distance;
    }
}

__device__
float dot2( float3 v ) {return dot(v,v);}

__device__ int sign(float x)
{
    return x > 0 ? 1 : -1;
}

// based on https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
// need to make it signed ...
__device__
float udTriangle( float3 p, float3 a, float3 b, float3 c )
{
  float3 ba = b - a; float3 pa = p - a;
  float3 cb = c - b; float3 pb = p - b;
  float3 ac = a - c; float3 pc = p - c;
  float3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}

