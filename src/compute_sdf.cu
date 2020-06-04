#include <iostream>
#include <math.h>
#include <algorithm>
#include <initializer_list> // for std::max of 3 numbers
#include "compute_sdf.h"

#include "cudaTools/cutil_math.h"

__device__ float dot2( float3 v ) {return dot(v,v);}
__device__ int sign(float x) {return x > 0 ? 1 : -1;}
__host__ __device__ bool operator==(float3 & a, float3 & b) {return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);}


__global__
void compute_sdf(float *V, int V_size, 
                 int *F, int F_size,
                 float *sdf, int D1, int D2, int D3, float grid_size, float *min_corner);

__device__
float3 closestTriangle( float3 v0, float3 v1, float3 v2, float3 p );

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
    int blockSize = 512;
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

    // memory check
    size_t free; size_t total;
    
    cudaMemGetInfo ( &free, &total );
    printf("free GPU memory: %lu = \n", free);
    printf("total GPU memory: %lu = \n", total);
    printf("tensor size: %lu = \n", D1 * D2 * D3 * sizeof(float));
    if (free < 0.2 * total)
        printf("error, not enough memory");
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

    int index = blockIdx.x * blockDim.x + threadIdx.x ;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < D1 * D2 * D3; i+=stride)
    {
        // define the query point
        int x_id = i % D1;
        int y_id = (i/D1) % D2;
        int z_id = (i/D1/D2) % D3;

        float x = float(x_id) * grid_size + min_corner[0];
        float y = float(y_id) * grid_size + min_corner[1];
        float z = float(z_id) * grid_size + min_corner[2];
        float3 query_point = make_float3(x, y, z);

        // ugly part of the code used as temp variable
        float distance_to_mesh = 9999999999999; // start from a large distance, should be inf
        float3 closest_point_on_mesh;

        // build a 2D array for the normal
        int normal_counter = 0;
        float3 normal_register[20]; // lazy initialization

        for (size_t j = 0; j < F_size; j++)
        {

            // define tested triangle with face `f` and points (`a`, `b`, and `c`)
            int3 f = make_int3 ( F[j], F[j + F_size], F[j + 2*F_size] );

            float3 a = make_float3(V[f.x], V[f.x + V_size], V[f.x + 2*V_size]);
            float3 b = make_float3(V[f.y], V[f.y + V_size], V[f.y + 2*V_size]);
            float3 c = make_float3(V[f.z], V[f.z + V_size], V[f.z + 2*V_size]);

            // closest point on the triangle:
            float3 closest_point_on_triangle = closestTriangle( query_point, a, b, c );
            float distance_to_triangle = abs(dot2(query_point - closest_point_on_triangle));

            // check if the distance need to be updated
            if (distance_to_triangle < distance_to_mesh) {
                closest_point_on_mesh = closest_point_on_triangle;
                distance_to_mesh = distance_to_triangle;

                normal_counter = 0;
                normal_register[normal_counter] = normalize(cross( b-a, c-a ));
                normal_counter ++;
            } else if ( closest_point_on_triangle == closest_point_on_mesh) {
                normal_register[normal_counter] = normalize(cross( b-a, c-a ));
                normal_counter ++;
            }

        }

        // process the normal stack
        float3 normal_of_closest_point = make_float3(0.0, 0.0, 0.0);
        int inside_out = 1;  // positive numbers are inside, negatice numbers are outside
        for (size_t normal_id = 0; normal_id < normal_counter; normal_id++)
            normal_of_closest_point += normal_register[normal_id];

        if (sign( dot(normal_of_closest_point, closest_point_on_mesh-query_point) ) == -1)
            inside_out = -1;

        sdf[i] = distance_to_mesh * inside_out;
    }
    
}

// see https://www.shadertoy.com/view/ttfGWl
__device__
float3 closestTriangle( float3 v0, float3 v1, float3 v2, float3 p )
{
    float3 v10 = v1 - v0; float3 p0 = p - v0;
    float3 v21 = v2 - v1; float3 p1 = p - v1;
    float3 v02 = v0 - v2; float3 p2 = p - v2;
    float3 nor = cross( v10, v02 );
  
    // method of barycentric space
    float3  q = cross( nor, p0 );
    float d = 1.0/dot2(nor);
    float u = d*dot( q, v02 );
    float v = d*dot( q, v10 );
    float w = 1.0-u-v;
    
         if( u<0.0 ) { w = clamp( dot(p2,v02)/dot2(v02), 0.0, 1.0 ); u = 0.0; v = 1.0-w; }
    else if( v<0.0 ) { u = clamp( dot(p0,v10)/dot2(v10), 0.0, 1.0 ); v = 0.0; w = 1.0-u; }
	else if( w<0.0 ) { v = clamp( dot(p1,v21)/dot2(v21), 0.0, 1.0 ); w = 0.0; u = 1.0-v; }
    
    return u*v1 + v*v2 + w*v0;  
}
