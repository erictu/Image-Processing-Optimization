#include <emmintrin.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    // creating a padded matrix
    int matrix_size = ((data_size_X+2)*(data_size_Y+2));
    float padded_in[matrix_size];

    #pragma omp parallel for firstprivate(matrix_size)
    for(int i=0; i<matrix_size; i++){
        padded_in[i] = 0.0;
    }
    #pragma omp parallel for firstprivate(data_size_X, data_size_Y)
    for(int q = 0; q < data_size_Y; q++){
        for(int w = 0; w < data_size_X; w++){
            *(padded_in+((q+1)*(data_size_X+2)+w+1)) = *(in+(q*(data_size_X)+w));
        }
    }

    // rotating kernel
    int length = KERNX*KERNY;
    float r_kernel[KERNX*KERNY];
    for(int i=0; i<length; i++){
        r_kernel[length-1-i] = kernel[i];
    }


    // main convolution loop
    int padded_x_size = data_size_X+2;
    #pragma omp parallel for firstprivate(padded_x_size, r_kernel, out)
    for(int y = 0; y < data_size_Y; y++){ // the x coordinate of the output location we're focusing on
        for(int x=0; x<=16*(data_size_X/16)-16; x+=16/*int x = 0; x < data_size_X; x++*/){ // the y coordinate of theoutput location we're focusing on
            // int i = 0;

            __m128 total_vector = _mm_setzero_ps();
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+0), _mm_loadu_ps(padded_in + ((x ) + (y)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+1), _mm_loadu_ps(padded_in + ((x+1 ) + (y)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+2), _mm_loadu_ps(padded_in + ((x+2 ) + (y)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+3), _mm_loadu_ps(padded_in + ((x ) + (y+1)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+4), _mm_loadu_ps(padded_in + ((x+1 ) + (y+1)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+5), _mm_loadu_ps(padded_in + ((x+2 ) + (y+1)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+6), _mm_loadu_ps(padded_in + ((x ) + (y+2)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+7), _mm_loadu_ps(padded_in + ((x+1 ) + (y+2)*(padded_x_size)))));
            total_vector = _mm_add_ps(total_vector, _mm_mul_ps(_mm_load1_ps(r_kernel+8), _mm_loadu_ps(padded_in + ((x+2 ) + (y+2)*(padded_x_size)))));

            __m128 total_vector1 = _mm_setzero_ps();
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+0), _mm_loadu_ps(padded_in + ((x+4 ) + (y)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+1), _mm_loadu_ps(padded_in + ((x+5 ) + (y)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+2), _mm_loadu_ps(padded_in + ((x+6 ) + (y)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+3), _mm_loadu_ps(padded_in + ((x+4 ) + (y+1)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+4), _mm_loadu_ps(padded_in + ((x+5 ) + (y+1)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+5), _mm_loadu_ps(padded_in + ((x+6 ) + (y+1)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+6), _mm_loadu_ps(padded_in + ((x+4 ) + (y+2)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+7), _mm_loadu_ps(padded_in + ((x+5 ) + (y+2)*(padded_x_size)))));
            total_vector1 = _mm_add_ps(total_vector1, _mm_mul_ps(_mm_load1_ps(r_kernel+8), _mm_loadu_ps(padded_in + ((x+6 ) + (y+2)*(padded_x_size)))));

            __m128 total_vector2 = _mm_setzero_ps();
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+0), _mm_loadu_ps(padded_in + ((x+8 ) + (y)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+1), _mm_loadu_ps(padded_in + ((x+9 ) + (y)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+2), _mm_loadu_ps(padded_in + ((x+10 ) + (y)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+3), _mm_loadu_ps(padded_in + ((x+8 ) + (y+1)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+4), _mm_loadu_ps(padded_in + ((x+9 ) + (y+1)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+5), _mm_loadu_ps(padded_in + ((x+10 ) + (y+1)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+6), _mm_loadu_ps(padded_in + ((x+8 ) + (y+2)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+7), _mm_loadu_ps(padded_in + ((x+9 ) + (y+2)*(padded_x_size)))));
            total_vector2 = _mm_add_ps(total_vector2, _mm_mul_ps(_mm_load1_ps(r_kernel+8), _mm_loadu_ps(padded_in + ((x+10 ) + (y+2)*(padded_x_size)))));

            __m128 total_vector3 = _mm_setzero_ps();
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+0), _mm_loadu_ps(padded_in + ((x+12 ) + (y)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+1), _mm_loadu_ps(padded_in + ((x+13 ) + (y)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+2), _mm_loadu_ps(padded_in + ((x+14 ) + (y)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+3), _mm_loadu_ps(padded_in + ((x+12 ) + (y+1)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+4), _mm_loadu_ps(padded_in + ((x+13 ) + (y+1)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+5), _mm_loadu_ps(padded_in + ((x+14 ) + (y+1)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+6), _mm_loadu_ps(padded_in + ((x+12 ) + (y+2)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+7), _mm_loadu_ps(padded_in + ((x+13 ) + (y+2)*(padded_x_size)))));
            total_vector3 = _mm_add_ps(total_vector3, _mm_mul_ps(_mm_load1_ps(r_kernel+8), _mm_loadu_ps(padded_in + ((x+14 ) + (y+2)*(padded_x_size)))));


            _mm_storeu_ps(out+(x+y*data_size_X), total_vector);
            _mm_storeu_ps(out+(x+4+y*data_size_X), total_vector1);
            _mm_storeu_ps(out+(x+8+y*data_size_X), total_vector2);
            _mm_storeu_ps(out+(x+12+y*data_size_X), total_vector3);
		}
	}

    #pragma omp parallel for firstprivate(kern_cent_X, kern_cent_Y, data_size_X, r_kernel)
    for(int y = 0; y < data_size_Y; y++){
        for (int x=16*(data_size_X/16); x< data_size_X; x++) {
            out[x+y*data_size_X] += *(r_kernel) * *(padded_in+((x+1+0%KERNX-kern_cent_X ) + (y+1+0/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+1) * *(padded_in+((x+1+1%KERNX-kern_cent_X ) + (y+1+1/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+2) * *(padded_in+((x+1+2%KERNX-kern_cent_X ) + (y+1+2/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+3) * *(padded_in+((x+1+3%KERNX-kern_cent_X ) + (y+1+3/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+4) * *(padded_in+((x+1+4%KERNX-kern_cent_X ) + (y+1+4/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+5) * *(padded_in+((x+1+5%KERNX-kern_cent_X ) + (y+1+5/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+6) * *(padded_in+((x+1+6%KERNX-kern_cent_X ) + (y+1+6/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+7) * *(padded_in+((x+1+7%KERNX-kern_cent_X ) + (y+1+7/KERNX-kern_cent_Y)*(data_size_X+2)));
            out[x+y*data_size_X] += *(r_kernel+8) * *(padded_in+((x+1+8%KERNX-kern_cent_X ) + (y+1+8/KERNX-kern_cent_Y)*(data_size_X+2)));
    }
    }
	return 1;
}
