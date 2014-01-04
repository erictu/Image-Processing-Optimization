#include <emmintrin.h>
#include <stdio.h>
#include <string.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.



int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    // creating a padded matrix
    int matrix_size = ((data_size_X+2)*(data_size_Y+2));
    float padded_in[matrix_size];
    memset(padded_in, 0.0, 4*matrix_size);
    for(int q = 0; q < data_size_Y; q++){
        for(int w = 0; w < data_size_X; w++){
            padded_in[((q+1)*(data_size_X+2)+w+1)]  = in[(q*(data_size_X)+w)];
        }
    }
    // rotating kernel
    int length = KERNX*KERNY;
    float r_kernel[KERNX*KERNY];
    for(int i=0; i<length; i++){
        r_kernel[length-1-i] = kernel[i];
    }

    __m128 kernel_vector;
    __m128 row_vector;
    __m128 total_vector;
    float* vector_offset;

    int x,y,i,j;
    for( y=0; y<(data_size_Y); y++){
        for( x=0; x<=4*(data_size_X/4)-4; x+=4){
            total_vector = _mm_setzero_ps();
            for( i=0; i<length; i++){
                kernel_vector = _mm_load1_ps(r_kernel+i);
                row_vector = _mm_loadu_ps(padded_in + ((x+1+i%KERNX-kern_cent_X ) + (y+1+i/KERNX-kern_cent_Y)*(data_size_X+2)));
                total_vector = _mm_add_ps(total_vector, _mm_mul_ps(kernel_vector, row_vector));
            }
            _mm_storeu_ps(out+(x+y*data_size_X), total_vector);
        }
        for (x=4*(data_size_X/4); x< data_size_X; x++) {
        	for( i=0; i<length; i++){
        		out[x+y*data_size_X] += *(r_kernel+i) * *(padded_in+((x+1+i%KERNX-kern_cent_X ) + (y+1+i/KERNX-kern_cent_Y)*(data_size_X+2)));
        	}
        }
    }
	return 1;
}
