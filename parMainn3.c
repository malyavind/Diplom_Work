#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include "MP/mmio.h"


void merge(int *a, int *a2, double *a3, int low, int mid, int high)
{
	// Variables declaration.
	int *b =  (int*)malloc(sizeof(int)*(high+1-low) );
	int *b2 = (int*)malloc(sizeof(int)*(high+1-low) );
	double *b3 = (double*)malloc(sizeof(double)*(high+1-low) );
	int h,i,j,k;
	h=low;
	i=0;
	j=mid+1;
	// Merges the two array's into b[] until the first one is finish
	while((h<=mid)&&(j<=high))
	{
		if(a[h]<=a[j])
		{
			b[i]=a[h];
			b2[i]=a2[h];
			b3[i]=a3[h];
			h++;
		}
		else
		{
			b[i]=a[j];
			b2[i]=a2[j];
			b3[i]=a3[j];
			j++;
		}
		i++;
	}
	// Completes the array filling in it the missing values
	if(h>mid)
	{
		for(k=j;k<=high;k++)
		{
			b[i]=a[k];
			b2[i]=a2[k];
			b3[i]=a3[k];
			i++;
		}
	}
	else
	{
		for(k=h;k<=mid;k++)
		{
			b[i]=a[k];
			b2[i]=a2[k];
			b3[i]=a3[k];
			i++;
		}
	}
	// Prints into the original array
	for(k=0;k<=high-low;k++)
	{
		a[k+low]=b[k];
		a2[k+low]=b2[k];
		a3[k+low]=b3[k];
	}
	free(b);
	free(b2);
	free(b3);
}

void merge_sort(int *a, int *a2, double* a3, int low, int high)
{// Recursive sort ...
	int mid;
	if(low<high)
	{
		mid=(low+high)/2;
		merge_sort(a, a2, a3, low,mid);
		merge_sort(a, a2, a3, mid+1,high);
		merge(a, a2, a3, low,mid,high);
	}
}
//#endif // CRS

double wtime()
{
    struct timeval t;
    gettimeofday( &t, NULL);
    return ( double )t.tv_sec + ( double )t.tv_usec * 1E-6;
}

int main( int argc, char *argv[] )
{
    int         ret_code, prevRow, oi = 0, i, j, t, M, N, nz;
    int         *rows, *cols, *rowOffset;
    double      *val, *vect, *result, *result2, *result3, *timeresult, time, resTime, flops;
    MM_typecode matcode;

///_______________________________________________ open file and checks
   
mm_read_unsymmetric_sparse("TSOPF_RS_b678_c2.mtx", &N, &M, &nz, &val, &rows, &cols);
//mm_read_unsymmetric_sparse("sinc18.mtx", &N, &M, &nz, &val, &rows, &cols);
//mm_read_unsymmetric_sparse("TSOPF_RS_b2383.mtx", &N, &M, &nz, &val, &rows, &cols);
//mm_read_unsymmetric_sparse("largebasis.mtx", &N, &M, &nz, &val, &rows, &cols);
///_______________________________________________ open file and checks



///______________________________________________________________ memory allocation
    result  =   ( double* )   calloc( M, sizeof( double ) );
    result2  =   ( double* )   calloc( M, sizeof( double ) );
    result3  =   ( double* )   calloc( M, sizeof( double ) );
    timeresult = ( double* )   calloc( omp_get_max_threads(), sizeof( double ) );
    rowOffset=  ( int* )      malloc( (N + 1 )* sizeof( int ) );
    vect    =   ( double* )   malloc( sizeof( double ) * nz );
    for (i = 0; i < nz; i++)
        vect[ i ] = -1.0;

///_______________________________________________________________ memory allocation

///__________________________________________________________________ compress

    merge_sort(rows, cols, val, 0, nz);


    
    for (i = 0; i < N; i++)
        result2[i] = 0;
    for (i = 0; i < nz; i++){
        result2[rows[i]] += val[i] * vect[cols[i]];
    }

    for (i = 0; i < N; i++)
        result3[i] = 0;
    for (i = 0; i < nz; i++){
        result3[cols[i]] += val[i] * vect[cols[i]];
	}	
  
 	rowOffset[0] = 0;
	int row = 0;
	int num, num2;
	for (i = 1; i < nz; i++){
		if (rows[i] != rows[i-1]){
			row++;
			rowOffset[row] = i;
			num = rows[i] - rows[i-1];
			if (num > 1){
				num2 = row + num - 1;
				for (j = row; j < num2; j++){
					row++;
					rowOffset[row] = i;
				}
			}
		}
	}	
	row++;		
	rowOffset[row] = i;

    ///______________________________________________________________________ compress
    ///______________________________________________________________________ distribute
    //___________________________________ distribut

    //_____________ MULTIPLICATION CRS

    int thr = 1;
    for(; thr <= omp_get_max_threads(); thr++){
        printf("\n%d -  ", thr);
	    time =  wtime();
	    int j, t;
	    for(t = 0; t < 100; t++){
	        for (i = 0; i < N; i++)
	            result[i] = 0;
	        #pragma omp parallel for private(i, j) schedule (dynamic, 30) num_threads(thr)
	        for(i = 0; i < N; i++ ){
	            int rowstart = rowOffset[i];
	            int rowend = rowOffset [i+1];
	            double r = 0.0; 
    		    for(j = rowstart; j < rowend; j++)
        		    r += val[j] * vect[cols[j]];
        		result[i] = r;
	        }
	    }	    
	    timeresult[thr] = wtime() - time;
        printf("Multiplic  %.10lf\n", timeresult[thr] / 100);
        flops = (2 * nz + 3 * N + 1) / (timeresult[thr] / 100);
        printf("flops = %.2lf\n", flops);
    }
    
    printf("\n");
     for(i = 1; i <= omp_get_max_threads(); i++)
        printf ("%d - speedup %.10lf\n", i, 1 / (timeresult[i] / timeresult[1]));
        

    double validation = 0, control = 0;
    for (i = 0; i < N; i++){
        validation += fabs(result2[i]) - fabs(result[i]);
        control += fabs (result2[i]);
    }    
    validation = validation / control;
    printf ("Validation: %.20lf\n", validation);

    
    thr = 1;
    for(; thr <= omp_get_max_threads(); thr++){
        printf("\n%d -  ", thr);
        time = wtime();
        for(t = 0; t < 100; t++){
            for (i = 0; i < N; i++)
                result[i] = 0.0;
            #pragma omp parallel for private(i, j) schedule (dynamic, 30) num_threads(thr)
          
	        for(i = 0; i < N; i++ ){
	            int rowstart = rowOffset[i];
	            int rowend = rowOffset [i+1];    
    		    for(j = rowstart; j < rowend; j++){
    		        #pragma omp atomic
    		    	result[cols[j]] += val[j] * vect[cols[j]];
				}
			}
		}           
        timeresult[thr] = wtime() - time;
        printf("TransMultiplic  %.10lf\n", timeresult[thr] / 100);
        flops = (2 * nz + 3 * N + 1) / (timeresult[thr] / 100);
		printf("flops = %.2lf\n", flops);
	}	
    
    printf("\n");
    for(i = 1; i <= omp_get_max_threads(); i++)
        printf ("%d - speedup %.10lf\n", i, 1 / (timeresult[i] / timeresult[1]));

    for (i = 0; i < N; i++){
        validation += fabs(result3[i]) - fabs(result[i]);
        control += fabs (result3[i]);
    }    
    validation = validation / control;
    printf ("ValidationTrans: %.20lf\n", validation);
    

	return 0;
}
