#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "utils.h"
#include "vector.h"
#include "semblance.h"
#include "su.h"
#include <errno.h>
#include <CL/opencl.h>
//#include <CL/cl.h>
//#include <utils.h>

#define MAXSOURCE 22048
#define MAX_DEVICE_NAME_SIZE 100

void print_opencl_error(FILE* fh, cl_int err);


cl_device_id theChosenDevice;





/*
 * compute_max finds the best parameters 'Aopt', 'Bopt', 'Copt', 'Dopt' and 'Eopt'
 * that fit a curve to the data in 'ap' from a reference point (m0, h0, t0). Also
 * returning its fit (coherence/semblance) through 'sem' and the average of values
 * along the curve through 'stack'
 *
 * The lower limit for searching each parameter is specified as a element in the
 * vector 'n0' and the upper limit in vector 'n1', the number of divisions for
 * the search space is specified through 'np'
 */ //compute_max(&ap, m0, h0, t0, p0, p1, np, &a, &b, &c, &d, &e, &sem, &stack);
void compute_max(aperture_t *ap, float m0, float h0, float t0,
    const float n0[5], const float n1[5], const int np[5], float *Aopt,
    float *Bopt, float *Copt, float *Dopt, float *Eopt, float *sem,
    float *stack)
{

    /* The parallel version of the code will compute the best parameters for
     * each value of the parameter 'A', so we need to store np[0] different
     * values of each parameter, stack and semblance */
    float _Aopt[np[0]], _Bopt[np[0]], _Copt[np[0]],
          _Dopt[np[0]], _Eopt[np[0]];
    float smax[np[0]];
    float _stack[np[0]];


    float range_A = n1[0] - n0[0];
    float range_B = n1[1] - n0[1];
    float range_C = n1[2] - n0[2];
    float range_D = n1[3] - n0[3];
    float range_E = n1[4] - n0[4];

    float a[np[0]];
    float b[np[1]];
    float c[np[2]];
    float d[np[3]];
    float e[np[4]];

    cl_mem *d_a;
    cl_mem *d_b;
    cl_mem *d_c;
    cl_mem *d_d;
    cl_mem *d_e;

    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

   
   

    char *kernelSource = (char *) malloc(MAXSOURCE*sizeof(char));
    FILE * file = fopen("kernels.cl","r");
	if(file == NULL)
	{
		printf("Error: open the kernel file (vec-add-opencl-kernel.cl)\n");
		exit(1);
	}

     // Read kernel code
	size_t source_size = fread(kernelSource, 1, MAXSOURCE, file);


     size_t globalSize, localSize;

     // Number of work items in each local work group
	localSize = 64;

     


    cl_int err;

    // Create a context
    context = clCreateContext(0, 1, &theChosenDevice, NULL, NULL, &err);
    print_opencl_error(stderr, err);
    queue = clCreateCommandQueue(context, theChosenDevice, 0, &err);

     // Create the compute program from the source buffer
     program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource,(const size_t *) &source_size, &err);
     print_opencl_error(stderr, err);

    err = clBuildProgram(program, 0,NULL, NULL, NULL, NULL);

    if (err == CL_BUILD_PROGRAM_FAILURE) {
				cl_int logStatus;
				char* buildLog = NULL;
				size_t buildLogSize = 0;
				logStatus = clGetProgramBuildInfo (program, theChosenDevice, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
				buildLog = (char*)malloc(buildLogSize);
				memset(buildLog, 0, buildLogSize);
				logStatus = clGetProgramBuildInfo (program, theChosenDevice, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
				printf("Ae %d %s",err,  buildLog);
				free(buildLog);
				return err;
			} else if (err!=0) {
				print_opencl_error(stderr, err);
				return err;
			}


    // Create the compute kernel in the program we wish to run
			kernel = clCreateKernel(program, "createParameters", &err);
			print_opencl_error(stderr, err);

    // Create the input and output arrays in device memory for our calculation
			d_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, np[0]*sizeof(float), NULL, NULL);
			d_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, np[1]*sizeof(float), NULL, NULL);
			d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, np[2]*sizeof(float), NULL, NULL);
			d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, np[3]*sizeof(float), NULL, NULL);
			d_e = clCreateBuffer(context, CL_MEM_WRITE_ONLY, np[4]*sizeof(float), NULL, NULL);

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
			err |= clSetKernelArg(kernel, 1, sizeof(float), &n0[0]);
			err |= clSetKernelArg(kernel, 2, sizeof(float), &n1[0]);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &np[0]);

	// Number of total work items - localSize must be devisor
	globalSize = ceil(np[0]/(float)localSize)*localSize;

    // Execute the kernel over the entire range of the data set
			err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);


   print_opencl_error(stderr, err);

			err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_b);
			err |= clSetKernelArg(kernel, 1, sizeof(float), &n0[1]);
			err |= clSetKernelArg(kernel, 2, sizeof(float), &n1[1]);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &np[1]);

	// Number of total work items - localSize must be devisor
	globalSize = ceil(np[1]/(float)localSize)*localSize;

err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

   print_opencl_error(stderr, err);


err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
			err |= clSetKernelArg(kernel, 1, sizeof(float), &n0[2]);
			err |= clSetKernelArg(kernel, 2, sizeof(float), &n1[2]);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &np[2]);

	// Number of total work items - localSize must be devisor
	globalSize = ceil(np[2]/(float)localSize)*localSize;

err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

   print_opencl_error(stderr, err);

err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_d);
			err |= clSetKernelArg(kernel, 1, sizeof(float), &n0[3]);
			err |= clSetKernelArg(kernel, 2, sizeof(float), &n1[3]);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &np[3]);

	// Number of total work items - localSize must be devisor
	globalSize = ceil(np[3]/(float)localSize)*localSize;

err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

   print_opencl_error(stderr, err);

err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_e);
			err |= clSetKernelArg(kernel, 1, sizeof(float), &n0[4]);
			err |= clSetKernelArg(kernel, 2, sizeof(float), &n1[4]);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &np[4]);

	// Number of total work items - localSize must be devisor
	globalSize = ceil(np[4]/(float)localSize)*localSize;

err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

   print_opencl_error(stderr, err);

    // Wait for the command queue to get serviced before reading back results
			clFinish(queue);

    // Read the results from the device
			clEnqueueReadBuffer(queue, d_a, CL_TRUE, 0,
					np[0]*sizeof(float), a, 0, NULL, NULL );

			clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0,
					np[1]*sizeof(float), b, 0, NULL, NULL );

			clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
					np[2]*sizeof(float), c, 0, NULL, NULL );

			clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0,
					np[3]*sizeof(float), d, 0, NULL, NULL );

			clEnqueueReadBuffer(queue, d_e, CL_TRUE, 0,
					np[4]*sizeof(float), e, 0, NULL, NULL );


    
    
    

    //for(int i = 0; i < np[0]; i++){
      //a[i] = n0[0] + ((float)i / (float)np[0]) * range_A;
    //}

  /*  for(int i = 0; i < np[1]; i++){
      b[i] = n0[1] + ((float)i / (float)np[1]) * range_B;
    }

    for(int i = 0; i < np[2]; i++){
      c[i] = n0[2] + ((float)i / (float)np[2]) * range_C;
    }

    for(int i = 0; i < np[3]; i++){
      d[i] = n0[3] + ((float)i / (float)np[3]) * range_D;
    }

    for(int i = 0; i < np[4]; i++){
      e[i] = n0[4] + ((float)i / (float)np[4]) * range_E;
    } */

    int np0, np1, np2, np3, np4;
    np0 = np[0];
    np1 = np[1];
    np2 = np[2];
    np3 = np[3];
    np4 = np[4];





    su_trace_t *tr = ap->traces.a[0];
    float dt = (float) tr->dt / 1000000;
    float idt = 1 / dt;

    /* Calculate coherence window (number of trace samples in the trace to
       include in the semblance) */
    int tau = MAX((int)(ap->ap_t * idt), 0);
    int w = 2 * tau + 1;


//******************************************** MENOS "FORs" """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    int amount_of_possibilities = np0*np1*np2*np3*np4;

    int inner_np2 = np3 * np4;
    int inner_np1 = inner_np2 * np2;

    cl_mem *d_s;
    //s_possibilities = malloc(amount_of_possibilities * sizeof(float));
    cl_mem *d_stack;
    //stack_possibilities = malloc(amount_of_possibilities * sizeof(float));

    
    kernel = clCreateKernel(program, "compute_max", &err);
			print_opencl_error(stderr, err);

    
    cl_mem *d_ap,*d_traces;

    //d_ap = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3*sizeof(float) + ap->*sizeof(aperture_t), NULL, NULL);
    d_trace = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3*sizeof(float) + ap->*sizeof(aperture_t), NULL, NULL);
    d_s = clCreateBuffer(context, CL_MEM_WRITE_ONLY, amount_of_possibilities*sizeof(float), NULL, NULL);
    d_stack = clCreateBuffer(context, CL_MEM_WRITE_ONLY, amount_of_possibilities*sizeof(float), NULL, NULL);



    //#pragma omp parallel for schedule(dynamic) num_threads(8)
  /*  for (int ia = 0; ia < np0; ia++) {

      smax[ia] = -1;

      #pragma omp parallel for schedule(dynamic) num_threads(8)
      for(int i = 0; i < amount_of_possibilities; i++){

          float st; */
          /* Check the fit of the parameters to the data and update the
           * maximum for that point if necessary */
	  
	//  float s = semblance_2d(ap, a[ia], b[(i/inner_np1)%np1], c[(i/inner_np2)%np2], d[(i/np4)%np3], e[i%np4], t0, m0, h0, &st,
          //        dt, idt, tau, w);

  /*        s_possibilities[i] = semblance_2d(ap, a[ia], b[(i/inner_np1)%np1], c[(i/inner_np2)%np2], d[(i/np4)%np3], e[i%np4], t0, m0, h0, &st,
                  dt, idt, tau, w);

	  stack_possibilities[i] = st;

      }


      for(int i = 0; i < amount_of_possibilities; i++){

          if (s_possibilities[i] > smax[ia]) {
              smax[ia] = s_possibilities[i];
              _stack[ia] = stack_possibilities[i];
              _Aopt[ia] = a[ia];
              _Bopt[ia] = b[(i/inner_np1)%np1];
              _Copt[ia] = c[(i/inner_np2)%np2];
              _Dopt[ia] = d[(i/np4)%np3];
              _Eopt[ia] = e[i%np4];
          }

      }

	


    } */

//****************************************** MAIS "FORs" """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    /* Split the outermost loop between threads. Each thread will
     * compute the best fit for a given parameter 'A' value */
    // #pragma omp parallel for schedule(dynamic) num_threads(4)
    // for (int ia = 0; ia < np0; ia++) {
    //     smax[ia] = -1;
    //     //float a = n0[0] + ((float)ia / (float)np[0])*(n1[0]-n0[0]);
    //     //float a = n0[0] + ((float)ia / (float)np[0]) * range_A;
    //     for (int ib = 0; ib < np1; ib++) {
    //         //float b = n0[1] + ((float)ib / (float)np[1])*(n1[1]-n0[1]);
    //         //float b = n0[1] + ((float)ib / (float)np[1])* range_B;
    //         for (int ic = 0; ic < np2; ic++) {
    //             //float c = n0[2] + ((float)ic / (float)np[2])*(n1[2]-n0[2]);
    //             //float c = n0[2] + ((float)ic / (float)np[2])* range_C;
    //             for (int id = 0; id < np3; id++) {
    //                 //float d = n0[3] + ((float)id / (float)np[3])*(n1[3]-n0[3]);
    //                 //float d = n0[3] + ((float)id / (float)np[3])*range_D;
    //                 for (int ie = 0; ie < np4; ie++) {
    //                     //float e = n0[4] + ((float)ie / (float)np[4])*(n1[4]-n0[4]);
    //                     //float e = n0[4] + ((float)ie / (float)np[4])*range_E;
    //                     float st;
    //                     /* Check the fit of the parameters to the data and update the
    //                      * maximum for that point if necessary */
    //                     float s = semblance_2d(ap, a[ia], b[ib], c[ic], d[id], e[ie], t0, m0, h0, &st);
    //                     if (s > smax[ia]) {
    //                         smax[ia] = s;
    //                         _stack[ia] = st;
    //                         _Aopt[ia] = a[ia];
    //                         _Bopt[ia] = b[ib];
    //                         _Copt[ia] = c[ic];
    //                         _Dopt[ia] = d[id];
    //                         _Eopt[ia] = e[ie];
    //                     }
    //                 }
    //             }
    //         }
    //         /* Uncomment this to roughly check the progress */
    //         /* fprintf(stderr, "."); */
    //     }
    // }


//******************************************""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

/*

    // /* Split the outermost loop between threads. Each thread will
    //  * compute the best fit for a given parameter 'A' value */
    // #pragma omp parallel for schedule(dynamic) num_threads(4)
    // for (int ia = 0; ia < np[0]; ia++) {
    //     smax[ia] = -1;
    //     //float a = n0[0] + ((float)ia / (float)np[0])*(n1[0]-n0[0]);
    //     float a = n0[0] + ((float)ia / (float)np[0]) * range_A;
    //     for (int ib = 0; ib < np[1]; ib++) {
    //         //float b = n0[1] + ((float)ib / (float)np[1])*(n1[1]-n0[1]);
    //         float b = n0[1] + ((float)ib / (float)np[1])* range_B;
    //         for (int ic = 0; ic < np[2]; ic++) {
    //             //float c = n0[2] + ((float)ic / (float)np[2])*(n1[2]-n0[2]);
    //             float c = n0[2] + ((float)ic / (float)np[2])* range_C;
    //             for (int id = 0; id < np[3]; id++) {
    //                 //float d = n0[3] + ((float)id / (float)np[3])*(n1[3]-n0[3]);
    //                 float d = n0[3] + ((float)id / (float)np[3])*range_D;
    //                 for (int ie = 0; ie < np[4]; ie++) {
    //                     //float e = n0[4] + ((float)ie / (float)np[4])*(n1[4]-n0[4]);
    //                     float e = n0[4] + ((float)ie / (float)np[4])*range_E;
    //                     float st;
    //                     /* Check the fit of the parameters to the data and update the
    //                      * maximum for that point if necessary */
    //                     float s = semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);
    //                     if (s > smax[ia]) {
    //                         smax[ia] = s;
    //                         _stack[ia] = st;
    //                         _Aopt[ia] = a;
    //                         _Bopt[ia] = b;
    //                         _Copt[ia] = c;
    //                         _Dopt[ia] = d;
    //                         _Eopt[ia] = e;
    //                     }
    //                 }
    //             }
    //         }
    //         /* Uncomment this to roughly check the progress */
    //         /* fprintf(stderr, "."); */
    //     }
    // }

    /* Now find the best fit between different 'A' values */
    float ssmax = -1.0;
    *stack = 0;
    for (int ia = 0; ia < np[0]; ia++) {
        if (smax[ia] > ssmax) {
            *Aopt = _Aopt[ia];
            *Bopt = _Bopt[ia];
            *Copt = _Copt[ia];
            *Dopt = _Dopt[ia];
            *Eopt = _Eopt[ia];
            *stack = _stack[ia];
            *sem = smax[ia];
            ssmax = smax[ia];
        }
    }
}

int main(int argc, char *argv[])
{
    int i;

    



/*******************************************************************************/


	char deviceName[MAX_DEVICE_NAME_SIZE];
	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_platform_id* platforms;
	cl_uint platformCount;
	
	
	cl_int err;

	// Bind to platforms
	clGetPlatformIDs(0, NULL, &platformCount);
	if (platformCount == 0) {
	  printf("Error, cound not find any OpenCL platforms on the system.\n");
	  exit (2);
	}

	platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount,platforms, NULL);

	err = 1;
	
	for (i = 0; i < platformCount /*&& err !=CL_SUCCESS*/; i++){
		cl_uint quantidade = 0;
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, NULL, NULL, &quantidade);
		cl_device_id new_device_id[quantidade];
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, quantidade, &new_device_id, NULL);
		
		int itam;
		for(itam = 0; itam < quantidade; itam++){
			//printf("new_device_id[%d] = %d\n",itam, new_device_id[itam]);
			device_id = new_device_id[itam];
			err = clGetDeviceInfo(device_id, CL_DEVICE_NAME,MAX_DEVICE_NAME_SIZE, deviceName, NULL);
			printf("Device: %s\n", deviceName);
			theChosenDevice = device_id;
		}
		
		
	}


/********************************************************************************/

    if (argc != 21) {
        fprintf(stderr, "Usage: %s M0 H0 T0 TAU A0 A1 NA B0 B1 NB "
            "C0 C1 NC D0 D1 ND E0 E1 NE INPUT\n", argv[0]);
        exit(1);
    }

    float m0 = atof(argv[1]); // MO
    float h0 = atof(argv[2]); // HO
    float t0 = atof(argv[3]); // TO
    float tau = strtof(argv[4], NULL); // TAU

    /* A, B, C, D, E */
    float p0[5], p1[5];
    int np[5];

    /* p0 is where the search starts, p1 is where the search ends and np is the
     * number of points in between p0 and p1 to do the search */
    for (i = 0; i < 5; i++) {
        p0[i] = atof(argv[5 + 3*i]);
        //printf("[Mauricio] - > argv[%d] = %f\n", 5 + 3*i, atof(argv[5 + 3*i]));
        p1[i] = atof(argv[5 + 3*i + 1]);
        //printf("[Mauricio] - > argv[%d] = %f\n", 5 + 3*i +1, atof(argv[5 + 3*i +1]));
        np[i] = atoi(argv[5 + 3*i + 2]);
        //printf("[Mauricio] - > argv[%d] = %f\n", 5 + 3*i +2, atof(argv[5 + 3*i +2]));
    }

    /* Load the traces from the file */

    char *path = argv[20];
    FILE *fp = fopen(path, "r");

    if (!fp) {
        fprintf(stderr, "Failed to open prestack file '%s'!\n", path);
        return 1;
    }

    su_trace_t tr;
    vector_t(su_trace_t) traces;
    vector_init(traces); // Inicializa valores com 0 e ponteiros com NULL

    while (su_fgettr(fp, &tr)) {
        vector_push(traces, tr);
    }

    // while (su_fgettr(fp, &tr)) {
    //   if ((traces).len == (traces).cap) {
    //     (traces).cap *= 2, (traces).cap++;
    //     (traces).a = realloc((traces).a, sizeof(__typeof((traces).t))*(traces).cap);
    //   }
    //   (traces).a[(traces).len++] = tr;
    // }


    /* Construct the aperture structure from the traces, which is a vector
     * containing pointers to traces */

    aperture_t ap;
    ap.ap_m = 0;
    ap.ap_h = 0;
    ap.ap_t = tau;
    vector_init(ap.traces);
    for (int i = 0; i < traces.len; i++)
        vector_push(ap.traces, &vector_get(traces, i));


    // for (int i = 0; i < traces.len; i++){
    //   if ((ap.traces).len == (ap.traces).cap) {
    //     (ap.traces).cap *= 2, (ap.traces).cap++;
    //     (ap.traces).a = realloc((ap.traces).a, sizeof(__typeof((ap.traces).t))*(ap.traces).cap);
    //   }
    //   (ap.traces).a[(ap.traces).len++] = &vector_get(traces, i);
    // }

    /* Find the best parameter combination */

    float a, b, c, d, e, sem, stack;
    compute_max(&ap, m0, h0, t0, p0, p1, np, &a, &b, &c, &d, &e, &sem, &stack);

    printf("A=%g\n", a);
    printf("B=%g\n", b);
    printf("C=%g\n", c);
    printf("D=%g\n", d);
    printf("E=%g\n", e);
    printf("Stack=%g\n", stack);
    printf("Semblance=%g\n", sem);
    printf("\n");

    return 0;
}



void print_opencl_error(FILE* fh, cl_int err)
{
#define PRINT_ERR(code) case code : fprintf(fh, #code); break
	switch(err) {
	PRINT_ERR(CL_INVALID_PROGRAM);
	PRINT_ERR(CL_INVALID_VALUE);
	PRINT_ERR(CL_INVALID_DEVICE);
	PRINT_ERR(CL_INVALID_BINARY);
	PRINT_ERR(CL_INVALID_BUILD_OPTIONS);
	PRINT_ERR(CL_INVALID_OPERATION);
	PRINT_ERR(CL_COMPILER_NOT_AVAILABLE);
	PRINT_ERR(CL_BUILD_PROGRAM_FAILURE);
	PRINT_ERR(CL_OUT_OF_RESOURCES);
	PRINT_ERR(CL_OUT_OF_HOST_MEMORY);
	default:
		if(err!=CL_SUCCESS)
			fprintf(fh, "unknown code");
		break;
	};
	return;
}


