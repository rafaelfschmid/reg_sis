__kernel void createParameters(__global float *d_a, const float n0, const float n1, const int np)
{
  //Get our global thread ID
  int id = get_global_id(0);

  // Make sure we do not go out of bounds
  if (id < np) {
    d_a[id] = n0 + ( (float)id / (float)np ) * ( n1 - n0 );
  }
}

/**************************************************************************************/


typedef struct su_trace su_trace_t;

struct su_trace {
	int tracl;
	int tracr;
	int fldr;
	int tracf;
	int ep;
	int cdp;
	int cdpt;
	short trid;
	short nvs;
	short nhs;
	short duse;
	int offset;
	int gelev;
	int selev;
	int sdepth;
	int gdel;
	int sdel;
	int swdep;
	int gwdep;
	short scalel;
	short scalco;
	int sx;
	int sy;
	int gx;
	int gy;
	short counit;
	short wevel;
	short swevel;
	short sut;
	short gut;
	short sstat;
	short gstat;
	short tstat;
	short laga;
	short lagb;
	short delrt;
	short muts;
	short mute;
	unsigned short ns;
	unsigned short dt;
	short gain;
	short igc;
	short igi;
	short corr;
	short sfs;
	short sfe;
	short slen;
	short styp;
	short stas;
	short stae;
	short tatyp;
	short afilf;
	short afils;
	short nofilf;
	short nofils;
	short lcf;
	short hcf;
	short lcs;
	short hcs;
	short year;
	short day;
	short hour;
	short minute;
	short sec;
	short timbas;
	short trwf;
	short grnors;
	short grnofr;
	short grnlof;
	short gaps;
	short otrav;
	float d1;
	float f1;
	float d2;
	float f2;
	float ungpow;
	float unscale;
	int ntr;
	short mark;
        short shortpad;
	short unass[14];
	float *data;
};



#define vector_t(type) struct {int len, cap; type t; type *a;}
#define vector_init(v) ((v).len = (v).cap = 0, (v).a = NULL)
#define vector_push(v, x) do { \
	if ((v).len == (v).cap) { \
		(v).cap *= 2, (v).cap++; \
		(v).a = realloc((v).a, sizeof(__typeof((v).t))*(v).cap); \
	} \
	(v).a[(v).len++] = x; \
} while (0)
#define vector_get(v, i) ((v).a[i])
#define vector_set(v, i, x) ((v).a[i]=x)

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t;
    vector_t(su_trace_t*) traces;
};




__kernel void compute_max(__global aperture_t *d_ap, __global float *d_a, __global float *d_b, __global float *d_c, __global float *d_d, __global float *d_e,

 const int np0, const int np1, const int np2, const int np3, const int np4,

 const float t0, const float m0, const float h0, __global float *d_stack, const float dt, const float idt, const float tau, const float w,

__global float *d_s, const int amount_of_possibilities)
{
	//Get our global thread ID
  int id = get_global_id(0);

  // Make sure we do not go out of bounds
  if (id < amount_of_possibilities) {


    //int inner_np4 = 1;
    //int inner_np3 = np4;
    int inner_np2 = np3 * np4;
    int inner_np1 = inner_np2 * np2;
    int inner_np0 = inner_np1 * np1;

    d_s[id] = semblance_2d(d_ap, d_a[(id/inner_np0)%np0], d_b[(id/inner_np1)%np1], d_c[(id/inner_np2)%np2], d_d[(id/np4)%np3], d_e[id%np4], t0, m0, h0, &d_stack[id],
                  dt, idt, tau, w);


  }



}
