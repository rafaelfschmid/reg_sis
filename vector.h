#ifndef VECTOR_H__
#define VECTOR_H__

#include <stdlib.h>

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

#endif /* VECTOR_H__ */


// float s = semblance_2d(ap, a[ia], b[(i/inner_np1)%np1],
//  													c[(i/inner_np2)%np2], d[(i/np4)%np3],
// 													 e[i%np4], t0, m0, h0, &st);
//
// semblance_2d(aperture_t *ap,
//         float A, float B, float C, float D, float E,
//         float t0, float m0, float h0,
//         float *stack)
