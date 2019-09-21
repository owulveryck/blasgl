// +build wasm

package blasgl

import (
	"fmt"
	"syscall/js"

	"gonum.org/v1/gonum/blas"
)

// Sgemm ...
func (Implementation) Sgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	switch {
	case tA != blas.NoTrans:
		fmt.Println("trans A")
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tB != blas.NoTrans:
		fmt.Println("trans A")
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tA == blas.NoTrans && m != lda:
		fmt.Println("m != lda")
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tB == blas.NoTrans && k != ldb:
		fmt.Println("k != ldb")
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case m != ldc:
		fmt.Println("m != ldc")
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	default:
		fmt.Println("default")
		//
		// TODO use the webladgl here
		/*
						// From the example
						// create Matrices as arrays
						var height_A = 1024, width_A = 1024,
						    height_B = 1024, width_B = 1024;

						var A = new Float32Array(height_A * width_A);
						var B = new Float32Array(height_B * width_B);

						// fill A and B with science

						var M = height_A,
						    N = width_B,
						    K = height_B; // must match width_A

						var alpha = 1.0;
						var beta = 0.0;
						var C = new Float32Array(width_B)      // specialized for neural net bias calculation

						// result will contain matrix multiply of A x B (times alpha)
						result = weblas.sgemmar height_A = 1024, width_A = 1024,
			    height_B = 1024, width_B = 1024;

			var A = new Float32Array(height_A * width_A);
			var B = new Float32Array(height_B * width_B);

			// fill A and B with science

			var M = height_A,
			    N = width_B,
			    K = height_B; // must match width_A

			var alpha = 1.0;
			var beta = 0.0;
			var C = new Float32Array(width_B)      // specialized for neural net bias calculation

			// result will contain matrix multiply of A x B (times alpha)
			result = weblas.sgemm(M, N, K, alpha, A, B, beta, C);(M, N, K, alpha, A, B, beta, C);
		*/
		aJS := SliceToTypedArray(a)
		bJS := SliceToTypedArray(b)
		cJS := SliceToTypedArray(c)
		ret := js.Global().Get("weblas").Call("sgemm", m, n, k, alpha, aJS, bJS, beta, cJS)
		fmt.Println(ret)
		//gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	}
}

// Dgemm ...
func (Implementation) Dgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	gonumImpl.Dgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

}
