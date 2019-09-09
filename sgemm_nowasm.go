// +build !wasm

package blasgl

import "gonum.org/v1/gonum/blas"

// Sgemm ...
func (i *Implementation) Sgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

}
