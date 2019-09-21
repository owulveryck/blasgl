// +build wasm

package blasgl

import (
	"gonum.org/v1/gonum/blas"
)

// Sgemm ...
func (Implementation) Sgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	switch {
	case tA != blas.NoTrans:
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tB != blas.NoTrans:
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tA == blas.NoTrans && m != lda:
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case tB == blas.NoTrans && k != ldb:
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case m != ldc:
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	default:
		// TODO use the webladgl here
		gonumImpl.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	}
}

// Dgemm ...
func (Implementation) Dgemm(tA blas.Transpose, tB blas.Transpose, m int, n int, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	gonumImpl.Dgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

}
