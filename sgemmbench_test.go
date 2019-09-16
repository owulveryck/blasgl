package blasgl

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas"
)

func BenchmarkSgemmSmSmSm(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Sm, Sm, Sm, NT, NT)
}

func BenchmarkSgemmMedMedMed(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Med, Med, Med, NT, NT)
}

func BenchmarkSgemmMedLgMed(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Med, Lg, Med, NT, NT)
}

func BenchmarkSgemmLgLgLg(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Lg, Lg, Lg, NT, NT)
}

func BenchmarkSgemmLgSmLg(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Lg, Sm, Lg, NT, NT)
}

func BenchmarkSgemmLgLgSm(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Lg, Lg, Sm, NT, NT)
}

func BenchmarkSgemmHgHgSm(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Hg, Hg, Sm, NT, NT)
}

func BenchmarkSgemmMedMedMedTNT(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Med, Med, Med, T, NT)
}

func BenchmarkSgemmMedMedMedNTT(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Med, Med, Med, NT, T)
}

func BenchmarkSgemmMedMedMedTT(b *testing.B) {
	sgemmBenchmark(b, Implementation{}, Med, Med, Med, T, T)
}
func sgemmBenchmark(b *testing.B, sgemm sgemmer, m, n, k int, tA, tB blas.Transpose) {
	a := make([]float32, m*k)
	for i := range a {
		a[i] = rand.Float32()
	}
	bv := make([]float32, k*n)
	for i := range bv {
		bv[i] = rand.Float32()
	}
	c := make([]float32, m*n)
	for i := range c {
		c[i] = rand.Float32()
	}
	var lda, ldb int
	if tA == blas.Trans {
		lda = m
	} else {
		lda = k
	}
	if tB == blas.Trans {
		ldb = k
	} else {
		ldb = n
	}
	ldc := n
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sgemm.Sgemm(tA, tB, m, n, k, 3.0, a, lda, bv, ldb, 1.0, c, ldc)
	}
}
