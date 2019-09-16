package blasgl

import (
	"testing"

	"github.com/chewxy/math32"
	"gonum.org/v1/gonum/blas"
)

func TestSgemm(t *testing.T) {
	testSgemm(t, Implementation{})
}

func testSgemm(t *testing.T, blasser sgemmer) {
	for i, test := range sgemmCases {
		// Test that it passes row major
		sgemmcomp(i, "RowMajorNoTrans", t, blasser, blas.NoTrans, blas.NoTrans,
			test.m, test.n, test.k, test.alpha, test.beta, test.a, test.b, test.c, test.ans)
		// Try with A transposed
		sgemmcomp(i, "RowMajorTransA", t, blasser, blas.Trans, blas.NoTrans,
			test.m, test.n, test.k, test.alpha, test.beta, transpose(test.a), test.b, test.c, test.ans)
		// Try with B transposed
		sgemmcomp(i, "RowMajorTransB", t, blasser, blas.NoTrans, blas.Trans,
			test.m, test.n, test.k, test.alpha, test.beta, test.a, transpose(test.b), test.c, test.ans)
		// Try with both transposed
		sgemmcomp(i, "RowMajorTransBoth", t, blasser, blas.Trans, blas.Trans,
			test.m, test.n, test.k, test.alpha, test.beta, transpose(test.a), transpose(test.b), test.c, test.ans)
	}
}

type sgemmer interface {
	Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
}
type sgemmCase struct {
	m, n, k     int
	alpha, beta float32
	a           [][]float32
	b           [][]float32
	c           [][]float32
	ans         [][]float32
}

var sgemmCases = []sgemmCase{

	{
		m:     4,
		n:     3,
		k:     2,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2},
			{4, 5},
			{7, 8},
			{10, 11},
		},
		b: [][]float32{
			{1, 5, 6},
			{5, -8, 8},
		},
		c: [][]float32{
			{4, 8, -9},
			{12, 16, -8},
			{1, 5, 15},
			{-3, -4, 7},
		},
		ans: [][]float32{
			{24, -18, 39.5},
			{64, -32, 124},
			{94.5, -55.5, 219.5},
			{128.5, -78, 299.5},
		},
	},
	{
		m:     4,
		n:     2,
		k:     3,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
			{10, 11, 12},
		},
		b: [][]float32{
			{1, 5},
			{5, -8},
			{6, 2},
		},
		c: [][]float32{
			{4, 8},
			{12, 16},
			{1, 5},
			{-3, -4},
		},
		ans: [][]float32{
			{60, -6},
			{136, -8},
			{202.5, -19.5},
			{272.5, -30},
		},
	},
	{
		m:     3,
		n:     2,
		k:     4,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2, 3, 4},
			{4, 5, 6, 7},
			{8, 9, 10, 11},
		},
		b: [][]float32{
			{1, 5},
			{5, -8},
			{6, 2},
			{8, 10},
		},
		c: [][]float32{
			{4, 8},
			{12, 16},
			{9, -10},
		},
		ans: [][]float32{
			{124, 74},
			{248, 132},
			{406.5, 191},
		},
	},
	{
		m:     3,
		n:     4,
		k:     2,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2},
			{4, 5},
			{8, 9},
		},
		b: [][]float32{
			{1, 5, 2, 1},
			{5, -8, 2, 1},
		},
		c: [][]float32{
			{4, 8, 2, 2},
			{12, 16, 8, 9},
			{9, -10, 10, 10},
		},
		ans: [][]float32{
			{24, -18, 13, 7},
			{64, -32, 40, 22.5},
			{110.5, -69, 73, 39},
		},
	},
	{
		m:     2,
		n:     4,
		k:     3,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2, 3},
			{4, 5, 6},
		},
		b: [][]float32{
			{1, 5, 8, 8},
			{5, -8, 9, 10},
			{6, 2, -3, 2},
		},
		c: [][]float32{
			{4, 8, 7, 8},
			{12, 16, -2, 6},
		},
		ans: [][]float32{
			{60, -6, 37.5, 72},
			{136, -8, 117, 191},
		},
	},
	{
		m:     2,
		n:     3,
		k:     4,
		alpha: 2,
		beta:  0.5,
		a: [][]float32{
			{1, 2, 3, 4},
			{4, 5, 6, 7},
		},
		b: [][]float32{
			{1, 5, 8},
			{5, -8, 9},
			{6, 2, -3},
			{8, 10, 2},
		},
		c: [][]float32{
			{4, 8, 1},
			{12, 16, 6},
		},
		ans: [][]float32{
			{124, 74, 50.5},
			{248, 132, 149},
		},
	},
}

// assumes [][]float32 is actually a matrix
func transpose(a [][]float32) [][]float32 {
	b := make([][]float32, len(a[0]))
	for i := range b {
		b[i] = make([]float32, len(a))
		for j := range b[i] {
			b[i][j] = a[j][i]
		}
	}
	return b
}

func sgemmcomp(i int, name string, t *testing.T, blasser sgemmer, tA, tB blas.Transpose, m, n, k int,
	alpha, beta float32, a [][]float32, b [][]float32, c [][]float32, ans [][]float32) {

	aFlat := flatten(a)
	aCopy := flatten(a)
	bFlat := flatten(b)
	bCopy := flatten(b)
	cFlat := flatten(c)
	ansFlat := flatten(ans)
	lda := len(a[0])
	ldb := len(b[0])
	ldc := len(c[0])

	// Compute the matrix multiplication
	blasser.Sgemm(tA, tB, m, n, k, alpha, aFlat, lda, bFlat, ldb, beta, cFlat, ldc)

	if !dSliceEqual(aFlat, aCopy) {
		t.Errorf("Test %v case %v: a changed during call to Dgemm", i, name)
	}
	if !dSliceEqual(bFlat, bCopy) {
		t.Errorf("Test %v case %v: b changed during call to Dgemm", i, name)
	}

	if !dSliceTolEqual(ansFlat, cFlat) {
		t.Errorf("Test %v case %v: answer mismatch. Expected %v, Found %v", i, name, ansFlat, cFlat)
	}
	// TODO: Need to add a sub-slice test where don't use up full matrix
}

func flatten(a [][]float32) []float32 {
	if len(a) == 0 {
		return nil
	}
	m := len(a)
	n := len(a[0])
	s := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			s[i*n+j] = a[i][j]
		}
	}
	return s
}

func dSliceTolEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !dTolEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func dTolEqual(a, b float32) bool {
	if math32.IsNaN(a) && math32.IsNaN(b) {
		return true
	}
	if a == b {
		return true
	}
	m := math32.Max(math32.Abs(a), math32.Abs(b))
	if m > 1 {
		a /= m
		b /= m
	}
	if math32.Abs(a-b) < 1e-14 {
		return true
	}
	return false
}

func dSliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !dTolEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}
