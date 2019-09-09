package blasgl

import (
	"testing"

	"gonum.org/v1/gonum/blas/testblas"
)

func TestDgemm(t *testing.T) {
	testblas.TestDgemm(t, Implementation{})
}
