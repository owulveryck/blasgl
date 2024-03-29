package blasgl

// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import (
	"testing"

	"gonum.org/v1/gonum/blas/testblas"
)

func BenchmarkDgemmSmSmSm(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Sm, Sm, Sm, NT, NT)
}

func BenchmarkDgemmMedMedMed(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Med, Med, Med, NT, NT)
}

func BenchmarkDgemmMedLgMed(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Med, Lg, Med, NT, NT)
}

func BenchmarkDgemmLgLgLg(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Lg, Lg, Lg, NT, NT)
}

func BenchmarkDgemmLgSmLg(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Lg, Sm, Lg, NT, NT)
}

func BenchmarkDgemmLgLgSm(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Lg, Lg, Sm, NT, NT)
}

func BenchmarkDgemmHgHgSm(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Hg, Hg, Sm, NT, NT)
}

func BenchmarkDgemmMedMedMedTNT(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Med, Med, Med, T, NT)
}

func BenchmarkDgemmMedMedMedNTT(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Med, Med, Med, NT, T)
}

func BenchmarkDgemmMedMedMedTT(b *testing.B) {
	testblas.DgemmBenchmark(b, Implementation{}, Med, Med, Med, T, T)
}
