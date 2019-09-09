This is an experimental package that implements Gorgonia's [BLAS](https://godoc.org/gorgonia.org/gorgonia#BLAS) interface with WebGL for WASM architecture.

**Warning** This package is highly experimental and is not suitable for production.

## How?

This package is copying all of gonmu's BLAS packages, and the goal is to replace some function by something specific when compiled with GOOS=js and GOARCH=wasm
