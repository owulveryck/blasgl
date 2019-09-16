This is an experimental package that implements Gorgonia's [BLAS](https://godoc.org/gorgonia.org/gorgonia#BLAS) interface with WebGL for WASM architecture.

**Warning** This package is highly experimental and is not suitable for production.

## How?

This package is copying all of gonmu's BLAS packages, and the goal is to replace some function by something specific when compiled with GOOS=js and GOARCH=wasm

Three steps:

- make it work
- make it right
- make it fast

## make it work

To validate the POC, this experiment is using the [weblas](https://github.com/waylonflinn/weblas) library.
To test the implementation easily, this POC is using a tweaked version of [wasmbrowsertest](https://github.com/agnivade/wasmbrowsertestt).

To run a test, first compile the wasmbrowsertest by running a `go build` within the `wasmbrowsertest` subdirectory.

Then run:

`GOOS=js GOARCH=wasm go test -exec ./wasmbrowsertest/wasmbrowsertest -v`

The tests should pass

The tests should pass
