package paragon

import (
	"math"
	"reflect"
)

type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// default fixed-point scale for “1.0” in integer types
const SCALE = 256

// getScaleForType picks the integer “1.0” for each T
func getScaleForType[T Numeric]() int64 {
	switch reflect.TypeOf(*new(T)).Kind() {
	case reflect.Int:
		return math.MaxInt32 // 2147483647, using int32 max for safety
	case reflect.Int8:
		return math.MaxInt8 // 127
	case reflect.Int16:
		return math.MaxInt16 // 32767
	case reflect.Int32:
		return math.MaxInt32 // 2147483647
	case reflect.Int64:
		return math.MaxInt32 // 2147483647, cap at int32 for consistency
	case reflect.Uint:
		return math.MaxUint32 // 4294967295, using uint32 max
	case reflect.Uint8:
		return math.MaxUint8 // 255
	case reflect.Uint16:
		return math.MaxUint16 // 65535
	case reflect.Uint32:
		return math.MaxUint32 // 4294967295
	case reflect.Uint64:
		return math.MaxUint32 // 4294967295, cap at uint32 for consistency
	case reflect.Float32, reflect.Float64:
		return 0 // floats don't use fixed-point scaling
	default:
		return SCALE
	}
}

// ApplyActivationGeneric dispatches by name
func ApplyActivationGeneric[T Numeric](x T, act string) T {
	switch act {
	case "relu":
		return ReLU(x)
	case "sigmoid":
		return Sigmoid(x)
	case "tanh":
		return Tanh(x)
	case "leaky_relu":
		return LeakyReLU(x)
	case "elu":
		return ELU(x)
	case "linear":
		return Linear(x)
	default:
		return x
	}
}

// ActivationDerivativeGeneric dispatches by name
func ActivationDerivativeGeneric[T Numeric](x T, act string) T {
	switch act {
	case "relu":
		return DReLU(x)
	case "sigmoid":
		return DSigmoid(x)
	case "tanh":
		return DTanh(x)
	case "leaky_relu":
		return DLeakyReLU(x)
	case "elu":
		return DELU(x)
	case "linear":
		return DLinear(x)
	default:
		return DLinear(x)
	}
}

// ─────── Activations ───────

func ReLU[T Numeric](x T) T {
	var zero T
	if x > zero {
		return x
	}
	return zero
}

func Sigmoid[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	// floats: exact
	if kind == reflect.Float32 || kind == reflect.Float64 {
		f := float64(any(x).(T))
		s := 1.0 / (1.0 + math.Exp(-f))
		return T(s)
	}
	// integers: fixed-point
	xi := int64(any(x).(T))
	scale := getScaleForType[T]()
	// map to real
	realX := float64(xi) / float64(scale)
	s := 1.0 / (1.0 + math.Exp(-realX))
	// back to integer
	out := int64(math.Round(s * float64(scale)))
	if out < 0 {
		out = 0
	} else if out > scale {
		out = scale
	}
	return T(out)
}

func Tanh[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	// floats: exact
	if kind == reflect.Float32 || kind == reflect.Float64 {
		f := float64(any(x).(T))
		return T(math.Tanh(f))
	}
	// integers: fixed-point
	xi := int64(any(x).(T))
	scale := getScaleForType[T]()
	realX := float64(xi) / float64(scale)
	t := math.Tanh(realX)
	out := int64(math.Round(t * float64(scale)))
	if out < -scale {
		out = -scale
	} else if out > scale {
		out = scale
	}
	return T(out)
}

func LeakyReLU[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		f := float64(any(x).(T))
		if f > 0 {
			return x
		}
		return T(f * 0.01)
	}
	// integers: small leak
	xi := int64(any(x).(T))
	if xi >= 0 {
		return x
	}
	scale := getScaleForType[T]()
	realX := float64(xi) / float64(scale)
	leak := realX * 0.01
	out := int64(math.Round(leak * float64(scale)))
	min := -scale
	if out < min {
		out = min
	}
	return T(out)
}

func ELU[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		f := float64(any(x).(T))
		if f >= 0 {
			return x
		}
		return T(math.Exp(f) - 1)
	}
	xi := int64(any(x).(T))
	scale := getScaleForType[T]()
	realX := float64(xi) / float64(scale)
	var e float64
	if realX >= 0 {
		e = realX
	} else {
		e = math.Exp(realX) - 1
	}
	out := int64(math.Round(e * float64(scale)))
	if out < -scale {
		out = -scale
	} else if out > scale {
		out = scale
	}
	return T(out)
}

func Linear[T Numeric](x T) T {
	return x
}

// ─────── Derivatives ───────

func DReLU[T Numeric](x T) T {
	var zero, one T = 0, 1
	if x > zero {
		return one
	}
	return zero
}

func DSigmoid[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		f := float64(any(x).(T))
		s := 1.0 / (1.0 + math.Exp(-f))
		return T(s * (1 - s))
	}
	// integer approximate
	xi := int64(any(x).(T))
	scale := getScaleForType[T]()
	realX := float64(xi) / float64(scale)
	s := 1.0 / (1.0 + math.Exp(-realX))
	ds := s * (1 - s)
	out := int64(math.Round(ds * float64(scale)))
	if out < 0 {
		out = 0
	} else if out > scale {
		out = scale
	}
	return T(out)
}

func DTanh[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		t := math.Tanh(float64(any(x).(T)))
		return T(1 - t*t)
	}
	xi := int64(any(x).(T))
	scale := getScaleForType[T]()
	realX := float64(xi) / float64(scale)
	t := math.Tanh(realX)
	dt := 1 - t*t
	out := int64(math.Round(dt * float64(scale)))
	if out < 0 {
		out = 0
	} else if out > scale {
		out = scale
	}
	return T(out)
}

// DLeakyReLU returns 1 (or 1.0) when x>0, and ~0.01 (or scaled ≈1 in fixed-point) when x<=0
func DLeakyReLU[T Numeric](x T) T {
	var zero T
	if x > zero {
		// untyped 1 converts to any int or float T
		return T(1)
	}
	// castLeakFactor handles float32/64 -> 0.01,
	// and int*/uint* -> int64(0.01*scale) with minimum 1
	return castLeakFactor[T](0.01)
}

func DELU[T Numeric](x T) T {
	kind := reflect.TypeOf(x).Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		return T(math.Exp(float64(any(x).(T))))
	}
	// integer fallback
	return T(1)
}

func DLinear[T Numeric](x T) T {
	return T(1)
}

// helper for DLeakyReLU
func castLeakFactor[T Numeric](val float64) T {
	var t T
	switch any(t).(type) {
	case float32:
		return T(float32(val))
	case float64:
		return T(val)
	default:
		scale := getScaleForType[T]()
		leak := int64(math.Round(val * float64(scale)))
		if leak < 1 {
			leak = 1
		}
		return T(leak)
	}
}
