package paragon

import "math"

// applyActivation applies the specified activation function
func applyActivation(value float64, activation string) float64 {
	switch activation {
	case "relu":
		return math.Max(0, value)
	case "sigmoid":
		return 1 / (1 + math.Exp(-value))
	case "tanh":
		return math.Tanh(value)
	case "leaky_relu":
		if value > 0 {
			return value
		}
		return 0.01 * value
	case "elu":
		if value >= 0 {
			return value
		}
		return 1.0 * (math.Exp(value) - 1)
	case "linear":
		return value
	default:
		return value // Fallback to linear
	}
}

// activationDerivative computes the derivative of the activation function
func activationDerivative(value float64, activation string) float64 {
	switch activation {
	case "relu":
		if value > 0 {
			return 1
		}
		return 0
	case "sigmoid":
		sig := 1 / (1 + math.Exp(-value))
		return sig * (1 - sig)
	case "tanh":
		t := math.Tanh(value)
		return 1 - t*t
	case "leaky_relu":
		if value > 0 {
			return 1
		}
		return 0.01
	case "elu":
		if value >= 0 {
			return 1
		}
		return math.Exp(value)
	case "linear":
		return 1
	default:
		return 1 // Fallback to linear
	}
}
