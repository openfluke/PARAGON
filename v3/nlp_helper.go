package paragon

import (
	"math"
	"math/rand"
)

// GenerateText generates a text sequence from the network given a seed sequence
func (n *Network[T]) GenerateText(seed []float64, vocabSize, length int, temperature float64) string {
	output := make([]int, 0, length)
	current := make([]float64, len(seed))
	copy(current, seed)

	for i := 0; i < length; i++ {
		// Prepare input as a 2D slice
		input := [][]float64{current}
		n.Forward(input)

		// Get output probabilities
		probs := n.GetOutput()
		// Apply temperature scaling
		for j := range probs {
			probs[j] = math.Pow(probs[j], 1.0/temperature)
		}
		// Normalize
		sum := 0.0
		for _, p := range probs {
			sum += p
		}
		for j := range probs {
			probs[j] /= sum
		}
		// Sample next character
		r := rand.Float64()
		cumSum := 0.0
		nextChar := 0
		for j, p := range probs {
			cumSum += p
			if r <= cumSum {
				nextChar = j
				break
			}
		}
		output = append(output, nextChar)

		// Shift input sequence
		copy(current[:len(current)-vocabSize], current[vocabSize:])
		for j := range current[len(current)-vocabSize:] {
			current[len(current)-vocabSize+j] = 0
		}
		current[len(current)-vocabSize+nextChar] = 1
	}

	// Convert indices to characters
	vocab := "abcdefghijklmnopqrstuvwxyz .,!?"
	result := make([]byte, len(output))
	for i, idx := range output {
		if idx >= 0 && idx < len(vocab) {
			result[i] = vocab[idx]
		} else {
			result[i] = '?'
		}
	}
	return string(result)
}

// ComputePerplexity calculates the perplexity of the network on a test set
func (n *Network[T]) ComputePerplexity(inputs [][][]float64, targets [][][]float64) float64 {
	totalLogProb := 0.0
	totalChars := 0

	for i := range inputs {
		n.Forward(inputs[i])
		output := n.GetOutput()
		target := targets[i][0] // Assuming single target per sequence
		for j, t := range target {
			if t > 0 {
				prob := output[j]
				if prob <= 1e-10 {
					prob = 1e-10
				}
				totalLogProb += math.Log(prob)
				totalChars++
			}
		}
	}

	if totalChars == 0 {
		return math.Inf(1)
	}
	return math.Exp(-totalLogProb / float64(totalChars))
}
