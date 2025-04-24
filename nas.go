package paragon

import (
	"fmt"
	"math/rand"
)

// List of possible activation functions for randomization
var possibleActivations = []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"}

// PartitionNASLayerWiseGrowing implements partitioned NAS with layer-wise growing.
// It trains only the specified partition until the target ADHD metric is reached.
// If activation is "", a random activation is chosen for each new layer; otherwise, the specified activation is used.
func (n *Network) PartitionNASLayerWiseGrowing(inputs [][][]float64, targets [][][]float64, minNeurons, maxNeurons int, activation string, targetMetric string, targetValue float64, numTags int, selectedTag int) error {
	// Validate inputs
	if minNeurons <= 1 || maxNeurons <= minNeurons {
		return fmt.Errorf("invalid neuron counts: minNeurons=%d must be > 1, maxNeurons=%d must be > minNeurons", minNeurons, maxNeurons)
	}
	if selectedTag < 0 || selectedTag >= numTags {
		return fmt.Errorf("invalid selectedTag: %d must be between 0 and %d", selectedTag, numTags-1)
	}
	if len(inputs) != len(targets) {
		return fmt.Errorf("mismatched inputs and targets: %d inputs, %d targets", len(inputs), len(targets))
	}

	learningRate := 0.01 // Fixed learning rate
	epochs := 1          // One epoch per layer addition

	for {
		// Determine activation for the new layer
		var act string
		if activation == "" {
			act = possibleActivations[rand.Intn(len(possibleActivations))]
		} else {
			act = activation
		}

		// Add a new layer with random neuron count and chosen activation
		width := rand.Intn(maxNeurons-minNeurons+1) + minNeurons
		n.AddLayer(n.OutputLayer, width, 1, act, true)

		// Train the selected partition for one epoch
		if epochs > 0 {
			for i := range inputs {
				n.ForwardTagged(inputs[i], numTags, selectedTag)
				n.BackwardTagged(targets[i], learningRate, numTags, selectedTag)
			}
		}

		// Evaluate the network
		EvaluateWithADHD(n, inputs, targets)
		var metricValue float64
		switch targetMetric {
		case "accuracy":
			metricValue = ComputeAccuracy(n, inputs, targets) * 100
		case "score":
			metricValue = n.Performance.Score
		default:
			return fmt.Errorf("unsupported target metric: %s", targetMetric)
		}

		// Stop if target metric is reached
		if metricValue >= targetValue {
			break
		}
	}
	return nil
}

// NormalNASLayerWiseGrowing implements standard NAS with layer-wise growing.
// It trains the entire network until the target ADHD metric is reached.
// If activation is "", a random activation is chosen for each new layer; otherwise, the specified activation is used.
func (n *Network) NormalNASLayerWiseGrowing(inputs [][][]float64, targets [][][]float64, minNeurons, maxNeurons int, activation string, targetMetric string, targetValue float64) error {
	// Validate inputs
	if minNeurons <= 1 || maxNeurons <= minNeurons {
		return fmt.Errorf("invalid neuron counts: minNeurons=%d must be > 1, maxNeurons=%d must be > minNeurons", minNeurons, maxNeurons)
	}
	if len(inputs) != len(targets) {
		return fmt.Errorf("mismatched inputs and targets: %d inputs, %d targets", len(inputs), len(targets))
	}

	learningRate := 0.01 // Fixed learning rate
	epochs := 1          // One epoch per layer addition

	for {
		// Determine activation for the new layer
		var act string
		if activation == "" {
			act = possibleActivations[rand.Intn(len(possibleActivations))]
		} else {
			act = activation
		}

		// Add a new layer with random neuron count and chosen activation
		width := rand.Intn(maxNeurons-minNeurons+1) + minNeurons
		n.AddLayer(n.OutputLayer, width, 1, act, true)

		// Train the entire network for one epoch
		if epochs > 0 {
			for i := range inputs {
				n.Forward(inputs[i])
				n.Backward(targets[i], learningRate)
			}
		}

		// Evaluate the network
		EvaluateWithADHD(n, inputs, targets)
		var metricValue float64
		switch targetMetric {
		case "accuracy":
			metricValue = ComputeAccuracy(n, inputs, targets) * 100
		case "score":
			metricValue = n.Performance.Score
		default:
			return fmt.Errorf("unsupported target metric: %s", targetMetric)
		}

		// Stop if target metric is reached
		if metricValue >= targetValue {
			break
		}
	}
	return nil
}
