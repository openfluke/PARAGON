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
// It trains the entire network until the target ADHD metric is reached or max iterations are exceeded.
// If activation is "", a random activation is chosen for each new layer; otherwise, the specified activation is used.
func (n *Network) NormalNASLayerWiseGrowing(
	inputs [][][]float64, targets [][][]float64,
	minNeurons, maxNeurons int, activation string,
	targetMetric string, targetValue float64,
	learningRate float64, // Fixed learning rate
	epochs int, // Epochs per layer addition
	maxIterations int, // Maximum number of layer additions
	earlyStopOnNegativeLoss bool, // Whether to stop training on negative loss
) error {
	// Log start of NAS process
	fmt.Printf("🚀 Starting NormalNASLayerWiseGrowing\n")
	fmt.Printf("  Parameters: minNeurons=%d, maxNeurons=%d, activation=%s, targetMetric=%s, targetValue=%.2f, learningRate=%.4f, epochs=%d, maxIterations=%d, earlyStopOnNegativeLoss=%t\n",
		minNeurons, maxNeurons, activation, targetMetric, targetValue, learningRate, epochs, maxIterations, earlyStopOnNegativeLoss)
	fmt.Printf("  Dataset: %d inputs, %d targets\n", len(inputs), len(targets))

	// Validate inputs
	if minNeurons <= 1 || maxNeurons <= minNeurons {
		err := fmt.Errorf("invalid neuron counts: minNeurons=%d must be > 1, maxNeurons=%d must be > minNeurons", minNeurons, maxNeurons)
		fmt.Printf("❌ Validation error: %v\n", err)
		return err
	}
	if len(inputs) != len(targets) {
		err := fmt.Errorf("mismatched inputs and targets: %d inputs, %d targets", len(inputs), len(targets))
		fmt.Printf("❌ Validation error: %v\n", err)
		return err
	}

	iteration := 0
	previousScore := n.Performance.Score // Track score to detect stagnation
	for iteration < maxIterations {
		iteration++
		fmt.Printf("\n🔄 Iteration %d/%d\n", iteration, maxIterations)

		// Determine activation for the new layer
		var act string
		if activation == "" {
			act = possibleActivations[rand.Intn(len(possibleActivations))]
			fmt.Printf("  🎲 Random activation chosen: %s\n", act)
		} else {
			act = activation
			fmt.Printf("  🛠️ Using specified activation: %s\n", act)
		}

		// Add a new layer with random neuron count and chosen activation
		width := rand.Intn(maxNeurons-minNeurons+1) + minNeurons
		fmt.Printf("  🏗️ Adding layer at index %d with width=%d, height=1, activation=%s\n", n.OutputLayer, width, act)
		n.AddLayer(n.OutputLayer, width, 1, act, true)

		// Train the entire network for specified epochs
		if epochs > 0 {
			fmt.Printf("  🏋️ Training for %d epoch(s) with learning rate %.4f\n", epochs, learningRate)
			n.Train(inputs, targets, epochs, learningRate, earlyStopOnNegativeLoss)
		}

		// Evaluate the network
		fmt.Printf("  📊 Evaluating network\n")
		expected := make([]float64, len(targets)*12)
		actual := make([]float64, len(targets)*12)
		for i := range inputs {
			n.Forward(inputs[i])
			output := n.ExtractOutput()
			if len(output) == 0 || len(targets[i][0]) == 0 {
				fmt.Printf("    ⚠️ Skipping empty prediction at sample %d\n", i)
				continue
			}
			for j := 0; j < 12; j++ {
				expected[i*12+j] = targets[i][0][j]
				actual[i*12+j] = output[j]
			}
		}

		n.EvaluateModel(expected, actual)
		var metricValue float64
		switch targetMetric {
		case "accuracy":
			metricValue = ComputeAccuracy(n, inputs, targets) * 100
			fmt.Printf("    📈 Accuracy: %.2f%%\n", metricValue)
		case "score":
			metricValue = n.Performance.Score
			fmt.Printf("    📈 ADHD Score: %.2f\n", metricValue)
		default:
			err := fmt.Errorf("unsupported target metric: %s", targetMetric)
			fmt.Printf("❌ Evaluation error: %v\n", err)
			return err
		}

		// Log deviation buckets
		fmt.Printf("    📊 Deviation Buckets:\n")
		for bucket, stats := range n.Performance.Buckets {
			fmt.Printf("      - %s: %d samples\n", bucket, stats.Count)
		}

		// Stop if target metric is reached
		if metricValue >= targetValue {
			fmt.Printf("🎯 Target %s (%.2f) reached with value %.2f\n", targetMetric, targetValue, metricValue)
			fmt.Printf("✅ NAS completed after %d iterations\n", iteration)
			return nil
		}

		// Early stopping if score doesn't improve
		if metricValue <= previousScore {
			fmt.Printf("⚠️ ADHD Score (%.2f) did not improve from previous (%.2f). Stopping early.\n", metricValue, previousScore)
			fmt.Printf("✅ NAS stopped after %d iterations\n", iteration)
			return nil
		}
		previousScore = metricValue
	}

	// Log if max iterations reached without achieving target
	fmt.Printf("⚠️ Max iterations (%d) reached without achieving target %s of %.2f. Final ADHD Score: %.2f\n",
		maxIterations, targetMetric, targetValue, n.Performance.Score)
	return nil
}
