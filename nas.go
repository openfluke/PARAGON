package paragon

import (
	"encoding/json"
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

// InitNAS performs NAS by cloning the network via JSON,
// applying richer mutations (architecture + weight perturbations),
// training, and selecting the best.
//
// - numClones: how many candidates to try
// - epochs:   number of epochs per clone
// - lr:       base learning rate
// - inputs, targets: data for train+eval
// - earlyStop: whether to stop epochs early on negative loss
func (n *Network) InitNAS(
	numClones, epochs int,
	baseLR, weightMutRate float64,
	inputs, targets [][][]float64,
	earlyStopOnNegativeLoss bool,
	tryNewActivationFunction bool,
) (*Network, float64, bool, error) {

	// 1. compute original score
	origExp, origAct := flattenIO(n, inputs, targets)
	n.EvaluateModel(origExp, origAct)
	origScore := n.Performance.Score

	// 2. serialize once
	serial := n.toSerializable()
	jsonBase, err := json.Marshal(serial)
	if err != nil {
		return nil, 0, false, fmt.Errorf("serialize error: %v", err)
	}

	type result struct {
		net   *Network
		score float64
	}
	results := make([]result, numClones)

	for i := 0; i < numClones; i++ {
		// 3a. clone
		clone := &Network{
			Layers:      []Grid{},
			AttnWeights: []AttentionWeights{},
			Performance: NewADHDPerformance(),
			Config:      n.Config,
		}
		var s SerializableNetwork
		if err := json.Unmarshal(jsonBase, &s); err != nil {
			return nil, 0, false, fmt.Errorf("clone unmarshal[%d]: %v", i, err)
		}
		if err := clone.fromSerializable(s); err != nil {
			return nil, 0, false, fmt.Errorf("clone load[%d]: %v", i, err)
		}

		// 3b. mutate one hidden layer’s activation function
		if len(clone.Layers) > 2 && tryNewActivationFunction {

			// pick a random hidden layer index (1 … OutputLayer-1)
			hidCount := clone.OutputLayer - 1
			layerIdx := 1 + rand.Intn(hidCount)

			// choose a new activation
			acts := []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"}
			newAct := acts[rand.Intn(len(acts))]

			// apply it to every neuron in that layer
			for y := 0; y < clone.Layers[layerIdx].Height; y++ {
				for x := 0; x < clone.Layers[layerIdx].Width; x++ {
					clone.Layers[layerIdx].Neurons[y][x].Activation = newAct
				}
			}
			if n.Debug {
				fmt.Printf("🔧 Clone %d: layer %d activation → %s\n", i, layerIdx, newAct)
			}
		}

		// 3c. perturb weights slightly
		perturbWeights(clone, weightMutRate, i)

		// 4. train
		for e := 0; e < epochs; e++ {
			lr := baseLR * (1.0 - float64(e)/float64(epochs)) // decay
			clone.Train(inputs, targets, 1, lr, earlyStopOnNegativeLoss)
		}

		// 5. evaluate
		exp, act := flattenIO(clone, inputs, targets)
		clone.EvaluateModel(exp, act)
		results[i] = result{clone, clone.Performance.Score}
	}

	// 6. pick best
	bestIdx, bestScore := 0, results[0].score
	for i, r := range results {
		if r.score > bestScore {
			bestIdx, bestScore = i, r.score
		}
	}
	improved := bestScore > origScore
	return results[bestIdx].net, bestScore, improved, nil
}

// flattenIO prepares flat expected/actual slices for evaluation
func flattenIO(n *Network, inputs, targets [][][]float64) (exp, act []float64) {
	for i := range inputs {
		n.Forward(inputs[i])
		out := n.ExtractOutput()
		tgt := targets[i][0]
		exp = append(exp, tgt...)
		act = append(act, out...)
	}
	return
}

// perturbWeights does a small gaussian perturbation to all connection weights
func perturbWeights(n *Network, rate float64, seed int) {
	rnd := rand.New(rand.NewSource(int64(seed)))
	for l := 1; l < len(n.Layers); l++ {
		for y := 0; y < n.Layers[l].Height; y++ {
			for x := 0; x < n.Layers[l].Width; x++ {
				for i := range n.Layers[l].Neurons[y][x].Inputs {
					δ := rnd.NormFloat64() * rate
					n.Layers[l].Neurons[y][x].Inputs[i].Weight += δ
				}
			}
		}
	}
}

// IterativeNAS runs multiple rounds of NormalNASLayerWiseGrowingEnhanced on a network,
// stopping early if the target ADHD score is reached.
// Returns the best‐found network and its ADHD score.
func (n *Network) IterativeInitNAS(
	numClones, nasEpochs int,
	baseLR, weightMutationRate float64,
	earlyStop, enableActMutation bool,
	targetADHD float64,
	maxAttempts int,
	inputs, targets [][][]float64,
) (*Network, float64) {
	parentNet := n
	parentScore := n.Performance.Score
	fmt.Printf("🔰 Starting ADHD Score: %.2f\n", parentScore)

	for attempt := 1; attempt <= maxAttempts && parentScore < targetADHD; attempt++ {
		fmt.Printf("\n🔄 NAS Attempt %d / %d (current ADHD %.2f)\n",
			attempt, maxAttempts, parentScore,
		)

		candNet, candScore, improved, err := parentNet.InitNAS(
			numClones,
			nasEpochs,
			baseLR,
			weightMutationRate,
			inputs,
			targets,
			earlyStop,
			enableActMutation, // <-- pass the new flag here
		)
		if err != nil {
			fmt.Printf("❌ NAS attempt %d failed: %v – skipping\n", attempt, err)
			continue
		}

		if improved && candScore > parentScore {
			fmt.Printf("✅ Improved: %.2f → %.2f\n", parentScore, candScore)
			parentNet = candNet
			parentScore = candScore
		} else {
			fmt.Printf("⚠️ No improvement this round (best %.2f)\n", parentScore)
			weightMutationRate *= 1.2
		}
	}

	return parentNet, parentScore
}
