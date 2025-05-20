package paragon

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
)

/*
InitNAS – single round of layer‑wise NAS
----------------------------------------
  - Creates numClones JSON deep‑copies of *n.
  - Each clone:
    – optional activation mutation on one hidden layer
    – Gaussian weight jitter
    – short training run (epochs)
  - Returns the clone with the highest ADHD score.

Return values:

	bestNet, bestScore, improved(relative‑to‑parent), error
*/
func (n *Network[T]) InitNAS(
	numClones, epochs int,
	baseLR, weightMutRate float64,
	inputs, targets [][][]float64,
	earlyStop bool,
	tryActMutation bool,
	clipUpper T,
	clipLower T,
) (*Network[T], float64, bool, error) {

	// 1. Evaluate parent performance
	exp, act := flattenIO(n, inputs, targets)
	n.EvaluateModel(exp, act)
	parentScore := n.Performance.Score

	// 2. Serialize parent network
	parentBytes, err := json.Marshal(n.ToS())
	if err != nil {
		return nil, 0, false, fmt.Errorf("marshal parent: %v", err)
	}

	type res struct {
		net   *Network[T]
		score float64
	}
	results := make([]res, numClones)

	// 3. Clone, mutate, train, evaluate
	for i := 0; i < numClones; i++ {
		var sn sNet
		if err := json.NewDecoder(bytes.NewReader(parentBytes)).Decode(&sn); err != nil {
			return nil, 0, false, fmt.Errorf("clone %d decode: %v", i, err)
		}
		clone := &Network[T]{}
		if err := clone.FromS(sn); err != nil {
			return nil, 0, false, fmt.Errorf("clone %d rebuild: %v", i, err)
		}

		// 3b. Optional activation mutation
		if tryActMutation && clone.OutputLayer > 1 {
			rng := rand.New(rand.NewSource(int64(i)))
			layerIdx := 1 + rng.Intn(clone.OutputLayer-1)
			candidates := []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"}
			newAct := candidates[rng.Intn(len(candidates))]
			for y := 0; y < clone.Layers[layerIdx].Height; y++ {
				for x := 0; x < clone.Layers[layerIdx].Width; x++ {
					clone.Layers[layerIdx].Neurons[y][x].Activation = newAct
				}
			}
			if n.Debug {
				fmt.Printf("🔧 clone %d  layer %d activation → %s\n", i, layerIdx, newAct)
			}
		}

		// 3c. Perturb weights
		perturbWeights(clone, weightMutRate, i)

		// 4. Train with clipping
		for e := 0; e < epochs; e++ {
			lr := baseLR * (1 - float64(e)/float64(epochs))
			clone.Train(inputs, targets, 1, lr, earlyStop, clipUpper, clipLower)
		}

		// 5. Evaluate clone performance
		exp, act := flattenIO(clone, inputs, targets)
		clone.EvaluateModel(exp, act)
		results[i] = res{clone, clone.Performance.Score}
	}

	// 6. Select best
	best := results[0]
	for _, r := range results[1:] {
		if r.score > best.score {
			best = r
		}
	}
	return best.net, best.score, best.score > parentScore, nil
}

/*──────────────────── helper functions ───────────────────*/

// flattenIO produces two flat slices suitable for EvaluateModel.
func flattenIO[T Numeric](n *Network[T], ins, tgts [][][]float64) (exp, act []float64) {
	for i := range ins {
		n.Forward(ins[i])
		act = append(act, n.ExtractOutput()...) // []float64 from T
		exp = append(exp, tgts[i][0]...)        // Expected: flat one-hot row
	}
	return
}

// perturbWeights adds N(0, rate) noise to every connection weight.
func perturbWeights[T Numeric](net *Network[T], rate float64, seed int) {
	rng := rand.New(rand.NewSource(int64(seed)))

	for l := 1; l < len(net.Layers); l++ { // skip input layer
		layer := net.Layers[l]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				for k := range neuron.Inputs {
					// Only perturb if T supports float math
					switch any(neuron.Inputs[k].Weight).(type) {
					case float32, float64:
						noise := rng.NormFloat64() * rate
						neuron.Inputs[k].Weight += T(noise)
					default:
						// Skip perturbation for int/uint types
					}
				}
			}
		}
	}
}

/*
IterativeInitNAS – multiple NAS rounds until target ADHD reached or attempts exhausted.
Returns bestNet, bestADHD.
*/
func (n *Network[T]) IterativeInitNAS(
	numClones, nasEpochs int,
	baseLR, weightMutRate float64,
	earlyStop, allowActMut bool,
	targetADHD float64,
	maxAttempts int,
	inputs, targets [][][]float64,
	clipUpper, clipLower T,
) (*Network[T], float64) {

	bestNet := n
	bestScore := n.Performance.Score
	fmt.Printf("🔰 start ADHD %.2f\n", bestScore)

	for att := 1; att <= maxAttempts && bestScore < targetADHD; att++ {
		fmt.Printf("\n🔄 NAS round %d / %d  (best %.2f)\n", att, maxAttempts, bestScore)

		cNet, cScore, improved, err := bestNet.InitNAS(
			numClones, nasEpochs, baseLR, weightMutRate,
			inputs, targets, earlyStop, allowActMut,
			clipUpper, clipLower, // 👈 added bounds here
		)

		if err != nil {
			fmt.Printf("❌ round %d failed: %v\n", att, err)
			continue
		}

		if improved {
			fmt.Printf("✅ improved: %.2f → %.2f\n", bestScore, cScore)
			bestNet, bestScore = cNet, cScore
		} else {
			fmt.Printf("⚠️  no gain (%.2f)\n", bestScore)
			weightMutRate *= 1.2 // widen mutation range
		}
	}

	return bestNet, bestScore
}
