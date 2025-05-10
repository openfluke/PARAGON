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
func (n *Network) InitNAS(
	numClones, epochs int,
	baseLR, weightMutRate float64,
	inputs, targets [][][]float64,
	earlyStop bool,
	tryActMutation bool,
) (*Network, float64, bool, error) {

	/* 1. Baseline ADHD of the parent ----------------------------------- */
	exp, act := flattenIO(n, inputs, targets)
	n.EvaluateModel(exp, act)
	parentScore := n.Performance.Score

	/* 2. Serialise parent once (loss‑less JSON) ------------------------- */
	parentBytes, err := json.Marshal(n.toS())
	if err != nil {
		return nil, 0, false, fmt.Errorf("marshal parent: %v", err)
	}

	type res struct {
		net   *Network
		score float64
	}
	results := make([]res, numClones)

	/* 3. Iterate clones ------------------------------------------------- */
	for i := 0; i < numClones; i++ {

		/* 3a. Deep copy via JSON round‑trip */
		var sn sNet
		if err := json.NewDecoder(bytes.NewReader(parentBytes)).Decode(&sn); err != nil {
			return nil, 0, false, fmt.Errorf("clone %d decode: %v", i, err)
		}
		clone := &Network{}
		if err := clone.fromS(sn); err != nil {
			return nil, 0, false, fmt.Errorf("clone %d rebuild: %v", i, err)
		}

		/* 3b. Optional activation‑function mutation on one hidden layer */
		if tryActMutation && clone.OutputLayer > 1 {
			rng := rand.New(rand.NewSource(int64(i)))     // deterministic per clone
			layerIdx := 1 + rng.Intn(clone.OutputLayer-1) // skip input(0) & output

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

		/* 3c. Weight perturbation */
		perturbWeights(clone, weightMutRate, i)

		/* 4. Train */
		for e := 0; e < epochs; e++ {
			lr := baseLR * (1 - float64(e)/float64(epochs)) // linear decay
			clone.Train(inputs, targets, 1, lr, earlyStop)
		}

		/* 5. Evaluate ADHD */
		exp, act := flattenIO(clone, inputs, targets)
		clone.EvaluateModel(exp, act)
		results[i] = res{clone, clone.Performance.Score}
	}

	/* 6. Pick the best clone ------------------------------------------- */
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
func flattenIO(n *Network, ins, tgts [][][]float64) (exp, act []float64) {
	for i := range ins {
		n.Forward(ins[i])
		act = append(act, n.ExtractOutput()...)
		exp = append(exp, tgts[i][0]...)
	}
	return
}

// perturbWeights adds N(0, rate) noise to every connection weight.
func perturbWeights(net *Network, rate float64, seed int) {
	rng := rand.New(rand.NewSource(int64(seed)))
	for l := 1; l < len(net.Layers); l++ { // skip input layer
		for y := 0; y < net.Layers[l].Height; y++ {
			for x := 0; x < net.Layers[l].Width; x++ {
				for k := range net.Layers[l].Neurons[y][x].Inputs {
					net.Layers[l].Neurons[y][x].Inputs[k].Weight += rng.NormFloat64() * rate
				}
			}
		}
	}
}

/*
IterativeInitNAS – multiple NAS rounds until target ADHD reached or attempts exhausted.
Returns bestNet, bestADHD.
*/
func (n *Network) IterativeInitNAS(
	numClones, nasEpochs int,
	baseLR, weightMutRate float64,
	earlyStop, allowActMut bool,
	targetADHD float64,
	maxAttempts int,
	inputs, targets [][][]float64,
) (*Network, float64) {

	bestNet := n
	bestScore := n.Performance.Score
	fmt.Printf("🔰 start ADHD %.2f\n", bestScore)

	for att := 1; att <= maxAttempts && bestScore < targetADHD; att++ {
		fmt.Printf("\n🔄 NAS round %d / %d  (best %.2f)\n", att, maxAttempts, bestScore)

		cNet, cScore, improved, err := bestNet.InitNAS(
			numClones, nasEpochs, baseLR, weightMutRate,
			inputs, targets, earlyStop, allowActMut)

		if err != nil {
			fmt.Printf("❌ round %d failed: %v\n", att, err)
			continue
		}
		if improved {
			fmt.Printf("✅ improved: %.2f → %.2f\n", bestScore, cScore)
			bestNet, bestScore = cNet, cScore
		} else {
			fmt.Printf("⚠️  no gain (%.2f)\n", bestScore)
			weightMutRate *= 1.2 // widen search next round
		}
	}
	return bestNet, bestScore
}
