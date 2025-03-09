// diffusion.go
package paragon

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// DiffusionConfig defines parameters for the diffusion process
type DiffusionConfig struct {
	NumTimesteps int     // Number of diffusion steps
	MaxLength    int     // Maximum sequence length
	LearningRate float64 // Learning rate for training
	Epochs       int     // Number of training epochs
	Temperature  float64 // Temperature for sampling
	TopK         int     // Top-k sampling parameter
}

// DiffusionModel encapsulates a network with diffusion capabilities
type DiffusionModel struct {
	Network       *Network
	Config        DiffusionConfig
	Tokenizer     *CustomTokenizer
	SpecialTokens map[int]bool
}

// CustomTokenizer (moved here for modularity)
type CustomTokenizer struct {
	Vocab         map[string]int
	ReverseVocab  map[int]string
	VocabSize     int
	SpecialTokens map[int]bool
}

func NewCustomTokenizer(sentences []string) *CustomTokenizer {
	t := &CustomTokenizer{
		Vocab:         make(map[string]int),
		ReverseVocab:  make(map[int]string),
		SpecialTokens: map[int]bool{},
	}
	specials := []string{"[PAD]", "[MASK]", "[CLS]", "[SEP]"}
	for i, tok := range specials {
		t.Vocab[tok] = i
		t.ReverseVocab[i] = tok
		t.SpecialTokens[i] = true
	}
	nextID := len(specials)
	for _, s := range sentences {
		words := strings.Fields(strings.ToLower(s))
		for _, w := range words {
			if _, exists := t.Vocab[w]; !exists {
				t.Vocab[w] = nextID
				t.ReverseVocab[nextID] = w
				nextID++
			}
		}
	}
	t.VocabSize = nextID
	return t
}

func (t *CustomTokenizer) Encode(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	ids := make([]int, len(words))
	for i, w := range words {
		if id, exists := t.Vocab[w]; exists {
			ids[i] = id
		} else {
			ids[i] = t.Vocab["[PAD]"]
		}
	}
	return ids
}

func (t *CustomTokenizer) Decode(ids []int) string {
	words := make([]string, 0, len(ids))
	for _, id := range ids {
		if word, exists := t.ReverseVocab[id]; exists && !t.SpecialTokens[id] {
			words = append(words, word)
		}
	}
	return strings.Join(words, " ")
}

// NewDiffusionModel initializes a diffusion model with a network
func NewDiffusionModel(network *Network, config DiffusionConfig, sentences []string) *DiffusionModel {
	tokenizer := NewCustomTokenizer(sentences)
	return &DiffusionModel{
		Network:       network,
		Config:        config,
		Tokenizer:     tokenizer,
		SpecialTokens: tokenizer.SpecialTokens,
	}
}

func (d *DiffusionModel) AddNoise(tokens []int, t int) []int {
	noiseLevel := math.Min(0.8, float64(t+1)/float64(d.Config.NumTimesteps))
	noisyTokens := make([]int, d.Config.MaxLength)
	padTokenID := d.Tokenizer.Vocab["[PAD]"]
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	if len(tokens) > d.Config.MaxLength {
		copy(noisyTokens, tokens[:d.Config.MaxLength])
	} else {
		copy(noisyTokens, tokens)
		for i := len(tokens); i < d.Config.MaxLength; i++ {
			noisyTokens[i] = padTokenID
		}
	}
	for i := 0; i < d.Config.MaxLength; i++ {
		if rand.Float64() < noiseLevel && noisyTokens[i] != padTokenID {
			noisyTokens[i] = maskTokenID
		}
	}
	return noisyTokens
}

func (d *DiffusionModel) AddNoiseMasked(tokens []int, tVal float64) []int {
	noiseLevel := tVal // No capping at 0.8, full range [0,1]
	noisyTokens := make([]int, d.Config.MaxLength)
	padTokenID := d.Tokenizer.Vocab["[PAD]"]
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	if len(tokens) > d.Config.MaxLength {
		copy(noisyTokens, tokens[:d.Config.MaxLength])
	} else {
		copy(noisyTokens, tokens)
		for i := len(tokens); i < d.Config.MaxLength; i++ {
			noisyTokens[i] = padTokenID
		}
	}
	for i := 0; i < d.Config.MaxLength; i++ {
		if noisyTokens[i] != padTokenID && rand.Float64() < noiseLevel {
			noisyTokens[i] = maskTokenID
		}
	}
	return noisyTokens
}

// diffusion.go (partial update)
func (d *DiffusionModel) Train(sentences []string) {
	data := make([][]int, len(sentences))
	for i, s := range sentences {
		ids := d.Tokenizer.Encode(s)
		if len(ids) > d.Config.MaxLength {
			data[i] = ids[:d.Config.MaxLength]
		} else {
			data[i] = make([]int, d.Config.MaxLength)
			copy(data[i], ids)
			for j := len(ids); j < d.Config.MaxLength; j++ {
				data[i][j] = d.Tokenizer.Vocab["[PAD]"]
			}
		}
	}

	for epoch := 0; epoch < d.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := d.Config.LearningRate * (1 - float64(epoch)/float64(d.Config.Epochs)) // Decay
		for _, tokens := range data {
			t := rand.Intn(d.Config.NumTimesteps)
			noisyTokens := d.AddNoise(tokens, t)
			input := make([][]float64, 1)
			input[0] = make([]float64, d.Config.MaxLength)
			for i, tok := range noisyTokens {
				input[0][i] = float64(tok)
			}
			output := d.Network.ForwardTransformer(input)

			loss := 0.0
			for i := 0; i < d.Config.MaxLength; i++ {
				probs := Softmax(output[i])
				target := tokens[i]
				loss -= math.Log(math.Max(probs[target], 1e-10))
			}
			totalLoss += loss / float64(d.Config.MaxLength)

			errorTerms := make([][]float64, d.Config.MaxLength)
			for i := 0; i < d.Config.MaxLength; i++ {
				probs := Softmax(output[i])
				errorTerms[i] = make([]float64, len(probs))
				for j := 0; j < len(probs); j++ {
					delta := probs[j]
					if j == tokens[i] {
						delta -= 1
					}
					if delta > 5.0 {
						delta = 5.0 // Gradient clipping
					} else if delta < -5.0 {
						delta = -5.0
					}
					errorTerms[i][j] = delta
				}
			}
			d.Network.Backward(errorTerms, lr)
		}
		if epoch%40 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(sentences)))
		}
	}
}

// diffusion.go (only Generate function updated)
func (d *DiffusionModel) Generate() string {
	current := make([]int, d.Config.MaxLength)
	for i := range current {
		current[i] = rand.Intn(d.Tokenizer.VocabSize)
	}
	fmt.Println("Initial random tokens:", d.Tokenizer.Decode(current))

	for t := d.Config.NumTimesteps - 1; t >= 0; t-- {
		noisyInput := d.AddNoise(current, t)
		input := make([][]float64, 1)
		input[0] = make([]float64, d.Config.MaxLength)
		for i, tok := range noisyInput {
			input[0][i] = float64(tok)
		}
		outputFlat := d.Network.ForwardTransformer(input) // [1][MaxLength*VocabSize], e.g., [1][690]

		// Reshape [1][690] to [10][69]
		output := make([][]float64, d.Config.MaxLength)
		for i := 0; i < d.Config.MaxLength; i++ {
			start := i * d.Tokenizer.VocabSize
			end := (i + 1) * d.Tokenizer.VocabSize
			output[i] = outputFlat[0][start:end]
		}

		for i := 0; i < d.Config.MaxLength; i++ {
			probs := Softmax(output[i]) // Now [69] per token
			for j := range probs {
				probs[j] /= d.Config.Temperature
				if d.SpecialTokens[j] {
					probs[j] = 0
				}
			}
			sum := 0.0
			for _, p := range probs {
				sum += p
			}
			for j := range probs {
				probs[j] /= sum
			}
			topK := make([]struct {
				idx  int
				prob float64
			}, len(probs))
			for j := range probs {
				topK[j] = struct {
					idx  int
					prob float64
				}{j, probs[j]}
			}
			sort.Slice(topK, func(i, j int) bool { return topK[i].prob > topK[j].prob })
			topK = topK[:d.Config.TopK]
			idx := rand.Intn(d.Config.TopK)
			current[i] = topK[idx].idx
		}
		fmt.Printf("Step %d, tokens: %s\n", t, d.Tokenizer.Decode(current))
	}
	return d.Tokenizer.Decode(current)
}

func trainMaskedDiffusion(model *DiffusionModel, sentences []string, tokenizer *CustomTokenizer,
	dConfig DiffusionConfig, tConfig TransformerConfig) {

	batchSize := 10
	multithreading := true
	cpuPercent := 0.8
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	numBatches := (len(sentences) + batchSize - 1) / batchSize
	fmt.Printf("Using %d threads (%d%% of %d cores), %d batches\n", numThreads, int(cpuPercent*100), numCores, numBatches)

	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		startTime := time.Now()
		// Learning rate schedule as before:
		lr := dConfig.LearningRate * (1 + math.Cos(float64(epoch)*math.Pi/float64(dConfig.Epochs))) / 2

		// Prepare data (tokenize and pad)
		data := make([][]int, len(sentences))
		for i, s := range sentences {
			ids := tokenizer.Encode(s)
			if len(ids) > dConfig.MaxLength {
				data[i] = ids[:dConfig.MaxLength]
			} else {
				data[i] = make([]int, dConfig.MaxLength)
				copy(data[i], ids)
				for j := len(ids); j < dConfig.MaxLength; j++ {
					data[i][j] = tokenizer.Vocab["[PAD]"]
				}
			}
		}

		// Prepare for multithreaded batch processing
		var wg sync.WaitGroup
		sem := make(chan struct{}, numThreads)
		totalLoss := 0.0
		accumulatedErrorTerms := make([][]float64, len(sentences))
		for i := range accumulatedErrorTerms {
			accumulatedErrorTerms[i] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
		}
		lossChan := make(chan float64, numBatches)
		errorTermsChan := make(chan struct {
			terms    [][]float64
			batchIdx int
		}, numBatches)

		for i := 0; i < len(sentences); i += batchSize {
			end := i + batchSize
			if end > len(sentences) {
				end = len(sentences)
			}
			batchData := data[i:end]
			batchIdx := i / batchSize

			if multithreading {
				wg.Add(1)
				sem <- struct{}{}
				go func(startIdx int, batch [][]int, idx int) {
					defer wg.Done()
					defer func() { <-sem }()

					// For each sample in the batch, sample a continuous t in [0,1]
					batchInputs := make([][]float64, len(batch))
					batchTargets := make([][]int, len(batch))
					noisyBatch := make([][]int, len(batch))
					for j, tokens := range batch {
						tVal := rand.Float64()
						noisyTokens := model.AddNoiseMasked(tokens, tVal)
						noisyBatch[j] = noisyTokens
						batchInputs[j] = make([]float64, dConfig.MaxLength)
						for k, tok := range noisyTokens {
							batchInputs[j][k] = float64(tok)
						}
						batchTargets[j] = tokens
					}

					// Forward pass through the transformer
					batchOutputs := make([][][]float64, len(batch))
					for j, input := range batchInputs {
						singleInput := [][]float64{input}
						batchOutputs[j] = model.Network.ForwardTransformer(singleInput)
					}

					// Compute loss only on positions where the noisy input is [MASK]
					loss := 0.0
					batchErrorTerms := make([][]float64, len(batch))
					for j := 0; j < len(batch); j++ {
						batchErrorTerms[j] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
						for k := 0; k < dConfig.MaxLength; k++ {
							if noisyBatch[j][k] == tokenizer.Vocab["[MASK]"] {
								startIdx := k * tConfig.VocabSize
								endIdx := (k + 1) * tConfig.VocabSize
								probs := Softmax(batchOutputs[j][0][startIdx:endIdx])
								target := batchTargets[j][k]
								loss -= math.Log(math.Max(probs[target], 1e-10))
								// Compute error term (with gradient clipping)
								for m := 0; m < tConfig.VocabSize; m++ {
									delta := probs[m]
									if m == target {
										delta -= 1
									}
									if delta > 5.0 {
										delta = 5.0
									} else if delta < -5.0 {
										delta = -5.0
									}
									batchErrorTerms[j][startIdx+m] = delta
								}
							} else {
								// For nonmasked tokens, set error term to zero.
								for m := 0; m < tConfig.VocabSize; m++ {
									batchErrorTerms[j][k*tConfig.VocabSize+m] = 0
								}
							}
						}
					}
					lossChan <- loss / float64(len(batch))
					errorTermsChan <- struct {
						terms    [][]float64
						batchIdx int
					}{batchErrorTerms, idx}
				}(i, batchData, batchIdx)
			} else {
				// (Non–multithreaded branch omitted for brevity)
			}
		}
		go func() {
			wg.Wait()
			close(lossChan)
			close(errorTermsChan)
		}()
		for l := range lossChan {
			totalLoss += l
		}
		for et := range errorTermsChan {
			start := et.batchIdx * batchSize
			for j, terms := range et.terms {
				if start+j < len(accumulatedErrorTerms) {
					accumulatedErrorTerms[start+j] = terms
				}
			}
		}
		model.Network.Backward(accumulatedErrorTerms, lr)
		totalLoss /= float64(numBatches)
		if epoch%10 == 0 {
			fmt.Printf("%s Epoch %d, Loss: %.4f, Time: %v\n", time.Now().String(), epoch, totalLoss, time.Since(startTime))
			fmt.Println("Generating sample text...")
			sample := model.GenerateMasked()
			fmt.Println("Sample generation:", sample)
		}
	}
}

func (d *DiffusionModel) GenerateMasked() string {
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	current := make([]int, d.Config.MaxLength)
	for i := range current {
		current[i] = maskTokenID
	}
	fmt.Println("Initial fully masked sequence:", d.Tokenizer.Decode(current))
	steps := d.Config.NumTimesteps
	for s := steps; s > 0; s-- {
		// Prepare input as [MaxLength][VocabSize]
		input := make([][]float64, d.Config.MaxLength)
		for k := 0; k < d.Config.MaxLength; k++ {
			input[k] = make([]float64, d.Tokenizer.VocabSize)
			tok := current[k]
			if tok >= 0 && tok < d.Tokenizer.VocabSize {
				input[k][tok] = 1.0
			}
		}
		// Forward pass with correctly shaped input
		outputFlat := d.Network.ForwardTransformer(input)
		output := make([][]float64, d.Config.MaxLength)
		for i := 0; i < d.Config.MaxLength; i++ {
			start := i * d.Tokenizer.VocabSize
			end := (i + 1) * d.Tokenizer.VocabSize
			output[i] = outputFlat[0][start:end]
		}
		// Update masked positions
		maskedPositions := []int{}
		for i := 0; i < d.Config.MaxLength; i++ {
			if current[i] == maskTokenID {
				maskedPositions = append(maskedPositions, i)
				probs := Softmax(output[i])
				maxProb := 0.0
				predToken := maskTokenID
				for j, p := range probs {
					if p > maxProb {
						maxProb = p
						predToken = j
					}
				}
				current[i] = predToken
			}
		}
		if s > 1 { // No remasking at the last step
			pRemask := float64(s-1) / float64(s)
			for _, i := range maskedPositions {
				if rand.Float64() < pRemask {
					current[i] = maskTokenID
				}
			}
		}
		fmt.Printf("Generation step %d: %s\n", s, d.Tokenizer.Decode(current))
	}
	return d.Tokenizer.Decode(current)
}
