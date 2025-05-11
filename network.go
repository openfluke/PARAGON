package paragon

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Grid represents a 2D layer of neurons
type Grid struct {
	Width   int         // Number of neurons along x-axis
	Height  int         // Number of neurons along y-axis
	Neurons [][]*Neuron // 2D slice of neuron pointers

	ReplayOffset int    // Offset to replay (e.g., -1 to replay previous layer)
	ReplayPhase  string // "before" or "after" to determine when to replay
	MaxReplay    int    // Maximum number of replays for this layer

	//dynamic replay
	ReplayEnabled    bool                            // Manual switch
	ReplayBudget     int                             // Max allowed replays
	ReplayGateFunc   func(input [][]float64) float64 // Output: [0.0–1.0]
	ReplayGateToReps func(score float64) int         // Maps score to actual replays

	CachedOutputs []float64 // used for entropy-based replay gating
	Dense         *DenseCache
}

type DenseCache struct {
	Weights []float64 // [out][in] flattened
	Biases  []float64 // length = out
	Out     []float64 // temp output values
	DGrad   []float64 // temp backprop error (same shape as Out)
}

// Neuron represents a single unit in the grid
type Neuron struct {
	ID         int          // Unique ID
	Value      float64      // Current activation
	Bias       float64      // Bias term
	Activation string       // e.g., "relu", "sigmoid"
	Type       string       // e.g., "dense", extensible for future types
	Inputs     []Connection // Incoming connections
	IsNew      bool         // Flag for newly added neurons
	Dimension  *Network
	RevValue   float64
}

// Connection defines a link from a source neuron
type Connection struct {
	SourceLayer int     // Layer index of source neuron
	SourceX     int     // X-coordinate of source neuron
	SourceY     int     // Y-coordinate of source neuron
	Weight      float64 // Weight of the connection
}

// Network encapsulates the entire model
type Network struct {
	Layers      []Grid
	InputLayer  int
	OutputLayer int
	Debug       bool

	NHeads     int
	FFWeights1 [][]float64 // [DModel][FeedForward]
	FFBias1    []float64   // [FeedForward]
	FFWeights2 [][]float64 // [FeedForward][DModel]
	FFBias2    []float64   // [DModel]

	Performance *ADHDPerformance
	Composite   *CompositePerformance
	ReplayStats map[int][]int // layer index -> []replay count per sample
	UseBLAS     bool
}

// NewNetwork initializes a network with specified layer sizes, activations, and connectivity
func NewNetwork(layerSizes []struct{ Width, Height int }, activations []string, fullyConnected []bool) *Network {
	if len(layerSizes) != len(activations) || len(layerSizes) != len(fullyConnected) {
		panic("mismatched layer sizes, activations, or connectivity settings")
	}
	n := &Network{
		Layers:      make([]Grid, len(layerSizes)),
		InputLayer:  0,
		OutputLayer: len(layerSizes) - 1,
		NHeads:      0, // Default to 0, set by NewTransformerEncoder if needed
		Performance: NewADHDPerformance(),
		ReplayStats: make(map[int][]int),
	}
	idCounter := 0
	for i, size := range layerSizes {
		grid := Grid{
			Width:   size.Width,
			Height:  size.Height,
			Neurons: make([][]*Neuron, size.Height),
		}
		for y := 0; y < size.Height; y++ {
			grid.Neurons[y] = make([]*Neuron, size.Width)
			for x := 0; x < size.Width; x++ {
				grid.Neurons[y][x] = &Neuron{
					ID:         idCounter,
					Bias:       0.0,
					Activation: activations[i],
					Type:       "dense",
				}
				idCounter++
			}
		}
		n.Layers[i] = grid
	}
	n.ConnectLayers(fullyConnected)
	return n
}

func (n *Network) getFullyConnectedInputs(srcLayer int, srcGrid Grid) []Connection {
	fanIn := srcGrid.Width * srcGrid.Height
	conns := make([]Connection, 0, fanIn)
	for y := 0; y < srcGrid.Height; y++ {
		for x := 0; x < srcGrid.Width; x++ {
			conns = append(conns, Connection{
				SourceLayer: srcLayer,
				SourceX:     x,
				SourceY:     y,
				Weight:      rand.NormFloat64() * math.Sqrt(2.0/float64(fanIn)),
			})
		}
	}
	return conns
}

func (n *Network) getLocalConnections(srcLayer, centerX, centerY, size, stride int) []Connection {
	half := size / 2
	srcGrid := n.Layers[srcLayer]
	conns := []Connection{}
	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			srcX := centerX*stride + dx
			srcY := centerY*stride + dy
			if srcX >= 0 && srcX < srcGrid.Width && srcY >= 0 && srcY < srcGrid.Height {
				conns = append(conns, Connection{
					SourceLayer: srcLayer,
					SourceX:     srcX,
					SourceY:     srcY,
					Weight:      rand.NormFloat64() * math.Sqrt(2.0/float64(size*size)),
				})
			}
		}
	}
	return conns
}

// ConnectLayers sets up connections based on connectivity settings
func (n *Network) ConnectLayers(fullyConnected []bool) {
	for l := 1; l < len(n.Layers); l++ {
		prevLayer := n.Layers[l-1]
		currLayer := n.Layers[l]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				if fullyConnected[l] {
					neuron.Inputs = n.getFullyConnectedInputs(l-1, prevLayer)
				} else {
					// Use 5x5 receptive field with stride 1
					neuron.Inputs = n.getLocalConnections(l-1, x, y, 5, 1)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Low‑level helpers (identical logic to your originals)
// ---------------------------------------------------------------------------
func (n *Network) forwardLayer(l int) {
	curr := n.Layers[l]
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			neuron := curr.Neurons[y][x]
			sum := neuron.Bias
			for _, c := range neuron.Inputs {
				src := n.Layers[c.SourceLayer].Neurons[c.SourceY][c.SourceX]
				sum += src.Value * c.Weight
			}
			neuron.Value = applyActivation(sum, neuron.Activation)
			if n.Debug {
				fmt.Printf("Layer %d, Neuron(%d,%d)=%.4f\n", l, x, y, neuron.Value)
			}
		}
	}
}

func (n *Network) backwardLayer(
	l int,
	err [][][]float64,
	lr float64,
) {
	curr, prev := n.Layers[l], n.Layers[l-1]

	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			neuron := curr.Neurons[y][x]
			local := err[l][y][x]

			// bias update
			neuron.Bias += lr * local

			// weights
			for i, c := range neuron.Inputs {
				src := prev.Neurons[c.SourceY][c.SourceX]
				grad := local * src.Value
				if grad > 5 {
					grad = 5
				} else if grad < -5 {
					grad = -5
				}
				neuron.Inputs[i].Weight += lr * grad

				if l-1 > 0 {
					err[l-1][c.SourceY][c.SourceX] += local * c.Weight
				}
			}
		}
	}
	// chain rule for next layer down
	if l-1 > 0 {
		for y := 0; y < prev.Height; y++ {
			for x := 0; x < prev.Width; x++ {
				err[l-1][y][x] *= activationDerivative(
					prev.Neurons[y][x].Value,
					prev.Neurons[y][x].Activation,
				)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Forward pass with optional layer‑replay
// ---------------------------------------------------------------------------
func (n *Network) Forward(inputs [][]float64) {
	in := n.Layers[n.InputLayer]
	if len(inputs) != in.Height || len(inputs[0]) != in.Width {
		panic(fmt.Sprintf("input mismatch: want %dx%d, got %dx%d",
			in.Height, in.Width, len(inputs), len(inputs[0])))
	}
	for y := 0; y < in.Height; y++ {
		for x := 0; x < in.Width; x++ {
			in.Neurons[y][x].Value = inputs[y][x]
		}
	}

	replayed := map[int]int{}

	for l := 1; l < len(n.Layers); l++ {
		layer := &n.Layers[l]

		// ► replay "before"
		if layer.ReplayOffset != 0 && layer.ReplayPhase == "before" && replayed[l] < layer.MaxReplay {
			start := max(l+layer.ReplayOffset, n.InputLayer+1)
			for i := start; i <= l; i++ {
				n.forwardDispatch(i)
				n.Layers[i].CachedOutputs = n.Layers[i].GetOutputValues()
			}
			replayed[l]++
		}

		// ► main forward
		n.forwardDispatch(l)
		layer.CachedOutputs = layer.GetOutputValues()

		// ► replay "after"
		if layer.ReplayOffset != 0 && layer.ReplayPhase == "after" && replayed[l] < layer.MaxReplay {
			start := max(l+layer.ReplayOffset, n.InputLayer+1)
			for i := start; i <= l; i++ {
				n.forwardDispatch(i)
				n.Layers[i].CachedOutputs = n.Layers[i].GetOutputValues()
			}
			replayed[l]++
		}

		// ► dynamic gated replay
		if layer.ReplayEnabled && layer.ReplayGateFunc != nil {
			score := layer.ReplayGateFunc(nil)
			nreps := layer.ReplayBudget
			if layer.ReplayGateToReps != nil {
				nreps = layer.ReplayGateToReps(score)
			}
			if nreps > layer.ReplayBudget {
				nreps = layer.ReplayBudget
			}
			for i := 0; i < nreps; i++ {
				n.forwardDispatch(l)
				layer.CachedOutputs = layer.GetOutputValues()
				replayed[l]++
			}
		}
	}

	// ► Softmax normalization if needed
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// ---------------------------------------------------------------------------
// Back‑prop with optional layer‑replay  (incl. attention weight update)
// ---------------------------------------------------------------------------
func (n *Network) Backward(targets [][]float64, lr float64) {
	nLayers := len(n.Layers)
	err := make([][][]float64, nLayers)
	for l := range err {
		err[l] = make([][]float64, n.Layers[l].Height)
		for y := range err[l] {
			err[l][y] = make([]float64, n.Layers[l].Width)
		}
	}

	// Output layer error
	out := n.Layers[n.OutputLayer]
	for y := 0; y < out.Height; y++ {
		for x := 0; x < out.Width; x++ {
			neuron := out.Neurons[y][x]
			err[n.OutputLayer][y][x] =
				(targets[y][x] - neuron.Value) *
					activationDerivative(neuron.Value, neuron.Activation)
		}
	}

	replayed := map[int]int{}

	for l := n.OutputLayer; l > 0; l-- {
		layer := &n.Layers[l]

		// ► replay "before"
		if layer.ReplayOffset != 0 && layer.ReplayPhase == "before" && replayed[l] < layer.MaxReplay {
			stop := max(l+layer.ReplayOffset, n.InputLayer+1)
			for i := l; i >= stop; i-- {
				n.backwardDispatch(i, err, lr)
			}
			replayed[l]++
		}

		// ► main backprop
		n.backwardDispatch(l, err, lr)

		// ► replay "after"
		if layer.ReplayOffset != 0 && layer.ReplayPhase == "after" && replayed[l] < layer.MaxReplay {
			stop := max(l+layer.ReplayOffset, n.InputLayer+1)
			for i := l; i >= stop; i-- {
				n.backwardDispatch(i, err, lr)
			}
			replayed[l]++
		}

		// ► dynamic gated replay
		if layer.ReplayEnabled && layer.ReplayGateFunc != nil {
			score := layer.ReplayGateFunc(nil)
			nreps := layer.ReplayBudget
			if layer.ReplayGateToReps != nil {
				nreps = layer.ReplayGateToReps(score)
			}
			if nreps > layer.ReplayBudget {
				nreps = layer.ReplayBudget
			}
			for i := 0; i < nreps; i++ {
				n.backwardDispatch(l, err, lr)
				replayed[l]++
			}
			if n.ReplayStats != nil {
				n.LogReplay(l, nreps)
			}
		}
	}
}

// Train runs the training loop
func (n *Network) Train(inputs [][][]float64, targets [][][]float64, epochs int, learningRate float64, earlyStopOnNegativeLoss bool) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))
		shuffledInputs := make([][][]float64, len(inputs))
		shuffledTargets := make([][][]float64, len(targets))
		for i, p := range perm {
			shuffledInputs[i] = inputs[p]
			shuffledTargets[i] = targets[p]
		}
		for b := 0; b < len(shuffledInputs); b++ {
			n.Forward(shuffledInputs[b])
			loss := n.ComputeLoss(shuffledTargets[b])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				fmt.Printf("⚠️ Negative loss (%.4f) detected at sample %d, epoch %d. Stopping training early.\n", loss, b, epoch)
				return // Stop training for this epoch and exit
			}
			totalLoss += loss
			n.Backward(shuffledTargets[b], learningRate)
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

func (n *Network) TrainTest(inputs [][][]float64, targets [][][]float64, epochs int, learningRate float64, earlyStopOnNegativeLoss bool) {
	const lambda = 0.01 // 💡 scaling factor for replay penalty

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))
		shuffledInputs := make([][][]float64, len(inputs))
		shuffledTargets := make([][][]float64, len(targets))
		for i, p := range perm {
			shuffledInputs[i] = inputs[p]
			shuffledTargets[i] = targets[p]
		}

		for b := 0; b < len(shuffledInputs); b++ {
			n.ResetReplayStats()

			// 🔁 Forward pass
			n.Forward(shuffledInputs[b])

			// 🎯 Base loss
			loss := n.ComputeLoss(shuffledTargets[b])

			// 📊 Penalty = λ * total replays * (1 - avg entropy)
			totalReplays := n.TotalReplayThisSample()
			avgEntropy := n.AvgEntropyThisSample()
			penalty := lambda * float64(totalReplays) * (1.0 - avgEntropy)
			loss += penalty

			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				fmt.Printf("⚠️  Negative loss (%.4f) at sample %d, epoch %d. Early stopping.\n", loss, b, epoch)
				return
			}

			totalLoss += loss

			// 🔁 Backward
			n.Backward(shuffledTargets[b], learningRate)
		}

		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

func (n *Network) AvgEntropyThisSample() float64 {
	totalEntropy := 0.0
	num := 0
	for _, layer := range n.Layers {
		if layer.ReplayEnabled && layer.ReplayGateFunc != nil {
			score := layer.ReplayGateFunc(nil)
			totalEntropy += score
			num++
		}
	}
	if num == 0 {
		return 0.0
	}
	return totalEntropy / float64(num)
}

func (n *Network) TrainTestWithLambda(
	inputs, targets [][][]float64,
	epochs int,
	learningRate float64,
	earlyStopOnNegativeLoss bool,
	lambda float64,
) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))

		for _, i := range perm {
			n.ResetReplayStats()
			n.Forward(inputs[i])
			loss := n.ComputeLoss(targets[i])
			replayPenalty := float64(n.TotalReplayThisSample())
			loss += lambda * replayPenalty

			if math.IsNaN(loss) || (earlyStopOnNegativeLoss && loss < 0) {
				continue
			}
			totalLoss += loss
			n.Backward(targets[i], learningRate)
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

func (n *Network) TotalReplayThisSample() int {
	total := 0
	for _, r := range n.ReplayStats {
		if len(r) > 0 {
			total += r[len(r)-1] // last entry is current sample's count
		}
	}
	return total
}

// ComputeLoss calculates the loss for a sample
func (n *Network) ComputeLoss(target [][]float64) float64 {
	loss := 0.0
	outputLayer := n.Layers[n.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			outputVal := outputLayer.Neurons[y][x].Value
			targetVal := target[y][x]
			if outputVal <= 0 {
				outputVal = 1e-10
			}
			loss += -targetVal * math.Log(outputVal)
		}
	}
	return loss
}

// AddNeuronsToLayer adds neurons to a specified layer
func (n *Network) AddNeuronsToLayer(layerIdx, numToAdd int) {
	if layerIdx <= n.InputLayer || layerIdx > n.OutputLayer {
		return
	}
	grid := &n.Layers[layerIdx]
	newHeight := grid.Height + numToAdd
	newNeurons := make([][]*Neuron, newHeight)
	copy(newNeurons, grid.Neurons)
	for y := grid.Height; y < newHeight; y++ {
		newNeurons[y] = make([]*Neuron, grid.Width)
		for x := 0; x < grid.Width; x++ {
			newNeurons[y][x] = &Neuron{
				ID:         n.getNextID(),
				Bias:       0.0,
				Activation: grid.Neurons[0][0].Activation,
				Type:       "dense",
				IsNew:      true,
			}
			prevLayer := n.Layers[layerIdx-1]
			if layerIdx == 1 && len(n.Layers) == 3 { // Initial setup check
				newNeurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
			} else {
				newNeurons[y][x].Inputs = n.getLocalConnections(layerIdx-1, x, y, 3, 1) // Added stride 1
			}
		}
	}
	grid.Neurons = newNeurons
	grid.Height = newHeight

	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := len(nextLayer.Neurons[0][0].Inputs) == grid.Width*grid.Height
		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, *grid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 3, 1) // Added stride 1
				}
			}
		}
	}
	if n.Debug {
		fmt.Printf("Added %d neurons to layer %d, new height: %d\n", numToAdd, layerIdx, grid.Height)
	}
}

// getNextID generates a unique neuron ID
func (n *Network) getNextID() int {
	maxID := 0
	for _, layer := range n.Layers {
		for _, row := range layer.Neurons {
			for _, neuron := range row {
				if neuron.ID > maxID {
					maxID = neuron.ID
				}
			}
		}
	}
	return maxID + 1
}

// ApplySoftmax normalizes output layer values
func (n *Network) ApplySoftmax() {
	outputGrid := n.Layers[n.OutputLayer]
	values := make([]float64, outputGrid.Width*outputGrid.Height)
	idx := 0
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			values[idx] = outputGrid.Neurons[y][x].Value
			idx++
		}
	}
	softmaxValues := Softmax(values)
	idx = 0
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			outputGrid.Neurons[y][x].Value = softmaxValues[idx]
			idx++
		}
	}
}

// AddLayer inserts a new layer at the specified index with given dimensions and connectivity
func (n *Network) AddLayer(layerIdx int, width, height int, activation string, fullyConnectedToPrev bool) {
	if layerIdx < n.InputLayer || layerIdx > n.OutputLayer+1 {
		return
	}
	newGrid := Grid{
		Width:   width,
		Height:  height,
		Neurons: make([][]*Neuron, height),
	}
	idCounter := n.getNextID()
	for y := 0; y < height; y++ {
		newGrid.Neurons[y] = make([]*Neuron, width)
		for x := 0; x < width; x++ {
			newGrid.Neurons[y][x] = &Neuron{
				ID:         idCounter,
				Bias:       0.0,
				Activation: activation,
				Type:       "dense",
			}
			idCounter++
		}
	}
	newLayers := make([]Grid, len(n.Layers)+1)
	copy(newLayers[:layerIdx], n.Layers[:layerIdx])
	newLayers[layerIdx] = newGrid
	copy(newLayers[layerIdx+1:], n.Layers[layerIdx:])
	n.Layers = newLayers
	n.OutputLayer = len(n.Layers) - 1

	if layerIdx > n.InputLayer {
		prevLayer := n.Layers[layerIdx-1]
		for y := 0; y < newGrid.Height; y++ {
			for x := 0; x < newGrid.Width; x++ {
				if fullyConnectedToPrev {
					newGrid.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
				} else {
					newGrid.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx-1, x, y, 5, 1)
				}
			}
		}
	}
	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := layerIdx == n.OutputLayer-1
		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, newGrid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 5, 1)
				}
			}
		}
	}
	if n.Debug {
		fmt.Printf("Added new layer at index %d with dimensions %dx%d\n", layerIdx, width, height)
	}
}

// network.go
/*func (n *Network) SetLayerDimension(layerIdx int, subNetwork *Network) {
	if layerIdx < 0 || layerIdx >= len(n.Layers) {
		panic(fmt.Sprintf("invalid layer index: %d", layerIdx))
	}
	n.Layers[layerIdx].Dimension = subNetwork
}*/

func (n *Network) GetOutput() []float64 {
	outputLayer := n.Layers[n.OutputLayer] // Access the output layer
	output := make([]float64, outputLayer.Width)
	for x := 0; x < outputLayer.Width; x++ {
		output[x] = outputLayer.Neurons[0][x].Value // Assuming neurons are stored as [Height][Width]
	}
	return output
}

// GetOutputValues returns a flattened 1D slice of all neuron values in this grid.
func (g *Grid) GetOutputValues() []float64 {
	values := make([]float64, 0, g.Width*g.Height)
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			values = append(values, g.Neurons[y][x].Value)
		}
	}
	return values
}

func (n *Network) ResetReplayStats() {
	for k := range n.ReplayStats {
		n.ReplayStats[k] = nil
	}
}

func (n *Network) LogReplay(l int, count int) {
	if _, exists := n.ReplayStats[l]; !exists {
		n.ReplayStats[l] = []int{}
	}
	n.ReplayStats[l] = append(n.ReplayStats[l], count)
}

func (n *Network) BLASForwardLayer(l int) {
	grid := &n.Layers[l]
	dc := grid.Dense
	if dc == nil {
		panic(fmt.Sprintf("Layer %d has no DenseCache; cannot use BLASForwardLayer", l))
	}

	prev := &n.Layers[l-1]
	inVec := prev.GetOutputValues() // Length = in
	outLen := len(dc.Biases)

	// y = W·x   (out = W * inVec)
	gA := blas64.General{
		Rows:   outLen,
		Cols:   len(inVec),
		Stride: len(inVec),
		Data:   dc.Weights,
	}
	gX := blas64.General{
		Rows:   len(inVec),
		Cols:   1,
		Stride: 1,
		Data:   inVec,
	}
	gY := blas64.General{
		Rows:   outLen,
		Cols:   1,
		Stride: 1,
		Data:   dc.Out,
	}

	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, gA, gX, 0.0, gY)

	// Add bias and apply activation
	idx := 0
	for y := 0; y < grid.Height; y++ {
		for x := 0; x < grid.Width; x++ {
			v := dc.Out[idx] + dc.Biases[idx]
			v = applyActivation(v, grid.Neurons[y][x].Activation)
			grid.Neurons[y][x].Value = v
			idx++
		}
	}
}

func (n *Network) BLASBackwardLayer(l int, err [][][]float64, lr float64) {
	curr := &n.Layers[l]
	prev := &n.Layers[l-1]
	dc := curr.Dense
	if dc == nil {
		panic(fmt.Sprintf("Layer %d has no DenseCache; cannot use BLASBackwardLayer", l))
	}

	in := prev.GetOutputValues() // [in]
	outLen := len(dc.Biases)
	inLen := len(in)

	// Flatten current error into dc.DGrad
	idx := 0
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			dc.DGrad[idx] = err[l][y][x]
			idx++
		}
	}

	// Weight update: W += lr * (dL/dY) ⊗ X
	gDL := blas64.General{Rows: outLen, Cols: 1, Stride: 1, Data: dc.DGrad}
	gX := blas64.General{Rows: 1, Cols: inLen, Stride: inLen, Data: in}
	gW := blas64.General{Rows: outLen, Cols: inLen, Stride: inLen, Data: dc.Weights}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, lr, gDL, gX, 1.0, gW)

	// Bias update
	for i := range dc.Biases {
		dc.Biases[i] += lr * dc.DGrad[i]
	}

	// Error propagation: δ = Wᵗ · dL/dY
	gWT := blas64.General{Rows: inLen, Cols: outLen, Stride: outLen, Data: dc.Weights}
	gPrev := blas64.General{Rows: inLen, Cols: 1, Stride: 1, Data: dc.DGrad} // reuse same buffer
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, gWT, gDL, 0.0, gPrev)

	// Store propagated error back into err[l-1], applying activation derivative
	idx = 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			v := prev.Neurons[y][x].Value
			err[l-1][y][x] += dc.DGrad[idx] * activationDerivative(v, prev.Neurons[y][x].Activation)
			idx++
		}
	}
}

func (n *Network) forwardDispatch(l int) {
	if n.UseBLAS && n.Layers[l].Dense != nil {
		n.BLASForwardLayer(l)
	} else {
		n.forwardLayer(l)
	}
}

func (n *Network) backwardDispatch(l int, err [][][]float64, lr float64) {
	if n.UseBLAS && n.Layers[l].Dense != nil {
		n.BLASBackwardLayer(l, err, lr)
	} else {
		n.backwardLayer(l, err, lr)
	}
}

func (n *Network) BakeDenseMatrices(fullyConnected []bool) {
	for l := 1; l < len(n.Layers); l++ {
		if !fullyConnected[l] {
			continue
		}

		prev := n.Layers[l-1]
		curr := &n.Layers[l]

		in := prev.Width * prev.Height
		out := curr.Width * curr.Height

		dc := &DenseCache{
			Weights: make([]float64, out*in),
			Biases:  make([]float64, out),
			Out:     make([]float64, out),
			DGrad:   make([]float64, in),
		}

		row := 0
		for y := 0; y < curr.Height; y++ {
			for x := 0; x < curr.Width; x++ {
				neuron := curr.Neurons[y][x]
				for _, c := range neuron.Inputs {
					idx := row*in + (c.SourceY*prev.Width + c.SourceX)
					dc.Weights[idx] = c.Weight
				}
				dc.Biases[row] = neuron.Bias
				row++
			}
		}

		curr.Dense = dc
	}
}
