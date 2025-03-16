package paragon

import (
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// SerializableNeuron holds serializable neuron parameters.
type SerializableNeuron struct {
	Bias    float64   `json:"bias"`
	Weights []float64 `json:"weights"`
}

// SerializableLayer holds serializable layer parameters.
type SerializableLayer struct {
	Neurons [][]SerializableNeuron `json:"neurons"`
}

// SerializableAttentionWeights holds serializable attention weights.
type SerializableAttentionWeights struct {
	QWeights [][]float64 `json:"q_weights"`
	KWeights [][]float64 `json:"k_weights"`
	VWeights [][]float64 `json:"v_weights"`
}

// SerializableNetwork holds all serializable network parameters.
type SerializableNetwork struct {
	Layers      []SerializableLayer            `json:"layers"`
	AttnWeights []SerializableAttentionWeights `json:"attn_weights"`
	FFWeights1  [][]float64                    `json:"ff_weights1"`
	FFBias1     []float64                      `json:"ff_bias1"`
	FFWeights2  [][]float64                    `json:"ff_weights2"`
	FFBias2     []float64                      `json:"ff_bias2"`
	Config      TransformerConfig              `json:"config"`
}

// toSerializable converts the Network to a SerializableNetwork.
func (n *Network) toSerializable() SerializableNetwork {
	serial := SerializableNetwork{
		Config: n.Config,
	}

	// Layers
	serial.Layers = make([]SerializableLayer, len(n.Layers))
	for i, layer := range n.Layers {
		serialLayer := SerializableLayer{
			Neurons: make([][]SerializableNeuron, layer.Height),
		}
		for y := 0; y < layer.Height; y++ {
			serialLayer.Neurons[y] = make([]SerializableNeuron, layer.Width)
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				serialNeuron := SerializableNeuron{
					Bias:    neuron.Bias,
					Weights: make([]float64, len(neuron.Inputs)),
				}
				for k := 0; k < len(neuron.Inputs); k++ {
					serialNeuron.Weights[k] = neuron.Inputs[k].Weight
				}
				serialLayer.Neurons[y][x] = serialNeuron
			}
		}
		serial.Layers[i] = serialLayer
	}

	// Attention weights
	serial.AttnWeights = make([]SerializableAttentionWeights, len(n.AttnWeights))
	for h, attn := range n.AttnWeights {
		serial.AttnWeights[h] = SerializableAttentionWeights{
			QWeights: attn.QWeights,
			KWeights: attn.KWeights,
			VWeights: attn.VWeights,
		}
	}

	// Feed-forward parameters
	serial.FFWeights1 = n.FFWeights1
	serial.FFBias1 = n.FFBias1
	serial.FFWeights2 = n.FFWeights2
	serial.FFBias2 = n.FFBias2

	return serial
}

// fromSerializable loads parameters from a SerializableNetwork into the Network.
// Assumes the Network is initialized with the correct architecture.
func (n *Network) fromSerializable(serial SerializableNetwork) error {
	// Validate layers
	if len(serial.Layers) != len(n.Layers) {
		return fmt.Errorf("layer count mismatch: got %d, expected %d", len(serial.Layers), len(n.Layers))
	}
	for i, layer := range n.Layers {
		serialLayer := serial.Layers[i]
		if len(serialLayer.Neurons) != layer.Height {
			return fmt.Errorf("height mismatch in layer %d: got %d, expected %d", i, len(serialLayer.Neurons), layer.Height)
		}
		for y := 0; y < layer.Height; y++ {
			if len(serialLayer.Neurons[y]) != layer.Width {
				return fmt.Errorf("width mismatch in layer %d, row %d: got %d, expected %d", i, y, len(serialLayer.Neurons[y]), layer.Width)
			}
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				serialNeuron := serialLayer.Neurons[y][x]
				if len(serialNeuron.Weights) != len(neuron.Inputs) {
					return fmt.Errorf("input count mismatch in layer %d, neuron (%d,%d): got %d, expected %d", i, y, x, len(serialNeuron.Weights), len(neuron.Inputs))
				}
			}
		}
	}

	// Validate attention weights
	if len(serial.AttnWeights) != len(n.AttnWeights) {
		return fmt.Errorf("attention weights count mismatch: got %d, expected %d", len(serial.AttnWeights), len(n.AttnWeights))
	}

	// Load layers
	for i, layer := range n.Layers {
		serialLayer := serial.Layers[i]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				serialNeuron := serialLayer.Neurons[y][x]
				neuron.Bias = serialNeuron.Bias
				for k := 0; k < len(neuron.Inputs); k++ {
					neuron.Inputs[k].Weight = serialNeuron.Weights[k]
				}
			}
		}
	}

	// Load attention weights
	for h, attn := range n.AttnWeights {
		serialAttn := serial.AttnWeights[h]
		attn.QWeights = serialAttn.QWeights
		attn.KWeights = serialAttn.KWeights
		attn.VWeights = serialAttn.VWeights
	}

	// Load feed-forward parameters
	n.FFWeights1 = serial.FFWeights1
	n.FFBias1 = serial.FFBias1
	n.FFWeights2 = serial.FFWeights2
	n.FFBias2 = serial.FFBias2

	return nil
}

// SaveToJSON saves the network parameters to a JSON file.
func (n *Network) SaveToJSON(filename string) error {
	// Ensure the directory exists.
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	serial := n.toSerializable()
	data, err := json.MarshalIndent(serial, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal to JSON: %v", err)
	}
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %v", err)
	}
	return nil
}

// LoadFromJSON loads the network parameters from a JSON file.
func (n *Network) LoadFromJSON(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %v", err)
	}
	var serial SerializableNetwork
	if err := json.Unmarshal(data, &serial); err != nil {
		return fmt.Errorf("failed to unmarshal JSON: %v", err)
	}
	return n.fromSerializable(serial)
}

// SaveToGob saves the network parameters to a gob file.
func (n *Network) SaveToGob(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create gob file: %v", err)
	}
	defer file.Close()
	encoder := gob.NewEncoder(file)
	serial := n.toSerializable()
	if err := encoder.Encode(serial); err != nil {
		return fmt.Errorf("failed to encode gob: %v", err)
	}
	return nil
}

// LoadFromGob loads the network parameters from a gob file.
func (n *Network) LoadFromGob(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open gob file: %v", err)
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)
	var serial SerializableNetwork
	if err := decoder.Decode(&serial); err != nil {
		return fmt.Errorf("failed to decode gob: %v", err)
	}
	return n.fromSerializable(serial)
}

func (n *Network) SaveToBinary(filename string) error {
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create model file %s: %w", filename, err)
	}
	defer f.Close()

	// Save number of layers.
	if err := binary.Write(f, binary.BigEndian, int32(len(n.Layers))); err != nil {
		return err
	}
	for _, layer := range n.Layers {
		// Save layer dimensions.
		if err := binary.Write(f, binary.BigEndian, int32(layer.Width)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.BigEndian, int32(layer.Height)); err != nil {
			return err
		}
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				// Save bias.
				if err := binary.Write(f, binary.BigEndian, neuron.Bias); err != nil {
					return err
				}
				// Save the number of inputs.
				numInputs := int32(len(neuron.Inputs))
				if err := binary.Write(f, binary.BigEndian, numInputs); err != nil {
					return err
				}
				// For each connection, save SourceLayer, SourceX, SourceY, and Weight.
				for _, conn := range neuron.Inputs {
					if err := binary.Write(f, binary.BigEndian, int32(conn.SourceLayer)); err != nil {
						return err
					}
					if err := binary.Write(f, binary.BigEndian, int32(conn.SourceX)); err != nil {
						return err
					}
					if err := binary.Write(f, binary.BigEndian, int32(conn.SourceY)); err != nil {
						return err
					}
					if err := binary.Write(f, binary.BigEndian, conn.Weight); err != nil {
						return err
					}
				}
			}
		}
	}
	fmt.Printf("✅ Model saved to %s\n", filename)
	return nil
}

func (n *Network) LoadFromBinary(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open model file %s: %w", filename, err)
	}
	defer f.Close()

	var numLayers int32
	if err := binary.Read(f, binary.BigEndian, &numLayers); err != nil {
		return err
	}

	// Reinitialize layers based on saved dimensions.
	n.Layers = make([]Grid, 0, numLayers)
	n.OutputLayer = int(numLayers) - 1

	for l := int32(0); l < numLayers; l++ {
		var width, height int32
		if err := binary.Read(f, binary.BigEndian, &width); err != nil {
			return err
		}
		if err := binary.Read(f, binary.BigEndian, &height); err != nil {
			return err
		}
		grid := Grid{
			Width:   int(width),
			Height:  int(height),
			Neurons: make([][]*Neuron, height),
		}
		for y := 0; y < int(height); y++ {
			grid.Neurons[y] = make([]*Neuron, int(width))
			for x := 0; x < int(width); x++ {
				neuron := &Neuron{}
				// Read bias.
				if err := binary.Read(f, binary.BigEndian, &neuron.Bias); err != nil {
					return err
				}
				// Read the number of inputs.
				var numInputs int32
				if err := binary.Read(f, binary.BigEndian, &numInputs); err != nil {
					return err
				}
				neuron.Inputs = make([]Connection, numInputs)
				// For each connection, read SourceLayer, SourceX, SourceY, and Weight.
				for i := 0; i < int(numInputs); i++ {
					var srcLayer, srcX, srcY int32
					if err := binary.Read(f, binary.BigEndian, &srcLayer); err != nil {
						return err
					}
					if err := binary.Read(f, binary.BigEndian, &srcX); err != nil {
						return err
					}
					if err := binary.Read(f, binary.BigEndian, &srcY); err != nil {
						return err
					}
					var weight float64
					if err := binary.Read(f, binary.BigEndian, &weight); err != nil {
						return err
					}
					neuron.Inputs[i] = Connection{
						SourceLayer: int(srcLayer),
						SourceX:     int(srcX),
						SourceY:     int(srcY),
						Weight:      weight,
					}
				}
				grid.Neurons[y][x] = neuron
			}
		}
		n.Layers = append(n.Layers, grid)
	}
	fmt.Printf("✅ Model loaded from %s\n", filename)
	return nil
}
