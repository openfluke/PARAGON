package paragon

import (
	"encoding/json"
	"fmt"
	"os"
)

/*────────────────────────────  aliases  ────────────────────────────────*/
// Paragon’s concrete layer type is Grid.  Alias it so we can keep using
// the more generic name “Layer” in this file.
type Layer = Grid

/*────────────────────────── serialisable shapes ───────────────────────*/

type sConn struct {
	L int     `json:"layer"` // source layer index
	X int     `json:"x"`     // source neuron X
	Y int     `json:"y"`     // source neuron Y
	W float64 `json:"w"`     // weight
}

type sNeuron struct {
	Bias float64 `json:"b"`
	Act  string  `json:"a"`
	In   []sConn `json:"in"`
}

type sLayer struct {
	W       int         `json:"w"` // width  (neurons per row)
	H       int         `json:"h"` // height (rows)
	Neurons [][]sNeuron `json:"n"`
}

type sNet struct {
	Layers []sLayer `json:"layers"`
}

/*────────────────── helpers: Network ↔ serialisable ───────────────────*/

// toS flattens a runtime Network into raw data ready for JSON.
func (n *Network) toS() sNet {
	s := sNet{Layers: make([]sLayer, len(n.Layers))}

	for li, L := range n.Layers {
		sl := sLayer{
			W:       L.Width,
			H:       L.Height,
			Neurons: make([][]sNeuron, L.Height),
		}

		for y := 0; y < L.Height; y++ {
			row := make([]sNeuron, L.Width)

			for x := 0; x < L.Width; x++ {
				src := L.Neurons[y][x]
				sn := sNeuron{
					Bias: src.Bias,
					Act:  src.Activation,
					In:   make([]sConn, len(src.Inputs)),
				}
				for k, c := range src.Inputs {
					sn.In[k] = sConn{L: c.SourceLayer, X: c.SourceX, Y: c.SourceY, W: c.Weight}
				}
				row[x] = sn
			}
			sl.Neurons[y] = row
		}
		s.Layers[li] = sl
	}

	return s
}

// fromS rebuilds a Network from the flattened form.
func (n *Network) fromS(s sNet) error {
	n.Layers = make([]Layer, len(s.Layers))

	for li, sl := range s.Layers {
		if sl.W == 0 || sl.H == 0 {
			return fmt.Errorf("layer %d has zero width or height", li)
		}

		L := Layer{
			Width:   sl.W,
			Height:  sl.H,
			Neurons: make([][]*Neuron, sl.H),
		}

		for y := 0; y < sl.H; y++ {
			if len(sl.Neurons[y]) != sl.W {
				return fmt.Errorf("layer %d row %d width mismatch", li, y)
			}

			row := make([]*Neuron, sl.W)

			for x := 0; x < sl.W; x++ {
				sn := sl.Neurons[y][x]
				nn := &Neuron{
					Bias:       sn.Bias,
					Activation: sn.Act,
					Inputs:     make([]Connection, len(sn.In)),
				}
				for k, c := range sn.In {
					nn.Inputs[k] = Connection{
						SourceLayer: c.L,
						SourceX:     c.X,
						SourceY:     c.Y,
						Weight:      c.W,
					}
				}
				row[x] = nn
			}
			L.Neurons[y] = row
		}
		n.Layers[li] = L
	}

	n.OutputLayer = len(n.Layers) - 1
	return nil
}

/*──────────────────────────── public API ───────────────────────────────*/

// SaveJSON writes the full topology + weights of the network.
func (n *Network) SaveJSON(path string) error {
	b, err := json.MarshalIndent(n.toS(), "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// LoadJSON restores a network that was saved with SaveJSON.
func (n *Network) LoadJSON(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var s sNet
	if err = json.Unmarshal(b, &s); err != nil {
		return err
	}
	return n.fromS(s)
}

// MarshalJSONModel returns the model as a JSON byte‑slice in the same
// loss‑less format SaveJSON uses (handy for in‑memory cloning).
func (n *Network) MarshalJSONModel() ([]byte, error) {
	return json.Marshal(n.toS())
}

// UnmarshalJSONModel overwrites *n with data produced by MarshalJSONModel.
func (n *Network) UnmarshalJSONModel(b []byte) error {
	var sn sNet
	if err := json.Unmarshal(b, &sn); err != nil {
		return err
	}
	return n.fromS(sn)
}
