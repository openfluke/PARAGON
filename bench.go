package paragon

import (
	"fmt"
	"sync"
	"time"
)

/*
ClonePulse  — hammer a model with constant inputs

clones   : how many independent copies to spin up
aps      : Forward() calls per second, per copy
duration : wall-clock time each copy should run
inputs   : constant [][]float64 fed to every tick
*/
func (n *Network[T]) ClonePulse(
	inputs [][]float64,
	clones int,
	aps int,
	duration time.Duration,
) {
	if clones <= 0 || aps <= 0 || duration <= 0 {
		return
	}

	/*── make a single JSON snapshot so cloning is cheap ──*/
	raw, err := n.MarshalJSONModel()
	if err != nil {
		fmt.Printf("paragon: ClonePulse marshal failed: %v\n", err)
		return
	}

	interval := time.Second / time.Duration(aps)
	var wg sync.WaitGroup
	wg.Add(clones)

	for i := 0; i < clones; i++ {
		go func() {
			defer wg.Done()

			// deep copy
			var c Network[T]
			if err := c.UnmarshalJSONModel(raw); err != nil {
				fmt.Printf("paragon: clone failed: %v\n", err)
				return
			}

			tick := time.NewTicker(interval)
			defer tick.Stop()

			end := time.After(duration)
			for {
				select {
				case <-tick.C:
					c.Forward(inputs) // Forward already wants [][]float64
				case <-end:
					return
				}
			}
		}()
	}
	wg.Wait()
}
