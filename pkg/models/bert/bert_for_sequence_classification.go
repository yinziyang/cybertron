// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"log"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &ModelForSequenceClassification{}

// ModelForSequenceClassification implements a Bert model for sequence classification.
type ModelForSequenceClassification struct {
	nn.Module
	// Bart is the fine-tuned BERT model.
	Bert *Model
	// Classifier is the linear layer for sequence classification.
	Classifier *linear.Model
}

func init() {
	gob.Register(&ModelForSequenceClassification{})
}

// NewModelForSequenceClassification returns a new model for sequence classification.
func NewModelForSequenceClassification[T float.DType](bert *Model) *ModelForSequenceClassification {
	return &ModelForSequenceClassification{
		Bert:       bert,
		Classifier: linear.New[T](bert.Config.HiddenSize, len(bert.Config.ID2Label)),
	}
}

// Classify returns the logits for the sequence classification.
func (m *ModelForSequenceClassification) Classify(tokens []string) mat.Tensor {
	log.Println("encode tokens start")
	encodeTokens := m.Bert.EncodeTokens(tokens)[0]
	log.Println("encode tokens end")

	log.Println("pooler start")
	pooler := m.Bert.Pooler.Forward(encodeTokens)
	log.Println("pooler end")

	log.Println("forward start")
	forward := m.Classifier.Forward(pooler)
	log.Println("forward end")

	return forward[0]
	// return m.Classifier.Forward(m.Bert.Pooler.Forward(m.Bert.EncodeTokens(tokens)[0]))[0]
}
