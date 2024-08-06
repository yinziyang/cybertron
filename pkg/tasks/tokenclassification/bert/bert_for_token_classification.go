// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"strings"

	"github.com/yinziyang/cybertron/pkg/models/bert"
	"github.com/yinziyang/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/mat"
)

type ModelForTokenClassification struct {
	*bert.ModelForTokenClassification
}

// Classify returns the logits for each token.
func (m *ModelForTokenClassification) Classify(tokens []string) []mat.Tensor {
	return m.Classifier.Forward(m.EncodeAndReduce(tokens)...)
}

func (m *ModelForTokenClassification) EncodeAndReduce(tokens []string) []mat.Tensor {
	encoded := m.Bert.EncodeTokens(tokens)

	result := make([]mat.Tensor, 0, len(tokens))
	for i, token := range tokens {
		if isSpecialToken(token) {
			encoded[i].Value() // important
			continue
		}
		result = append(result, encoded[i])
	}
	return result
}

func isSpecialToken(token string) bool {
	return strings.HasPrefix(token, wordpiecetokenizer.DefaultSplitPrefix) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultClassToken) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultSequenceSeparator) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultMaskToken)
}
