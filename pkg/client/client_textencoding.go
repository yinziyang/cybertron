// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"fmt"
	"time"

	textencodingv1 "github.com/yinziyang/cybertron/pkg/server/gen/proto/go/textencoding/v1"
	"github.com/yinziyang/cybertron/pkg/tasks/textencoding"
	"github.com/nlpodyssey/spago/mat"
)

var _ textencoding.Interface = &clientForTextEncoding{}

// clientForTextEncoding is a client for text classification implementing textencoding.Interface
type clientForTextEncoding struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForTextEncoding creates a new client for text classification.
func NewClientForTextEncoding(target string, opts Options) textencoding.Interface {
	return &clientForTextEncoding{
		target: target,
		opts:   opts,
	}
}

// Encode returns the encoded representation of the given text.
func (c *clientForTextEncoding) Encode(ctx context.Context, text string, poolingStrategy int) (textencoding.Response, error) {
	conn, err := Dial(ctx, c.target, c.opts)
	if err != nil {
		return textencoding.Response{}, fmt.Errorf("failed to dial %q: %w", c.target, err)
	}
	cc := textencodingv1.NewTextEncodingServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Encode(ctx, &textencodingv1.EncodingRequest{
		Input:           text,
		PoolingStrategy: int32(poolingStrategy),
	})
	if err != nil {
		return textencoding.Response{}, err
	}
	return textencoding.Response{
		Vector: mat.NewDense[float32](mat.WithBacking(response.Vector)),
	}, nil
}
