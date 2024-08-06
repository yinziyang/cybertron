// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	questionansweringv1 "github.com/yinziyang/cybertron/pkg/server/gen/proto/go/questionanswering/v1"
	"github.com/yinziyang/cybertron/pkg/tasks/questionanswering"
	"google.golang.org/grpc"
)

// serverForQuestionAnswering is a server that provides gRPC and HTTP/2 APIs for Interface task.
type serverForQuestionAnswering struct {
	questionansweringv1.UnimplementedQuestionAnsweringServiceServer
	engine questionanswering.Interface
}

func NewServerForQuestionAnswering(engine questionanswering.Interface) RequestHandler {
	return &serverForQuestionAnswering{engine: engine}
}

func (s *serverForQuestionAnswering) RegisterServer(r grpc.ServiceRegistrar) error {
	questionansweringv1.RegisterQuestionAnsweringServiceServer(r, s)
	return nil
}

func (s *serverForQuestionAnswering) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return questionansweringv1.RegisterQuestionAnsweringServiceHandlerServer(ctx, mux, s)
}

// ExtractAnswer handles the Answer request.
func (s *serverForQuestionAnswering) ExtractAnswer(ctx context.Context, req *questionansweringv1.AnswerRequest) (*questionansweringv1.AnswerResponse, error) {
	params := req.GetOptions()
	opts := &questionanswering.Options{
		MaxAnswers:      int(params.GetMaxAnswers()),
		MaxAnswerLength: int(params.GetMaxAnswersLen()),
		MinScore:        params.GetMinScore(),
		MaxCandidates:   int(params.GetMaxCandidates()),
	}

	result, err := s.engine.ExtractAnswer(ctx, req.GetQuestion(), req.GetPassage(), opts)
	if err != nil {
		return nil, err
	}
	answers := make([]*questionansweringv1.Answer, len(result.Answers))
	for i, answer := range result.Answers {
		answers[i] = &questionansweringv1.Answer{
			Text:  answer.Text,
			Score: answer.Score,
			Start: int64(answer.Start),
			End:   int64(answer.End),
		}
	}
	resp := &questionansweringv1.AnswerResponse{
		Answers: answers,
	}
	return resp, nil
}
