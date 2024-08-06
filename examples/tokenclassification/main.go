// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/yinziyang/cybertron/examples"
	"github.com/yinziyang/cybertron/pkg/tasks"
	"github.com/yinziyang/cybertron/pkg/tasks/tokenclassification"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", tokenclassification.DefaultEnglishModel)

	m, err := tasks.Load[tokenclassification.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	params := tokenclassification.Parameters{
		AggregationStrategy: tokenclassification.AggregationStrategySimple,
	}

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Classify(context.Background(), text, params)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(MarshalJSON(result))
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
