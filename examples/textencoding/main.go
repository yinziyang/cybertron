// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/yinziyang/cybertron/examples"
	"github.com/yinziyang/cybertron/pkg/models/bert"
	"github.com/yinziyang/cybertron/pkg/tasks"
	"github.com/yinziyang/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const limit = 10 // number of dimensions to show

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", textencoding.DefaultModelMulti)

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	fn := func(text string) error {
		result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
		if err != nil {
			return err
		}
		fmt.Println(result.Vector.Data().F64()[:limit])
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
