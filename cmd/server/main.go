// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/joho/godotenv"
	"github.com/yinziyang/cybertron/pkg/server"
	"github.com/yinziyang/cybertron/pkg/tasks"
	"github.com/yinziyang/cybertron/pkg/tasks/languagemodeling"
	"github.com/yinziyang/cybertron/pkg/tasks/questionanswering"
	"github.com/yinziyang/cybertron/pkg/tasks/textclassification"
	"github.com/yinziyang/cybertron/pkg/tasks/textencoding"
	"github.com/yinziyang/cybertron/pkg/tasks/textgeneration"
	"github.com/yinziyang/cybertron/pkg/tasks/tokenclassification"
	"github.com/yinziyang/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

const defaultModelsDir = "models"
const addrRandomPort = ":0"

// main is the entry point of the application.
func main() {
	if err := run(); err != nil {
		log.Error().Err(err).Send()
		os.Exit(1)
	}
}

// run set the configuration and starts the server.
func run() error {
	initLogger()
	loadDotenv()

	conf := &config{
		loaderConfig: &tasks.Config{ModelsDir: defaultModelsDir},
		serverConfig: &server.Config{Address: addrRandomPort},
	}

	// load env vars values *before* parsing command line flags:
	// this gives to the flag a priority over values from the environment.
	if err := conf.loadEnv(); err != nil {
		return err
	}

	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	conf.bindFlagSet(fs)

	err := fs.Parse(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}

	m, err := loadModelForTask(conf)
	if err != nil {
		return err
	}

	logMetrics()

	requestHandler, err := server.ResolveRequestHandler(m)
	if err != nil {
		return err
	}

	s := server.New(conf.serverConfig, requestHandler)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill)
	defer stop()

	return s.Start(ctx)
}

func logMetrics() {
	// Set up zerolog to print with human-readable timestamps
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Get total CPU count
	totalCpu, _ := cpu.Counts(false)
	// Get process CPU percentage
	p, _ := process.NewProcess(int32(os.Getpid()))
	percent, _ := p.CPUPercent()

	log.Info().
		Int("total_cpus", totalCpu).
		Float64("cpu_used_by_process_percent", percent).
		Msg("CPU Metrics")

	// Get total available RAM
	vmStat, _ := mem.VirtualMemory()
	// Get process RAM usage
	memInfo, _ := p.MemoryInfo()

	log.Info().
		Uint64("total_RAM_available", vmStat.Total).
		Uint64("RAM_used_by_process", memInfo.RSS).
		Msg("RAM Metrics")
}

func loadModelForTask(conf *config) (m any, err error) {
	switch conf.task {
	case ZeroShotClassificationTask:
		return tasks.Load[zeroshotclassifier.Interface](conf.loaderConfig)
	case TextGenerationTask:
		return tasks.Load[textgeneration.Interface](conf.loaderConfig)
	case QuestionAnsweringTask:
		return tasks.Load[questionanswering.Interface](conf.loaderConfig)
	case TextClassificationTask:
		return tasks.Load[textclassification.Interface](conf.loaderConfig)
	case TokenClassificationTask:
		return tasks.Load[tokenclassification.Interface](conf.loaderConfig)
	case TextEncodingTask:
		return tasks.Load[textencoding.Interface](conf.loaderConfig)
	case LanguageModelingTask:
		return tasks.Load[languagemodeling.Interface](conf.loaderConfig)
	default:
		return nil, fmt.Errorf("failed to load model/task type %s", conf.task)
	}
}

// initLogger initializes the logger.
func initLogger() {
	log.Logger = log.Output(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		TimeFormat: time.RFC3339,
	})
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

// loadDotenv loads the .env file if it exists.
func loadDotenv() {
	_, err := os.Stat(".env")
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		log.Warn().Err(err).Msg("failed to read .env file")
		return
	}
	err = godotenv.Load()
	if err != nil {
		log.Warn().Err(err).Msg("failed to read .env file")
	}
}
