// llm-router is a cost-optimizing reverse proxy for LLM APIs.
// It classifies request complexity and routes to the cheapest model that can handle it.
//
// Based on research from:
//   - FleetOpt: Analytical Fleet Provisioning with Compress-and-Route (arXiv:2603.16514)
//   - AMRO-S: Ant Colony LLM Routing (arXiv:2603.12933)
//   - TARo: Token-level Adaptive Routing (arXiv:2603.18411)
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/timholm/llm-router/router"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	cfg, err := router.LoadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	srv := router.NewServer(cfg)
	log.Printf("llm-router listening on %s (%d models configured)", *addr, len(cfg.Models))
	if err := srv.ListenAndServe(*addr); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
