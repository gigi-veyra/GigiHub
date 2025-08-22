# Makefile for GigiHub (FastAPI, lean)
# Usage examples:
#   make dev
#   make restart
#   make test
#   make test-health
#   make test-generate
#   make test-generate-code

SHELL := /bin/bash
PORT ?= 8001
HOST ?= 0.0.0.0
APP  ?= main:app

.PHONY: help dev restart stop install env test test-health test-generate test-generate-code

help:
	@echo "Targets:"
	@echo "  make dev                 - Run uvicorn with reload on $(HOST):$(PORT)"
	@echo "  make restart             - Kill anything on $(PORT) then run dev"
	@echo "  make stop                - Kill anything on $(PORT)"
	@echo "  make install             - pip install -r requirements.txt"
	@echo "  make env                 - Copy .env.example -> .env (if missing)"
	@echo "  make test                - Run all endpoint checks"
	@echo "  make test-health         - Curl /health"
	@echo "  make test-generate       - Curl /generate (text prompt)"
	@echo "  make test-generate-code  - Curl /generate_code (code-only prompt)"

dev:
	uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

restart: stop
	uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

stop:
	@lsof -ti :$(PORT) | xargs -r kill -9 || true

install:
	pip install -r requirements.txt

env:
	@[ -f .env ] || cp .env.example .env

test: test-health test-generate test-generate-code

test-health:
	@curl -s http://localhost:$(PORT)/health || true
	@echo

test-generate:
	@curl -s -X POST http://localhost:$(PORT)/generate \
	  -H "Content-Type: application/json" \
	  -d '{"text":"Write a Python function that reverses a string."}' || true
	@echo

test-generate-code:
	@curl -s -X POST http://localhost:$(PORT)/generate_code \
	  -H "Content-Type: application/json" \
	  -d '{"prompt":"Write a Python function that checks if a number is prime."}' || true
	@echo
