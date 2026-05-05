# singlet monorepo — convenience targets
# Usage: make test | make build | make pipeline | make clean

.PHONY: test test-cpp test-python lint format typecheck coverage build pipeline clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: test-cpp test-python ## Run all tests (C++ + Python)

test-cpp: ## Build and run 88 C++ unit tests
	@cmake -B build-tests -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF 2>&1 | tail -1
	@cmake --build build-tests -j$$(nproc) 2>&1 | tail -1
	@ctest --test-dir build-tests -j$$(nproc) --output-on-failure

test-python: ## Run 515 Python tests
	@python -m pytest tests/python/ -x -q

coverage: ## Run tests with coverage report
	@python -m coverage run --source=python/singlet -m pytest tests/python/ -q --no-header
	@python -m coverage report --skip-covered --omit="*/gpu/*,*/mcp/*,*/torch/*"

lint: ## Lint Python code with ruff
	@ruff check python/ tests/python/
	@ruff format --check python/ tests/python/

typecheck: ## Type check core Python modules with pyright
	@pyright

format: ## Auto-format Python code
	@ruff check python/ tests/python/ --fix
	@ruff format python/ tests/python/

build: ## Build C++ tests only (no run)
	cmake -B build-tests -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF
	cmake --build build-tests -j$$(nproc)

pipeline: ## Build the singlify pipeline binary
	cmake -B build -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$$(nproc)

clean: ## Remove all build directories
	rm -rf build build-tests build-gpu
