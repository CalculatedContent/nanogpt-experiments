.PHONY: test lint smoke
lint:
	ruff check .
test:
	pytest
smoke:
	./scripts/run_smoke_test.sh /tmp
