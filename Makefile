.PHONY: build test clean

build:
	go build -o bin/llm-router .

test:
	go test ./...

clean:
	rm -rf bin/
