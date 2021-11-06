CXX=clang++
CXX_FLAGS=-std=c++20 -g -O0 -Wall -Wextra -Werror -Iincludes/

MAKE_BIN := mkdir bin
SOURCES := $(wildcard src/*.cc)
INCLUDES := $(wildcard includes/*.hpp)

exec: bin/exec
tests: bin/tests

bin/exec: $(SOURCES) $(INCLUDES)
	$(MAKE_BIN)
	$(CXX) $(CXX_FLAGS) driver.cc $(SOURCES) -o bin/exec

bin/tests: tests/tests.cc tests/catch.hpp $(SOURCES) $(INCLUDES)
	$(MAKE_BIN)
	$(CXX) $(CXX_FLAGS) tests/tests.cc $(SOURCES) -o bin/tests

.DEFAULT_GOAL := exec
.PHONY: clean exec tests

clean:
	rm -rf bin/*
