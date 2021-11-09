CXX=clang++
CXX_FLAGS=-std=c++20 -O3 -Iincludes/
DEBUG_FLAGS=-std=c++20 -g -O0 -Wall -Wextra -Werror -Iincludes/

MAKE_BIN := mkdir -p bin
SOURCES := $(wildcard src/*.cc)
INCLUDES := $(wildcard includes/*.hpp)

exec: bin/exec
debug: bin/debug
tests: bin/tests

bin/exec: $(SOURCES) $(INCLUDES)
	$(MAKE_BIN)
	$(CXX) $(CXX_FLAGS) driver.cc $(SOURCES) -o bin/exec

bin/debug: $(SOURCES) $(INCLUDES)
	$(MAKE_BIN)
	$(CXX) $(DEBUG_FLAGS) driver.cc $(SOURCES) -o bin/debug

bin/tests: tests/tests.cc tests/catch.hpp $(SOURCES) $(INCLUDES)
	$(MAKE_BIN)
	$(CXX) $(CXX_FLAGS) tests/tests.cc $(SOURCES) -o bin/tests

.DEFAULT_GOAL := exec
.PHONY: clean exec debug tests

clean:
	rm -rf bin/*
