CXX      := g++
CXXFLAGS := -std=gnu++17 -O3 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  := 

TARGET := search

SRC_DIR := src
SOURCES := \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/helper.cpp \
	$(SRC_DIR)/mnist.cpp \
	$(SRC_DIR)/kmeans.cpp \
	$(SRC_DIR)/lsh.cpp \
	$(SRC_DIR)/ivff.cpp \
	$(SRC_DIR)/silhouette.cpp \
	$(SRC_DIR)/hypercube.cpp

OBJECTS := $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp \
	include/helper.h \
	include/mnist.h \
	include/kmeans.h \
	include/lsh.h \
	include/ivff.h \
	include/silhouette.h \
	include/hypercube.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean distclean
clean:
	@echo "→ Καθαρισμός αντικειμένων..."
	@rm -f $(OBJECTS)

distclean: clean
	@echo "→ Καθαρισμός εκτελέσιμου..."
	@rm -f $(TARGET)

.PHONY: run
run: all
	@echo "→ Εκτέλεση: ./$(TARGET)"
	@./$(TARGET)
