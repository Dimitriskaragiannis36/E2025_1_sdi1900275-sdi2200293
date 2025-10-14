CXX      := g++
CXXFLAGS := -std=gnu++17 -O3 -Wall -Wextra -Wpedantic
LDFLAGS  := 

TARGET := search

SOURCES := \
	main.cpp \
	helper.cpp \
	mnist.cpp \
	kmeans.cpp \
	lsh.cpp \
	ivff.cpp \
	silhouette.cpp \
	hypercube.cpp

OBJECTS := $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp \
	helper.h \
	mnist.h \
	kmeans.h \
	lsh.h \
	ivff.h \
	silhouette.h \
	hypercube.h
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
