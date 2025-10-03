CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

TARGET = lsh_test
OBJS = main.o lsh.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp lsh.h
	$(CXX) $(CXXFLAGS) -c main.cpp

lsh.o: lsh.cpp lsh.h
	$(CXX) $(CXXFLAGS) -c lsh.cpp

clean:
	rm -f $(OBJS) $(TARGET)
