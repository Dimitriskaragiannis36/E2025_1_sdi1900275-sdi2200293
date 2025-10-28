CXX = g++ #Compiler
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude #Compiler flags

TARGET = bin/search #στόχος εκτέλεσης
SRCS = $(wildcard src/*.cpp) #πηγαία αρχεία
OBJS = $(patsubst src/%.cpp, build/%.o, $(SRCS)) #αντικείμενα

.PHONY: all clean run #δηλώνει τις ψευδοεντολές

all: $(TARGET) #προεπιλεγμένος στόχος

$(TARGET): $(OBJS) #σύνδεση αντικειμένων
	@mkdir -p bin 
	$(CXX) $(CXXFLAGS) -o $@ $^ 

build/%.o: src/%.cpp #μεταγλώττιση πηγαίων αρχείων
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET) #εκτέλεση προγράμματος
	./$(TARGET)

clean: #καθαρισμός παραγόμενων αρχείων
	rm -rf build bin plots 
	rm -f results*.txt
