CXX = g++ #μεταγλωττιστής C++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude #σημαίες μεταγλώττισης

TARGET = bin/search #όνομα του εκτελέσιμου αρχείου
SRCS = $(wildcard src/*.cpp) #όλα τα αρχεία πηγαίου κώδικα
OBJS = $(patsubst src/%.cpp, build/%.o, $(SRCS)) #αρχεία αντικειμένου

.PHONY: all clean run #δηλώνει τις ψευδοεντολές

all: $(TARGET) bin/find_best_k #προεπιλεγμένος στόχος

#χτίσιμο του κύριου εκτελέσιμου αρχείου
$(TARGET): $(filter-out build/find_best_k.o,$(OBJS))
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^

#χτίσιμο των αρχείων αντικειμένου
build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

#χτίσιμο του εκτελέσιμου αρχείου για την εύρεση του καλύτερου k
bin/find_best_k: build/find_best_k.o $(filter-out build/find_best_k.o build/main.o,$(OBJS))
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^

#εκτέλεση του προγράμματος
run: $(TARGET)
	./$(TARGET)

#καθαρισμός των παραγόμενων αρχείων
clean:
	rm -rf build bin plots
	rm -f results*.txt
	rm -f silhouette*
	
