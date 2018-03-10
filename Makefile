all: gacurveapprox.elf

gacurveapprox.elf: main.cpp
	$(CXX) -std=c++17 $^ -O3 -o $@

