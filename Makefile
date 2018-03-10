all: gacurveapprox.elf plot_stats.elf

gacurveapprox.elf: main.cpp
	$(CXX) $^ -std=c++17 -lGL -lGLU -lglut -O3 -o $@

plot_stats.elf: plot_stats.cpp
	$(CXX) $^ -std=c++17 -lGL -lGLU -lglut -O3 -o $@

clean:
	rm -f *.elf

