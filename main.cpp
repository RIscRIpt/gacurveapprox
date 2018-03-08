#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <variant>
#include <memory>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>

std::default_random_engine random_engine(std::chrono::system_clock::now().time_since_epoch().count());

struct vec2 {
    vec2() = default;
    vec2(double x, double y)
        : x(x)
        , y(y)
    {}
    double x, y;
};

struct vec3 : vec2 {
    vec3() = default;
    vec3(double x, double y, double z)
        : vec2(x, y)
        , z(z)
    {}
    double z;
};

struct fitness_history_entry {
    double min, avg, max;
};

enum class eqfunction {
    addition,
    subtraction,
    multiplication,
    /* division, */
    power,
    sine,
    //cosine,
    //tangent,
    count,
};

enum class eqvalue {
    x, y, constant,
    count,
};

size_t const function_operand_count[] = {
    2,
    2,
    2,
    /* 2, */
    2,
    1,
    1,
    1,
};

size_t operand_count(eqfunction f) {
    int fn = static_cast<int>(f);
    if (fn >= 0 && fn < static_cast<int>(eqfunction::count))
        return function_operand_count[fn];
    throw std::out_of_range("invalid eqfunction");
}

eqfunction rand_func() {
    std::uniform_int_distribution<> random(0, static_cast<int>(eqfunction::count) - 1);
    return static_cast<eqfunction>(random(random_engine));
}

eqvalue rand_value() {
    std::uniform_int_distribution<> random(0, static_cast<int>(eqvalue::count) - 1);
    return static_cast<eqvalue>(random(random_engine));
}

double rand_constant() {
    std::normal_distribution<> random(0, 1000);
    return random(random_engine);
}

class SurfaceEquation {
public:
    SurfaceEquation(eqvalue value) noexcept
        : operation(value)
    {}

    SurfaceEquation(double value) noexcept
        : operation(value)
    {}

    SurfaceEquation(eqfunction func, std::vector<std::unique_ptr<SurfaceEquation>> &&operands)
        : operation(func)
        , operands(std::move(operands))
    {}

    SurfaceEquation(SurfaceEquation const &other)
    {
        //std::cout << "P" << std::flush;
        //other.print(std::cout);
        //std::cout << std::flush;
        operation = other.operation;
        for (auto const &op : other.operands) {
            operands.emplace_back(op->copy());
        }
    }

    std::unique_ptr<SurfaceEquation> copy() const {
        return std::unique_ptr<SurfaceEquation>(new SurfaceEquation(*this));
    }

    size_t size() const {
        size_t total_size = 1;
        for (auto const &op : operands) {
            total_size += op->size();
        }
        return total_size;
    }

    size_t depth() const {
        size_t depth = 1;
        for (auto const &op : operands) {
            depth = std::max(depth, op->depth());
        }
        return depth;
    }

    eqfunction function() const {
        return std::get<eqfunction>(operation);
    }

    double value(vec2 const &v) const {
        switch (std::get<eqvalue>(operation)) {
            case eqvalue::x: return v.x;
            case eqvalue::y: return v.y;
            default:
                throw std::out_of_range("invalid eqvalue");
        }
    }

    double constant() const {
        return std::get<double>(operation);
    }

    bool is_value() const {
        return std::holds_alternative<eqvalue>(operation);
    }

    bool is_constant() const {
        return std::holds_alternative<double>(operation);
    }

    double evaluate(vec2 const &v) const {
        if (is_value())
            return value(v);
        if (is_constant())
            return constant();
        switch (function()) {
            case eqfunction::addition:
                assert(operands.size() == 2);
                return operands[0]->evaluate(v) + operands[1]->evaluate(v);
                break;
            case eqfunction::subtraction:
                assert(operands.size() == 2);
                return operands[0]->evaluate(v) - operands[1]->evaluate(v);
                break;
            case eqfunction::multiplication:
                assert(operands.size() == 2);
                return operands[0]->evaluate(v) * operands[1]->evaluate(v);
                break;
            /* case eqfunction::division: */
            /*     assert(operands.size() == 2); */
            /*     return operands[0]->evaluate(v) / operands[1]->evaluate(v); */
            /*     break; */
            case eqfunction::power:
                assert(operands.size() == 2);
                return std::pow(operands[0]->evaluate(v), operands[1]->evaluate(v));
                break;
            case eqfunction::sine:
                assert(operands.size() == 1);
                return std::sin(operands.front()->evaluate(v));
                break;
            /*case eqfunction::cosine:
                assert(operands.size() == 1);
                return std::cos(operands.front()->evaluate(v));
                break;
            case eqfunction::tangent:
                assert(operands.size() == 1);
                return std::tan(operands.front()->evaluate(v));
                break;*/
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    void print(std::ostream &os) const{
        if (is_value()) {
            switch (std::get<eqvalue>(operation)) {
                case eqvalue::x: os << 'x'; break;
                case eqvalue::y: os << 'y'; break;
                default: throw std::out_of_range("invalid eqvalue");
            }
            return;
        }
        if (is_constant()) {
            os << constant();
            return;
        }
        switch (function()) {
            case eqfunction::addition:
                assert(operands.size() == 2);
                os << '(';
                operands[0]->print(os);
                os << ")+(";
                operands[1]->print(os);
                os << ')';
                break;
            case eqfunction::subtraction:
                assert(operands.size() == 2);
                os << '(';
                operands[0]->print(os);
                os << ")-(";
                operands[1]->print(os);
                os << ')';
                break;
            case eqfunction::multiplication:
                assert(operands.size() == 2);
                os << '(';
                operands[0]->print(os);
                os << ")*(";
                operands[1]->print(os);
                os << ')';
                break;
            /* case eqfunction::division: */
            /*     assert(operands.size() == 2); */
            /*     os << '('; */
            /*     operands[0]->print(os); */
            /*     os << ")/("; */
            /*     operands[1]->print(os); */
            /*     os << ')'; */
            /*     break; */
            case eqfunction::power:
                assert(operands.size() == 2);
                os << '(';
                operands[0]->print(os);
                os << ")^(";
                operands[1]->print(os);
                os << ')';
                break;
            case eqfunction::sine:
                assert(operands.size() == 1);
                os << "sin(";
                operands.front()->print(os);
                os << ')';
                break;
            /*case eqfunction::cosine:
                assert(operands.size() == 1);
                os << "cos(";
                operands.front()->print(os);
                os << ')';
                break;
            case eqfunction::tangent:
                assert(operands.size() == 1);
                os << "tan(";
                operands.front()->print(os);
                os << ')';
                break;*/
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    SurfaceEquation& get_nth_node(size_t n) {
        return _get_nth_node(n);
    }

    void swap(SurfaceEquation &other) {
        std::swap(operation, other.operation);
        operands.swap(other.operands);
    }

private:
    SurfaceEquation& _get_nth_node(size_t &n) {
        if (n == 0)
            return *this;
        for (auto const &op : operands) {
            n--;
            auto &node = op->get_nth_node(n);
            if (n == 0)
                return node;
        }
        return *this;
    }

    std::variant<eqfunction, eqvalue, double> operation;
    std::vector<std::unique_ptr<SurfaceEquation>> operands;
};

std::unique_ptr<SurfaceEquation> rand_surface_equation(int max_depth) {
    if (max_depth <= 0)
        throw std::out_of_range("invalid max_depth");

    if (max_depth == 1) {
        auto value = rand_value();
        if (value == eqvalue::constant)
            return std::make_unique<SurfaceEquation>(rand_constant());
        else
            return std::make_unique<SurfaceEquation>(value);
    }

    auto func = rand_func();
    std::vector<std::unique_ptr<SurfaceEquation>> operands;
    std::uniform_int_distribution<> random_max_depth(1, max_depth - 1);
    for (size_t i = 0; i < operand_count(func); i++) {
        int depth = max_depth;
        if (i != operand_count(func) - 1) {
            depth = random_max_depth(random_engine);
            max_depth -= depth;
        }
        operands.emplace_back(rand_surface_equation(depth));
    }

    return std::make_unique<SurfaceEquation>(func, std::move(operands));
}

double get_fitness(std::vector<vec3> const &points, SurfaceEquation const &surfeq) {
    double fitness = 0.0;
    for (auto const &p : points) {
        fitness += pow(surfeq.evaluate(p) - p.z, 2.0);
    }
    fitness = 1.0 / fitness;
    if (std::isnan(fitness))
        fitness = 0.0;
    else if (fitness > 1e6)
        fitness = 1e6;
    return fitness;
}

std::pair<std::unique_ptr<SurfaceEquation>, std::unique_ptr<SurfaceEquation>>
crossover_surface_equations(SurfaceEquation const &eq1, SurfaceEquation const &eq2)
{
    //std::cout << "A" << std::flush;
    auto eq1copy = eq1.copy();
    //std::cout << "Z" << std::flush;
    auto eq2copy = eq2.copy();
    //std::cout << "X" << std::flush;
    auto children = std::make_pair(std::move(eq1copy), std::move(eq2copy));

    //std::cout << "S" << std::flush;
    std::uniform_int_distribution<> random;

    //std::cout << "D" << std::flush;
    auto c1size = children.first->size();
    auto c2size = children.second->size();

    //std::cout << "F" << std::flush;
    auto &c1cross_node = children.first->get_nth_node(random(random_engine) % c1size);
    //std::cout << "G" << std::flush;
    auto &c2cross_node = children.second->get_nth_node(random(random_engine) % c2size);

    //std::cout << "H" << std::flush;
    c1cross_node.swap(c2cross_node);

    //std::cout << "J" << std::flush;
    return children;
}

void mutate_surface_equation(SurfaceEquation &equation) {
    std::uniform_int_distribution<> random_node(0, equation.size());
    auto &mutation_base = equation.get_nth_node(random_node(random_engine));
    std::uniform_int_distribution<> random_depth(1, mutation_base.depth() * 2);
    mutation_base.swap(*rand_surface_equation(random_depth(random_engine)));
}

bool in_optimum(std::vector<fitness_history_entry> const &fitness_history) {
    //std::cout << "X" << std::endl;
    const long optimum_history_length = 20;
    double avg_mean = 0.0;
    for (
        auto he = fitness_history.rbegin();
        he != fitness_history.rend()
            && he - fitness_history.rbegin() < optimum_history_length;
        ++he)
    {
        avg_mean += he->avg;
    }
    const long history_length = std::min(optimum_history_length, static_cast<long>(fitness_history.size()));
    avg_mean /= history_length;

    //std::cout << "Y" << std::endl;
    double sd = 0.0;
    for (
        auto he = fitness_history.rbegin();
        he != fitness_history.rend()
            && std::distance(he, fitness_history.rbegin()) < optimum_history_length;
        ++he)
    {
        sd += pow(he->avg - avg_mean, 2.0);
    }
    sd = sqrt(sd / (history_length - 1));

    //std::cout << "Z" << std::endl;
    return history_length == optimum_history_length && sd < 10.0 && avg_mean > 1e6;
}

using equation_list = std::vector<std::pair<std::unique_ptr<SurfaceEquation>, double>>;

int main() {
    const size_t population_size = 500;

    std::vector<vec3> points {
        vec3{ 0.0, 0.0, 0.0 },
        vec3{ 10.0, 10.0, 0.0 },
        vec3{ 100.0, 100.0, 10.0 },
    };

    std::vector<fitness_history_entry> fitness_history;
    equation_list equations;

    // Initial population
    std::uniform_int_distribution<> random_max_depth(1, 2);
    for (size_t i = 0; i < population_size; i++) {
        auto equation = rand_surface_equation(random_max_depth(random_engine));
        equations.emplace_back(std::make_pair(std::move(equation), get_fitness(points, *equation)));
    }

    do {
        //std::cout << "A" << std::endl;
        std::uniform_int_distribution<> random_equation(0, equations.size() - 1);
        // Crossover
        while (equations.size() < population_size * 2) {
            auto first_equation_id = random_equation(random_engine);
            auto second_equation_id = random_equation(random_engine);
            auto children = crossover_surface_equations(*equations[first_equation_id].first, *equations[second_equation_id].first);
            auto fitness = std::make_pair(get_fitness(points, *children.first), get_fitness(points, *children.second));
            equations.emplace_back(std::make_pair(std::move(children.first), fitness.first));
            equations.emplace_back(std::make_pair(std::move(children.second), fitness.second));
        }

        //std::cout << "B" << std::endl;
        // Mutation
        std::uniform_real_distribution<> random_mutation(0.0, 1.0);
        for (auto &eq : equations) {
            if (random_mutation(random_engine) < 0.10) {
                mutate_surface_equation(*eq.first);
            }
        }

        //std::cout << "C" << std::endl;
        // Calculate fitness stats
        double fitness_sum = 0.0;
        fitness_history_entry history_entry;
        history_entry.min = std::numeric_limits<double>::max();
        history_entry.max = 0.0;
        for (auto &eq : equations) {
            auto fitness = get_fitness(points, *eq.first);
            history_entry.min = std::min(history_entry.min, fitness);
            history_entry.max = std::max(history_entry.max, fitness);
            fitness_sum += fitness;
            eq.second = fitness;
        }
        history_entry.avg = fitness_sum / static_cast<double>(equations.size());
        fitness_history.emplace_back(history_entry);

        //std::cout << "D" << std::endl;
        // Selection
        /*equation_list new_equations;
        std::uniform_real_distribution<> random_selection(0.0, fitness_sum);
        while (new_equations.size() < population_size) {
            auto selection = random_selection(random_engine);
            size_t selected_id = 0;
            for (; selection > 0.0 && selected_id < equations.size(); selected_id++) {
                selection -= equations[selected_id].second;
            }
            if (selected_id >= equations.size())
                selected_id = equations.size() - 1;
            auto equation = equations[selected_id].first->copy();
            auto fitness = equations[selected_id].second;
            new_equations.emplace_back(std::make_pair(std::move(equation), fitness));
        }*/

        //std::cout << "E" << std::endl;
        //equations.swap(new_equations);
        //std::cout << "F" << std::endl;

        //std::cout << fitness_history.back().min << '\t' << fitness_history.back().avg << '\t' << fitness_history.back().max << std::endl;

        //std::cout << "~" << std::endl;

        std::sort(equations.begin(), equations.end(), [&points](std::pair<std::unique_ptr<SurfaceEquation>, double> const &eq1, std::pair<std::unique_ptr<SurfaceEquation>, double> const &eq2) {
            return eq1.second > eq2.second;
        });

        std::cout << equations.front().second << "\t";
        equations.front().first->print(std::cout);
        std::cout << std::endl;

        equations.erase(equations.begin() + population_size, equations.end());

    } while (!in_optimum(fitness_history));

    getchar();

    return 0;
}

