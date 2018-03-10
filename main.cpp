#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <vector>
#include <variant>
#include <memory>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>

constexpr int MAX_NODES = 15;

std::default_random_engine random_engine(std::chrono::system_clock::now().time_since_epoch().count());

struct vec2 {
    vec2() = default;
    vec2(double x, double y)
        : x(x)
        , y(y)
    {}
    double x, y;
};

struct fitness_history_entry {
    double min, avg, max;
};

enum class eqfunction {
    addition,
    subtraction,
    multiplication,
    power,
    count,
};

enum class eqvalue {
    x, constant,
    count,
};

size_t const function_operand_count[] = {
    2,
    2,
    2,
    2,
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

class CurveEquation {
public:
    CurveEquation(eqvalue value) noexcept
        : operation_(value)
        , nodes_(1)
    {}

    CurveEquation(double value) noexcept
        : operation_(value)
        , nodes_(1)
    {}

    CurveEquation(eqfunction func, std::vector<std::unique_ptr<CurveEquation>> &&operands)
        : operation_(func)
        , operands_(std::move(operands))
        , nodes_(1)
    {
        for (auto const &op : operands_) {
            nodes_ += op->nodes();
        }
    }

    CurveEquation(CurveEquation const &other)
        : operation_(other.operation_)
        , nodes_(other.nodes_)
    {
        for (auto const &op : other.operands_) {
            operands_.emplace_back(op->copy());
        }
    }

    std::unique_ptr<CurveEquation> copy() const {
        return std::make_unique<CurveEquation>(*this);
    }

    int nodes() const {
        return nodes_;
    }

    eqfunction function() const {
        return std::get<eqfunction>(operation_);
    }

    double value(double x) const {
        switch (std::get<eqvalue>(operation_)) {
            case eqvalue::x: return x;
            default:
                throw std::out_of_range("invalid eqvalue");
        }
    }

    double constant() const {
        return std::get<double>(operation_);
    }

    bool is_value() const {
        return std::holds_alternative<eqvalue>(operation_);
    }

    bool is_constant() const {
        return std::holds_alternative<double>(operation_);
    }

    double evaluate(double x) const {
        if (is_value())
            return value(x);
        if (is_constant())
            return constant();
        switch (function()) {
            case eqfunction::addition:
                assert(operands_.size() == 2);
                return operands_[0]->evaluate(x) + operands_[1]->evaluate(x);
                break;
            case eqfunction::subtraction:
                assert(operands_.size() == 2);
                return operands_[0]->evaluate(x) - operands_[1]->evaluate(x);
                break;
            case eqfunction::multiplication:
                assert(operands_.size() == 2);
                return operands_[0]->evaluate(x) * operands_[1]->evaluate(x);
                break;
            case eqfunction::power:
                assert(operands_.size() == 2);
                return std::pow(operands_[0]->evaluate(x), operands_[1]->evaluate(x));
                break;
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    void print(std::ostream &os) const{
        if (is_value()) {
            switch (std::get<eqvalue>(operation_)) {
                case eqvalue::x: os << 'x'; break;
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
                assert(operands_.size() == 2);
                os << '(';
                operands_[0]->print(os);
                os << ")+(";
                operands_[1]->print(os);
                os << ')';
                break;
            case eqfunction::subtraction:
                assert(operands_.size() == 2);
                os << '(';
                operands_[0]->print(os);
                os << ")-(";
                operands_[1]->print(os);
                os << ')';
                break;
            case eqfunction::multiplication:
                assert(operands_.size() == 2);
                os << '(';
                operands_[0]->print(os);
                os << ")*(";
                operands_[1]->print(os);
                os << ')';
                break;
            case eqfunction::power:
                assert(operands_.size() == 2);
                os << '(';
                operands_[0]->print(os);
                os << ")^(";
                operands_[1]->print(os);
                os << ')';
                break;
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    CurveEquation& get_nth_node(int n) {
        assert(n >= 0);
        if (n == 0)
            return *this;
        n--;
        for (auto const &op : operands_) {
            if (op->nodes() > n)
                return op->get_nth_node(n);
            n -= op->nodes();
        }
    }

    void swap(CurveEquation &other) {
        operands_.swap(other.operands_);
        std::swap(operation_, other.operation_);
        std::swap(nodes_, other.nodes_);
    }

    double get_fitness(std::vector<vec2> const &points) {
        double error = 0.0;
        for (auto const &p : points) {
            error += pow(evaluate(p.x) - p.y, 2.0);
            if (std::isnan(error))
                return 0.0;
        }
        double fitness = 1.0 / (error + 1.0) + 1.0 / nodes();
        return fitness;
    }

    void mutate();

    int recalculate_nodes_count() {
        nodes_ = 1;
        for (auto const & op :operands_)
            nodes_ += op->recalculate_nodes_count();
        return nodes_;
    }

private:
    std::variant<eqfunction, eqvalue, double> operation_;
    std::vector<std::unique_ptr<CurveEquation>> operands_;
    int nodes_;
};

std::unique_ptr<CurveEquation> rand_curve_equation(int max_nodes) {
    assert(max_nodes > 0);
    assert(max_nodes < MAX_NODES);
    assert(max_nodes & 1);

    if (max_nodes == 1) {
        auto value = rand_value();
        if (value == eqvalue::constant)
            return std::make_unique<CurveEquation>(rand_constant());
        else
            return std::make_unique<CurveEquation>(value);
    }

    max_nodes--;

    auto func = rand_func();
    std::vector<std::unique_ptr<CurveEquation>> operands;
    std::uniform_int_distribution<> random_max_nodes(1, max_nodes / 2);
    for (size_t i = 0; i < operand_count(func); i++) {
        int nodes = max_nodes;
        if (i != operand_count(func) - 1) {
            nodes = random_max_nodes(random_engine) * 2 - 1;
            max_nodes -= nodes;
        }
        operands.emplace_back(rand_curve_equation(nodes));
    }

    return std::make_unique<CurveEquation>(func, std::move(operands));
}

void CurveEquation::mutate() {
    std::uniform_int_distribution<> random_node(0, nodes() - 1);
    auto &mutation_base = get_nth_node(random_node(random_engine));
    auto max_nodes = MAX_NODES - (nodes() - mutation_base.nodes());
    std::uniform_int_distribution<> random_nodes(1, std::max(max_nodes / 2, 1));
    auto nodes = random_nodes(random_engine) * 2 - 1;
    mutation_base.swap(*rand_curve_equation(nodes));
    recalculate_nodes_count();
}

std::pair<std::unique_ptr<CurveEquation>, std::unique_ptr<CurveEquation>>
crossover_curve_equations(CurveEquation const &eq1, CurveEquation const &eq2)
{
    auto eq1copy = eq1.copy();
    auto eq2copy = eq2.copy();
    auto children = std::make_pair(std::move(eq1copy), std::move(eq2copy));

    std::uniform_int_distribution<> random;

    auto c1size = children.first->nodes();
    auto c2size = children.second->nodes();

    while (true) {
        auto &c1cross_node = children.first->get_nth_node(random(random_engine) % c1size);
        auto &c2cross_node = children.second->get_nth_node(random(random_engine) % c2size);
        if (children.first->nodes() - c1cross_node.nodes() + c2cross_node.nodes() > MAX_NODES ||
                children.second->nodes() - c2cross_node.nodes() + c1cross_node.nodes() > MAX_NODES)
        {
            continue;
        }
        c1cross_node.swap(c2cross_node);
        break;
    }

    children.first->recalculate_nodes_count();
    children.second->recalculate_nodes_count();

    return children;
}

bool in_optimum(std::vector<fitness_history_entry> const &fitness_history) {
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

    return history_length == optimum_history_length && sd < 10.0 && avg_mean > 1e6;
}

using equation_w_fitness = std::pair<std::unique_ptr<CurveEquation>, double>;
using equation_list = std::vector<equation_w_fitness>;

int main() {
    const size_t population_size = 1000;

    std::vector<vec2> points {
        vec2{ -4, 2, },
        vec2{ -2, 0, },
        vec2{ 0, 6 },
    };

    std::vector<fitness_history_entry> fitness_history;
    equation_list equations;

    // Initial population
    std::uniform_int_distribution<> random_max_nodes(1, 2);
    for (size_t i = 0; i < population_size; i++) {
        auto max_nodes = random_max_nodes(random_engine) * 2 + 1;
        auto equation = rand_curve_equation(max_nodes);
        assert(equation->nodes() == max_nodes);
        equations.emplace_back(std::make_pair(std::move(equation), equation->get_fitness(points)));
    }

    do {
        std::uniform_int_distribution<> random_equation(0, equations.size() - 1);
        // Crossover
        /* std::cout << "Crossover" << std::endl; */
        while (equations.size() < population_size * 2) {
            auto first_equation_id = random_equation(random_engine);
            auto second_equation_id = random_equation(random_engine);
            auto children = crossover_curve_equations(
                    *equations[first_equation_id].first,
                    *equations[second_equation_id].first
                    );
            auto fitness = std::make_pair(
                    children.first->get_fitness(points),
                    children.second->get_fitness(points)
                    );
            equations.emplace_back(std::make_pair(std::move(children.first), fitness.first));
            equations.emplace_back(std::make_pair(std::move(children.second), fitness.second));
        }

        // Mutation
        /* std::cout << "Mutation" << std::endl; */
        std::uniform_real_distribution<> random_mutation(0.0, 1.0);
        for (auto &eq : equations) {
            if (random_mutation(random_engine) < 0.1) {
                eq.first->mutate();
            }
        }

        // Calculate fitness stats
        /* std::cout << "Calculate fitness stats" << std::endl; */
        double fitness_sum = 0.0;
        fitness_history_entry history_entry;
        history_entry.min = std::numeric_limits<double>::max();
        history_entry.max = 0.0;
        CurveEquation *best_equation;
        for (auto &eq : equations) {
            auto fitness = eq.first->get_fitness(points);
            history_entry.min = std::min(history_entry.min, fitness);
            history_entry.max = std::max(history_entry.max, fitness);
            if (history_entry.max == fitness)
                best_equation = eq.first.get();
            fitness_sum += fitness;
            eq.second = fitness;
        }
        history_entry.avg = fitness_sum / static_cast<double>(equations.size());
        fitness_history.emplace_back(history_entry);

        // Selection
        /* std::cout << "Selection" << std::endl; */
        equation_list new_equations;

        std::sort(equations.begin(), equations.end(),
                [](equation_w_fitness const &e1, equation_w_fitness const &e2) {
                    return e1.second > e2.second;
                });

        /* for (auto const &eq : equations) { */
        /*     std::cout << eq.second << '\t'; */
        /*     eq.first->print(std::cout); */
        /*     std::cout << '\n'; */
        /* } */

        /* for (size_t i = 0; i < population_size * 0.01; i++) { */
        /*     new_equations.emplace_back(std::make_pair( */
        /*             equations[i].first->copy(), */
        /*             equations[i].second */
        /*             )); */
        /* } */

        std::uniform_real_distribution<> random_selection(0.0, fitness_sum);
        while (new_equations.size() < population_size) {
            auto selection = random_selection(random_engine);
            size_t selected_id = 0;
            for (; selection > 0.0 && selected_id < equations.size(); selected_id++) {
                selection -= equations[selected_id].second;
            }
            if (selected_id >= equations.size())
                selected_id = equations.size() - 1;
            new_equations.emplace_back(std::make_pair(
                        equations[selected_id].first->copy(),
                        equations[selected_id].second
                        ));
        }

        std::cout
            << std::setw(16) << history_entry.min << ' '
            << std::setw(16) << history_entry.avg << ' '
            << std::setw(16) << history_entry.max << '\t';
        best_equation->print(std::cout);
        std::cout << std::endl;

        /* getchar(); */

        equations.swap(new_equations);
    } while (true || !in_optimum(fitness_history));

    return 0;
}

