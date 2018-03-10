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

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h> //glut.h extension for fonts

double PLOT_STEP = 0.01;

double WND_LEFT = 0.0;
double WND_RIGHT = 1.0;

double WND_BOTTOM = 0.0;
double WND_TOP = 1.0;

constexpr int MAX_NODES = 31;

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

std::vector<vec2> points {
    vec2{ 1, 4, },
    vec2{ 2, 1 },
    vec2{ 3, 10 },
    vec2{ 4, 5 },
    vec2{ 5, 7 },
    vec2{ 6, 3 },
    vec2{ 7, 4 },
    vec2{ 8, 1 },
    vec2{ 9, 2 },
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
    std::normal_distribution<> random(0, 10);
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

    void replace_with_constant(double value) {
        operation_ = value;
        operands_.clear();
        nodes_ = 1;
    }

    int nodes() const {
        return nodes_;
    }

    bool is_function() const { return std::holds_alternative<eqfunction>(operation_); }
    eqfunction& function() { return std::get<eqfunction>(operation_); }
    eqfunction function() const { return std::get<eqfunction>(operation_); }

    bool is_constant() const { return std::holds_alternative<double>(operation_); }
    double& constant() { return std::get<double>(operation_); }
    double constant() const { return std::get<double>(operation_); }

    bool is_value() const { return std::holds_alternative<eqvalue>(operation_); }
    double value(double x) const {
        switch (std::get<eqvalue>(operation_)) {
            case eqvalue::x: return x;
            default:
                std::cout << "EQVALUE = " << (int)std::get<eqvalue>(operation_) << std::endl;
                throw std::out_of_range("invalid eqvalue");
        }
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
                default:
                    std::cout << "EQVALUE = " << (int)std::get<eqvalue>(operation_) << std::endl;
                    throw std::out_of_range("invalid eqvalue");
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

    std::pair<CurveEquation&, int> get_nth_node_parent(int n, int id = 0) {
        if (n <= 0) {
            std::cout << n << std::endl;
        }
        assert(n > 0);
        if (n == 1)
            return {*this, id};
        n--;
        id++;
        for (auto const &op : operands_) {
            if (n == 0)
                return {*this, id};
            if (op->nodes() > n)
                return op->get_nth_node_parent(n, id);
            n -= op->nodes();
            id += op->nodes();
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
            double y = evaluate(p.x);
            if (std::isnan(y))
                return 0.0;
            error += pow(y - p.y, 2.0);
        }
        double fitness = 1.0 / (error + 1.0);
        /* fitness += std::sqrt(MAX_NODES - nodes()) * 0.1; */
        return fitness;
    }

    bool simplify() {
        if (!is_function())
            return false;
        assert(operands_.size() == 2);
        if (operands_[0]->is_constant() && operands_[1]->is_constant()) {
            switch (function()) {
                case eqfunction::addition:
                    replace_with_constant(operands_[0]->constant() + operands_[1]->constant());
                    return true;
                case eqfunction::subtraction:
                    replace_with_constant(operands_[0]->constant() - operands_[1]->constant());
                    return true;
                case eqfunction::multiplication:
                    replace_with_constant(operands_[0]->constant() * operands_[1]->constant());
                    return true;
                case eqfunction::power:
                    replace_with_constant(std::pow(operands_[0]->constant(), operands_[1]->constant()));
                    return true;
                default:
                case eqfunction::count:
                    throw std::out_of_range("invalid eqfunction");
            }
        }
    }

    void mutate_tree();
    void mutate_coefficients();

    int recalculate_nodes_count() {
        nodes_ = 1;
        for (auto const & op :operands_)
            nodes_ += op->recalculate_nodes_count();
        return nodes_;
    }

    bool is_valid() const {
        if (is_function()) {
            if (function() == eqfunction::power) {
                assert(operands_.size() == 2);
                if (operands_[1]->nodes() != 1)
                    return false;
                return operands_[0]->is_valid();
            }
        }
        for (auto const &op : operands_) {
            if (!op->is_valid())
                return false;
        }
        return true;
    }

    static int const ROOT_NODE = 0;

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
    if (func != eqfunction::power) {
        for (size_t i = 0; i < operand_count(func); i++) {
            int nodes = max_nodes;
            if (i != operand_count(func) - 1) {
                nodes = random_max_nodes(random_engine) * 2 - 1;
                max_nodes -= nodes;
            }
            operands.emplace_back(rand_curve_equation(nodes));
        }
    } else {
        operands.emplace_back(rand_curve_equation(max_nodes - 1));
        operands.emplace_back(std::make_unique<CurveEquation>(rand_constant()));
    }

    return std::make_unique<CurveEquation>(func, std::move(operands));
}

bool is_first_child_node(int parent, int child) {
    return parent + 1 == child;
}

void CurveEquation::mutate_tree() {
    std::uniform_int_distribution<> random_node(0, nodes() - 1);
    auto base_node = random_node(random_engine);
    auto &mutation_base = get_nth_node(base_node);
    if (base_node != ROOT_NODE) {
        auto base_parent = get_nth_node_parent(base_node);
        if (base_parent.first.is_function()) {
            if (!is_first_child_node(base_parent.second, base_node)) {
                mutation_base.mutate_coefficients();
                return;
            }
        }
    }
    auto max_nodes = MAX_NODES - (nodes() - mutation_base.nodes());
    std::uniform_int_distribution<> random_nodes(1, std::max(max_nodes / 2, 1));
    auto nodes = random_nodes(random_engine) * 2 - 1;
    mutation_base.swap(*rand_curve_equation(nodes));
    recalculate_nodes_count();
    simplify();
}

void CurveEquation::mutate_coefficients() {
    if (is_constant()) {
        std::uniform_int_distribution<> random_sign(0, 1);
        std::normal_distribution<> random_mutation(constant(), constant() * 0.1);
        bool sign = random_sign(random_engine);
        constant() = random_mutation(random_engine);
        if (sign)
            constant() = -constant();
    } else {
        for (auto const &op : operands_) {
            op->mutate_coefficients();
        }
    }
}

std::pair<std::unique_ptr<CurveEquation>, std::unique_ptr<CurveEquation>>
crossover_curve_equations(CurveEquation const &eq1, CurveEquation const &eq2)
{
    auto children = std::make_pair(eq1.copy(), eq2.copy());

    std::uniform_int_distribution<> random;

    auto c1size = children.first->nodes();
    auto c2size = children.second->nodes();

    auto &c1cross_node = children.first->get_nth_node(random(random_engine) % c1size);
    auto &c2cross_node = children.second->get_nth_node(random(random_engine) % c2size);
    if (children.first->nodes() - c1cross_node.nodes() + c2cross_node.nodes() > MAX_NODES ||
            children.second->nodes() - c2cross_node.nodes() + c1cross_node.nodes() > MAX_NODES)
    {
        return {nullptr, nullptr};
    }

    // try swap
    c1cross_node.swap(c2cross_node);

    children.first->recalculate_nodes_count();
    children.second->recalculate_nodes_count();

    if (!children.first->is_valid() || !children.second->is_valid())
        return {nullptr, nullptr};

    children.first->mutate_tree();
    children.second->mutate_tree();

    children.first->mutate_coefficients();
    children.second->mutate_coefficients();

    children.first->simplify();
    children.second->simplify();

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

CurveEquation *best_equation = nullptr;

void display();
void reshape(int width, int height);

void init_opengl(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    glutInitWindowSize(640, 480);
    glutCreateWindow("Curve");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    for (auto const &p : points) {
        WND_LEFT = std::min(WND_LEFT, p.x);
        WND_RIGHT = std::max(WND_RIGHT, p.x);
        WND_TOP = std::max(WND_TOP, p.y);
        WND_BOTTOM = std::min(WND_BOTTOM, p.y);
    }

    double wnd_unit = (WND_RIGHT - WND_LEFT) * 0.05;
    WND_LEFT -= wnd_unit;
    WND_RIGHT += wnd_unit;
    WND_TOP += wnd_unit;
    WND_BOTTOM -= wnd_unit;
}

int main(int argc, char *argv[]) {
    init_opengl(argc, argv);

    const size_t population_size = 200;

    std::vector<fitness_history_entry> fitness_history;
    equation_list equations;

    // Initial population
    std::uniform_int_distribution<> random_max_nodes(1, 2);
    for (size_t i = 0; i < population_size; i++) {
        auto max_nodes = random_max_nodes(random_engine) * 2 + 1;
        auto equation = rand_curve_equation(max_nodes);
        assert(equation->nodes() == max_nodes);
        auto fitness = equation->get_fitness(points);
        equations.emplace_back(std::make_pair(std::move(equation), fitness));
    }

    size_t generation = 0;
    do {
        std::uniform_int_distribution<> random_equation(0, equations.size() - 1);
        // Crossover
        /* std::cout << "Crossover" << std::endl; */
        while (equations.size() < population_size * 10) {
            auto first_equation_id = random_equation(random_engine);
            auto second_equation_id = random_equation(random_engine);
            auto children = crossover_curve_equations(
                    *equations[first_equation_id].first,
                    *equations[second_equation_id].first
                    );
            if (children.first == nullptr || children.second == nullptr)
                continue;
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
                eq.first->mutate_tree();
                eq.second = eq.first->get_fitness(points);
            }
            if (random_mutation(random_engine) < 0.25) {
                eq.first->mutate_coefficients();
                eq.second = eq.first->get_fitness(points);
            }
        }

        // Calculate fitness stats
        /* std::cout << "Calculate fitness stats" << std::endl; */
        double fitness_sum = 0.0;
        fitness_history_entry history_entry;
        history_entry.min = std::numeric_limits<double>::max();
        history_entry.max = 0.0;
        for (auto &eq : equations) {
            auto fitness = eq.second;
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

        for (size_t i = 0; i < population_size * 0.05; i++) {
            new_equations.emplace_back(std::make_pair(
                    equations[i].first->copy(),
                    equations[i].second
                    ));
        }

        while (new_equations.size() < population_size) {
            auto eq1 = random_equation(random_engine);
            auto eq2 = random_equation(random_engine);
            if (equations[eq2].second > equations[eq1].second)
                eq1 = eq2;
            new_equations.emplace_back(std::make_pair(
                        equations[eq1].first->copy(),
                        equations[eq1].second
                        ));
        }

        if (generation % 100 == 0) {
            std::cout
                << std::setw(8)  << generation << ' '
                << std::setw(12) << history_entry.min << ' '
                << std::setw(12) << history_entry.avg << ' '
                << std::setw(12) << history_entry.max << '\t';
            best_equation->print(std::cout);
            std::cout << std::endl;
        }

        glutPostRedisplay();
        glutMainLoopEvent();

        equations.swap(new_equations);

        generation++;
    } while (true || !in_optimum(fitness_history));

    return 0;
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(WND_LEFT, WND_RIGHT, WND_BOTTOM, WND_TOP);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void display() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_DOUBLE, sizeof(vec2), points.data());
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(10.0);
    glDrawArrays(GL_POINTS, 0, points.size());
    glDisableClientState(GL_VERTEX_ARRAY);

    if (best_equation) {
        glColor3f(0.0f, 0.0f, 1.0f);
        glLineWidth(2.0);
        glBegin(GL_LINE_STRIP);
        double x = points.front().x;
        double step = (WND_RIGHT - WND_LEFT) * PLOT_STEP;
        for (; x < points.back().x + step / 2; x += step) {
            auto y = best_equation->evaluate(x);
            glVertex2d(x, y);
            if (std::isnan(y)) {
                std::cout << "NAN DETECTED!!! X " << x << " Y " << y << '\n';
                best_equation->print(std::cout);
                std::cout << std::endl;
                /* assert(false); */
            }
            /* std::cout << "plotting: " << x << '\t' << best_equation->evaluate(x) << '\n'; */
        }
        glEnd();
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_POINTS);
        for (auto const &p : points) {
            glVertex2d(p.x, best_equation->evaluate(p.x));
            /* std::cout << "point: " << p.x << '\t' << best_equation->evaluate(p.x) << '\n'; */
        }
        glEnd();
    }

    glFlush();
}

