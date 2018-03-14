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

double PLOT_STEP = 0.001;

double WND_LEFT = 0.0;
double WND_RIGHT = 1.0;

double WND_BOTTOM = 0.0;
double WND_TOP = 1.0;

constexpr int MAX_NODES = 15;

std::default_random_engine random_engine(
        std::chrono::system_clock::now().time_since_epoch().count()
        );

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
    division,
    power,
    sine,
    count,
};

enum class eqvalue {
    x, constant,
    count,
};

std::vector<vec2> points;
std::vector<eqfunction> allowed_eqfunctions;

eqfunction rand_func() {
    /* std::uniform_int_distribution<> random(0, static_cast<int>(eqfunction::count) - 1); */
    std::uniform_int_distribution<> random(0, allowed_eqfunctions.size() - 1);
    return allowed_eqfunctions[random(random_engine)];
}

eqvalue rand_value() {
    std::uniform_int_distribution<> random(0, static_cast<int>(eqvalue::count) - 1);
    return static_cast<eqvalue>(random(random_engine));
}

double rand_constant() {
    std::normal_distribution<> random(0, 10);
    return random(random_engine);
}

double rand_power_constant() {
    std::uniform_real_distribution<> random(1, 10);
    return random(random_engine);
}

class CurveEquation;
using PCurveEquation = std::unique_ptr<CurveEquation>;

class CurveEquation {
public:
    CurveEquation(eqvalue value) noexcept
        : operation_(value)
        , left_(nullptr)
        , right_(nullptr)
        , nodes_(1)
    {}

    CurveEquation(double value) noexcept
        : operation_(value)
        , left_(nullptr)
        , right_(nullptr)
        , nodes_(1)
    {}

    CurveEquation(eqfunction func, PCurveEquation left, PCurveEquation right)
        : operation_(func)
        , left_(std::move(left))
        , right_(std::move(right))
        , nodes_(1)
    {
        assert(left_ && right_);
        nodes_ += left_->nodes();
        nodes_ += right_->nodes();
    }

    CurveEquation(CurveEquation const &other)
        : operation_(other.operation_)
        , left_(other.left_ ? other.left_->copy() : nullptr)
        , right_(other.right_ ? other.right_->copy() : nullptr)
        , nodes_(other.nodes_)
    {}

    PCurveEquation copy() const {
        return std::make_unique<CurveEquation>(*this);
    }

    void replace_with_constant(double value) {
        operation_ = value;
        left_ = nullptr;
        right_ = nullptr;
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

    bool is_variable() const { return std::holds_alternative<eqvalue>(operation_); }
    double value(double x) const {
        switch (std::get<eqvalue>(operation_)) {
            case eqvalue::x: return x;
            default:
                throw std::out_of_range("invalid eqvalue");
        }
    }

    double evaluate(double x) const {
        if (is_variable())
            return value(x);
        if (is_constant())
            return constant();
        assert(left_ && right_);
        switch (function()) {
            case eqfunction::addition:
                return left_->evaluate(x) + right_->evaluate(x);
                break;
            case eqfunction::subtraction:
                return left_->evaluate(x) - right_->evaluate(x);
                break;
            case eqfunction::multiplication:
                return left_->evaluate(x) * right_->evaluate(x);
                break;
            case eqfunction::division:
                return left_->evaluate(x) / right_->evaluate(x);
                break;
            case eqfunction::power:
                return std::pow(left_->evaluate(x), right_->evaluate(x));
                break;
            case eqfunction::sine:
                return left_->evaluate(x) * std::sin(right_->evaluate(x));
                break;
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    void print(std::ostream &os) const{
        if (is_variable()) {
            switch (std::get<eqvalue>(operation_)) {
                case eqvalue::x: os << 'x'; break;
                default:
                    throw std::out_of_range("invalid eqvalue");
            }
            return;
        }
        if (is_constant()) {
            os << constant();
            return;
        }
        assert(left_ && right_);
        switch (function()) {
            case eqfunction::addition:
                os << '(';
                left_->print(os);
                os << ")+(";
                right_->print(os);
                os << ')';
                break;
            case eqfunction::subtraction:
                os << '(';
                left_->print(os);
                os << ")-(";
                right_->print(os);
                os << ')';
                break;
            case eqfunction::multiplication:
                os << '(';
                left_->print(os);
                os << ")*(";
                right_->print(os);
                os << ')';
                break;
            case eqfunction::division:
                os << '(';
                left_->print(os);
                os << ")/(";
                right_->print(os);
                os << ')';
                break;
            case eqfunction::power:
                os << '(';
                left_->print(os);
                os << ")^(";
                right_->print(os);
                os << ')';
                break;
            case eqfunction::sine:
                os << '(';
                left_->print(os);
                os << "sin(";
                right_->print(os);
                os << "))";
                break;
            default:
            case eqfunction::count:
                throw std::out_of_range("invalid eqfunction");
        }
    }

    CurveEquation& get_nth_node(int n) {
        assert(n >= ROOT_NODE);
        if (n == ROOT_NODE)
            return *this;
        assert(left_ && right_);
        n--;
        if (left_->nodes() > n)
            return left_->get_nth_node(n);
        return right_->get_nth_node(n - left_->nodes());
    }

    std::pair<CurveEquation&, int> get_nth_node_parent(int n, int id = 0) {
        assert(n > ROOT_NODE);
        assert(left_ && right_);
        n--;
        id++;
        if (n == 0)
            return {*this, id};
        if (left_->nodes() > n)
            return left_->get_nth_node_parent(n, id);
        n -= left_->nodes();
        id += left_->nodes();
        if (n == 0)
            return {*this, id};
        return right_->get_nth_node_parent(n, id);
    }

    void swap(CurveEquation &other) {
        std::swap(operation_, other.operation_);
        std::swap(left_, other.left_);
        std::swap(right_, other.right_);
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
        fitness += std::sqrt(MAX_NODES - nodes()) * 0.01;
        return fitness;
    }

    void mutate_tree();
    void mutate_coefficients();
    void mutate_power_coefficient();

    int recalculate_nodes_count() {
        nodes_ = 1;
        if (left_) nodes_ += left_->recalculate_nodes_count();
        if (right_) nodes_ += right_->recalculate_nodes_count();
        return nodes_;
    }

    bool is_valid() const {
        if (is_function()) {
            assert(left_ && right_);
            if (function() == eqfunction::power) {
                if (!right_->is_constant())
                    return false;
                return left_->is_valid();
            }
            return left_->is_valid() && right_->is_valid();
        }
        return true;
    }

    bool simplify() {
        if (simplify_without_node_recalculation()) {
            recalculate_nodes_count();
            return true;
        }
        return false;
    }

    size_t hash() const {
        if (is_function()) {
            assert(left_ && right_);
            return (left_->hash() >> 16) ^ (right_->hash() << 16) ^ std::hash<size_t>()(static_cast<size_t>(function()));
        } else if (is_constant()) {
            return std::hash<double>()(constant());
        } else if (is_variable()) {
            return 0xAAAAAAAAAAAAAAAA;
        }
        assert(false);
    }

    static int const ROOT_NODE = 0;

private:
    bool simplify_without_node_recalculation() {
        if (!is_function())
            return false;
        assert(left_ && right_);
        if (left_->is_constant() && right_->is_constant()) {
            switch (function()) {
                case eqfunction::addition:
                    replace_with_constant(left_->constant() + right_->constant());
                    return true;
                case eqfunction::subtraction:
                    replace_with_constant(left_->constant() - right_->constant());
                    return true;
                case eqfunction::multiplication:
                    replace_with_constant(left_->constant() * right_->constant());
                    return true;
                case eqfunction::division:
                    replace_with_constant(left_->constant() / right_->constant());
                    return true;
                case eqfunction::power:
                    replace_with_constant(std::pow(left_->constant(), right_->constant()));
                    return true;
                case eqfunction::sine:
                    replace_with_constant(left_->constant() * std::sin(right_->constant()));
                    return true;
                default:
                case eqfunction::count:
                    throw std::out_of_range("invalid eqfunction");
            }
        } else if (left_->is_variable() && right_->is_variable()) {
            if (function() == eqfunction::subtraction) {
                replace_with_constant(0);
                return true;
            }
        }
        return false;
    }

    std::variant<eqfunction, eqvalue, double> operation_;
    PCurveEquation left_;
    PCurveEquation right_;
    int nodes_;
};

PCurveEquation rand_curve_equation(int max_nodes) {
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
    PCurveEquation left, right;
    std::uniform_int_distribution<> random_max_nodes(1, max_nodes / 2);
    if (func != eqfunction::power) {
        int nodes = random_max_nodes(random_engine) * 2 - 1;
        left = rand_curve_equation(nodes);
        right = rand_curve_equation(max_nodes - nodes);
    } else {
        left = rand_curve_equation(max_nodes - 1);
        right = std::make_unique<CurveEquation>(rand_power_constant());
    }

    return std::make_unique<CurveEquation>(func, std::move(left), std::move(right));
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
    /* recalculate_nodes_count(); */
    simplify();
}

void CurveEquation::mutate_coefficients() {
    if (is_constant()) {
        std::uniform_int_distribution<> random_sign(0, 1);
        auto mu = constant();
        auto sd = std::abs(constant() * 0.5);
        if (sd == 0.0)
            sd = 1;
        std::normal_distribution<> random_mutation(mu, sd);
        bool sign = random_sign(random_engine);
        constant() = random_mutation(random_engine);
        if (sign)
            constant() = -constant();
    } else if (is_function()) {
        assert(left_ && right_);
        left_->mutate_coefficients();
        if (function() != eqfunction::power)
            right_->mutate_coefficients();
        else
            right_->mutate_power_coefficient();
    }
}

void CurveEquation::mutate_power_coefficient() {
    replace_with_constant(rand_power_constant());
}

std::vector<PCurveEquation>
crossover_curve_equations(CurveEquation const &eq1, CurveEquation const &eq2)
{
    std::vector<PCurveEquation> all_descendants;
    auto children = std::make_pair(eq1.copy(), eq2.copy());

    auto tree_mutants = std::make_pair(eq1.copy(), eq2.copy());
    auto coef_mutants = std::make_pair(eq1.copy(), eq2.copy());
    tree_mutants.first->mutate_tree();
    tree_mutants.second->mutate_tree();
    coef_mutants.first->mutate_coefficients();
    coef_mutants.second->mutate_coefficients();
    all_descendants.emplace_back(std::move(tree_mutants.first));
    all_descendants.emplace_back(std::move(tree_mutants.second));
    all_descendants.emplace_back(std::move(coef_mutants.first));
    all_descendants.emplace_back(std::move(coef_mutants.second));

    std::uniform_int_distribution<> random;

    auto c1size = children.first->nodes();
    auto c2size = children.second->nodes();

    auto &c1cross_node = children.first->get_nth_node(random(random_engine) % c1size);
    auto &c2cross_node = children.second->get_nth_node(random(random_engine) % c2size);
    if (children.first->nodes() - c1cross_node.nodes() + c2cross_node.nodes() > MAX_NODES ||
            children.second->nodes() - c2cross_node.nodes() + c1cross_node.nodes() > MAX_NODES)
    {
        return all_descendants;
    }

    // try swap
    c1cross_node.swap(c2cross_node);

    children.first->recalculate_nodes_count();
    children.second->recalculate_nodes_count();

    if (!children.first->is_valid() || !children.second->is_valid())
        return all_descendants;

    children.first->mutate_tree();
    children.second->mutate_tree();

    children.first->mutate_coefficients();
    children.second->mutate_coefficients();

    children.first->simplify();
    children.second->simplify();

    all_descendants.emplace_back(std::move(children.first));
    all_descendants.emplace_back(std::move(children.second));

    return all_descendants;
}

double fitness_history_sd(std::vector<fitness_history_entry> const &fitness_history) {
    const long optimum_history_length = 1000;
    double max_mean = 0.0;
    const long history_length = std::min(
            optimum_history_length,
            static_cast<long>(fitness_history.size())
            );
    for (
        auto he = fitness_history.rbegin();
        he != fitness_history.rend()
            && he - fitness_history.rbegin() < optimum_history_length;
        ++he)
    {
        max_mean += he->max;
    }
    max_mean /= history_length;

    double sd = 0.0;
    for (
        auto he = fitness_history.rbegin();
        he != fitness_history.rend()
            && he - fitness_history.rbegin() < optimum_history_length;
        ++he)
    {
        sd += pow(he->max - max_mean, 2.0);
    }
    sd = sqrt(sd / history_length);
    return sd;
}

using equation_w_fitness = std::pair<PCurveEquation, double>;
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

    double wnd_unit = (WND_RIGHT - WND_LEFT) * 0.5;
    WND_LEFT -= wnd_unit;
    WND_RIGHT += wnd_unit;
    WND_TOP += wnd_unit;
    WND_BOTTOM -= wnd_unit;
}

int main(int argc, char *argv[]) {
    std::string allowed_eqf;
    std::cin >> allowed_eqf;
    for (auto f : allowed_eqf) {
        switch (f) {
            case '+':
                allowed_eqfunctions.emplace_back(eqfunction::addition);
                break;
            case '-':
                allowed_eqfunctions.emplace_back(eqfunction::subtraction);
                break;
            case '*':
                allowed_eqfunctions.emplace_back(eqfunction::multiplication);
                break;
            case '/':
                allowed_eqfunctions.emplace_back(eqfunction::division);
                break;
            case 'p':
                allowed_eqfunctions.emplace_back(eqfunction::power);
                break;
            case 's':
                allowed_eqfunctions.emplace_back(eqfunction::sine);
                break;
        }
    }

    size_t npoints;
    std::cin >> npoints;
    points.resize(npoints);
    for (auto &p : points) {
        std::cin >> p.x >> p.y;
    }

    init_opengl(argc, argv);

    double const elite_part = 0.1;
    double const similar_part = 0.167;
    double const stop_fitness_sd = 1e-9;

    size_t const population_size = 200;
    size_t const max_population_size = population_size * 12;

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

    double fitness_sd;
    size_t similar_equations;
    size_t generation = 0;
    do {
        /* // Mutation */
        /* /1* std::cout << "Mutation" << std::endl; *1/ */
        /* std::uniform_real_distribution<> random_mutation(0.0, 1.0); */
        /* for (auto &eq : equations) { */
        /*     if (random_mutation(random_engine) < 0.1) { */
        /*         eq.first->mutate_tree(); */
        /*         eq.second = eq.first->get_fitness(points); */
        /*     } */
        /* } */

        std::uniform_int_distribution<> random_equation(0, equations.size() - 1);
        // Crossover
        /* std::cout << "Crossover" << std::endl; */
        while (equations.size() < max_population_size) {
            auto first_equation_id = random_equation(random_engine);
            auto second_equation_id = random_equation(random_engine);
            auto children = crossover_curve_equations(
                    *equations[first_equation_id].first,
                    *equations[second_equation_id].first
                    );
            for (auto &&child : children) {
                equations.emplace_back(std::make_pair(std::move(child), child->get_fitness(points)));
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

        // Elitism
        std::sort(equations.begin(), equations.end(),
                [](equation_w_fitness const &e1, equation_w_fitness const &e2) {
                    return e1.second > e2.second;
                });

        for (size_t i = 0; i < population_size * elite_part; i++) {
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

        // Count similar equations
        size_t best_equation_hash = best_equation->hash();
        similar_equations = 0;
        for (auto const &eq : equations) {
            if (eq.first->hash() == best_equation_hash)
                similar_equations++;
        }

        fitness_sd = fitness_history_sd(fitness_history);

        std::cout
            << std::setw(8)  << generation << ' '
            << std::setw(8)  << similar_equations << ' '
            << std::setw(12) << fitness_sd << ' '
            << std::setw(12) << history_entry.min << ' '
            << std::setw(12) << history_entry.avg << ' '
            << std::setw(12) << history_entry.max << std::endl;
        best_equation->print(std::cerr);
        std::cerr << std::endl;

        glutPostRedisplay();
        glutMainLoopEvent();

        equations.swap(new_equations);

        generation++;
    } while (similar_equations < max_population_size * similar_part || fitness_sd > stop_fitness_sd); 

    glutMainLoop();

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

    if (best_equation) {
        glColor3f(0.0f, 0.0f, 1.0f);
        glLineWidth(2.0);
        glBegin(GL_LINE_STRIP);
        double step = (WND_RIGHT - WND_LEFT) * PLOT_STEP;
        for (double x = WND_LEFT; x <= WND_RIGHT; x += step) {
            auto y = best_equation->evaluate(x);
            glVertex2d(x, y);
            if (std::isnan(y)) {
                /* std::cerr << "NAN DETECTED!!! X " << x << " Y " << y << '\n'; */
                /* best_equation->print(std::cerr); */
                /* std::cerr << std::endl; */
                /* assert(!best_equation->is_valid()); */
            }
        }
        glEnd();
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_POINTS);
        for (auto const &p : points) {
            glVertex2d(p.x, best_equation->evaluate(p.x));
        }
        glEnd();
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_DOUBLE, sizeof(vec2), points.data());
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(10.0);
    glDrawArrays(GL_POINTS, 0, points.size());
    glDisableClientState(GL_VERTEX_ARRAY);

    glFlush();
}

