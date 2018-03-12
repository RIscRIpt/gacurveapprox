#include <iostream>
#include <string>
#include <deque>
#include <cmath>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h> //glut.h extension for fonts

double LEFT = 0.0;
double RIGHT = 1.0;

double BOTTOM = 0.0;
double TOP = 0.0;

struct fitness_history_entry {
    double min, avg, max;
};

std::deque<fitness_history_entry> fitness_history;

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
}

void display() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(LEFT, RIGHT, BOTTOM, TOP);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    double Xwidth = RIGHT - LEFT;
    double Xstep = Xwidth / 100.0; //fitness_history.size();

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINE_STRIP);
    double x = 0;
    size_t istep = std::ceil(fitness_history.size() / 100.0);
    for (
            auto i = fitness_history.begin();
            std::distance(i, fitness_history.end()) > 0;
            i += istep)
    {
        glVertex2d(x, i->max);
        x += Xstep;
    }
    glEnd();

    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINE_STRIP);
    x = 0;
    for (
            auto i = fitness_history.begin();
            std::distance(i, fitness_history.end()) > 0;
            i += istep)
    {
        glVertex2d(x, i->avg);
        x += Xstep;
    }
    glEnd();

    auto str_avg = "Average: " + std::to_string(fitness_history.back().avg);
    auto str_max = "Maximum: " + std::to_string(fitness_history.back().max);

    auto height = TOP - BOTTOM;
    glColor3f(0.0f, 0.0f, 0.5f);
    glRasterPos2f(LEFT, TOP - height * 0.04);
    glutBitmapString(GLUT_BITMAP_9_BY_15, reinterpret_cast<unsigned char const*>(str_avg.c_str()));
    glColor3f(0.5f, 0.0f, 0.0f);
    glRasterPos2f(LEFT, TOP - height * 0.02);
    glutBitmapString(GLUT_BITMAP_9_BY_15, reinterpret_cast<unsigned char const*>(str_max.c_str()));

    glFlush();
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    glutInitWindowSize(640, 480);
    glutCreateWindow("Stats");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    while (true) {
        size_t generation;
        std::cin >> generation;

        if (std::cin.fail())
            break;

        fitness_history_entry history_entry;
        std::cin
            >> history_entry.min
            >> history_entry.avg
            >> history_entry.max;

        TOP = std::max(TOP, history_entry.max * 1.1);

        fitness_history.emplace_back(history_entry);

        /* if (fitness_history.size() > 200) */
        /*     fitness_history.pop_front(); */

        glutPostRedisplay();
        glutMainLoopEvent();
    }

    glutMainLoop();

    return 0;
}

