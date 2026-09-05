/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 5650
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"
#include "kernel.h"
#include "benchmark.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

// ================
// Configuration
// ================

// The defines below are the defaults. Every one of them can be overridden on
// the command line so that a sweep script can walk the experiment matrix
// without rebuilding:
//
//   cis5650_boids.exe --mode=scattered --n=5000 --block=128 --bench=stages
//
//   --mode=naive|scattered|coherent   which neighbor search to run
//   --n=<int>                         number of boids
//   --block=<int>                     CUDA block size
//   --visualize=0|1                   render the boids
//   --bench=none|fps|total|stages     what to measure
//   --warmup=<int> --frames=<int>     measurement window
//
// With no arguments the program behaves exactly as the defines say.
// Results are appended to benchmark_results.csv, with the config recorded in
// every row, so a row can never be mislabelled.

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 0
#define UNIFORM_GRID 1
#define COHERENT_GRID 0

// LOOK-1.2 - change this to adjust particle count in the simulation
int N_FOR_VIS = 5000;
const float DT = 0.2f;

bool visualize = VISUALIZE;

#if UNIFORM_GRID && COHERENT_GRID
bench::SimMode simMode = bench::SimMode::Coherent;
#elif UNIFORM_GRID
bench::SimMode simMode = bench::SimMode::Scattered;
#else
bench::SimMode simMode = bench::SimMode::Naive;
#endif

static bool parseArgs(int argc, char* argv[]);

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "5650 CUDA Intro: Boids";

  if (!parseArgs(argc, argv)) {
    return 1;
  }

  if (init(argc, argv)) {
    mainLoop();
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

static bool parseArgs(int argc, char* argv[]) {
  char text[64];
  int number = 0;

  for (int i = 1; i < argc; i++) {
    if (sscanf(argv[i], "--n=%d", &number) == 1) {
      N_FOR_VIS = number;
    } else if (sscanf(argv[i], "--block=%d", &number) == 1) {
      bench::config.blockSize = number;
    } else if (sscanf(argv[i], "--visualize=%d", &number) == 1) {
      visualize = (number != 0);
    } else if (sscanf(argv[i], "--warmup=%d", &number) == 1) {
      bench::config.warmup = number;
    } else if (sscanf(argv[i], "--frames=%d", &number) == 1) {
      bench::config.frames = number;
    } else if (sscanf(argv[i], "--mode=%31s", text) == 1) {
      if (!strcmp(text, "naive")) {
        simMode = bench::SimMode::Naive;
      } else if (!strcmp(text, "scattered")) {
        simMode = bench::SimMode::Scattered;
      } else if (!strcmp(text, "coherent")) {
        simMode = bench::SimMode::Coherent;
      } else {
        std::cout << "Unknown --mode: " << text << std::endl;
        return false;
      }
    } else if (sscanf(argv[i], "--bench=%31s", text) == 1) {
      if (!strcmp(text, "none")) {
        bench::config.mode = bench::Mode::None;
      } else if (!strcmp(text, "fps")) {
        bench::config.mode = bench::Mode::Fps;
      } else if (!strcmp(text, "total")) {
        bench::config.mode = bench::Mode::Total;
      } else if (!strcmp(text, "stages")) {
        bench::config.mode = bench::Mode::Stages;
      } else {
        std::cout << "Unknown --bench: " << text << std::endl;
        return false;
      }
    } else {
      std::cout << "Unknown argument: " << argv[i] << std::endl;
      return false;
    }
  }

  bench::config.simMode = simMode;
  bench::config.numBoids = N_FOR_VIS;
  bench::config.visualize = visualize;
  return true;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);

  // INSTRUCTION.md requires V-sync off for performance testing. Doing it here
  // rather than only in the Nvidia Control Panel keeps the setting visible in
  // the repo and reproducible on any machine. Only affects visualized runs;
  // with visualization off we never call glfwSwapBuffers at all.
  glfwSwapInterval(0);

  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
  Boids::setBlockSize(bench::config.blockSize);
  Boids::initSimulation(N_FOR_VIS);

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  return true;
}

void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    // INSTRUCTION.md defines "visualization off" as measuring the simulation
    // only, so the GL interop map/unmap is skipped too -- otherwise every
    // frame still pays for a driver round trip that has nothing to do with
    // the simulation. (Deviation from the base repo; see README methodology.)
    if (visualize) {
      cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
      cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);
    }

    // execute the kernel
    bench::frameBegin();
    switch (simMode) {
      case bench::SimMode::Coherent:  Boids::stepSimulationCoherentGrid(DT); break;
      case bench::SimMode::Scattered: Boids::stepSimulationScatteredGrid(DT); break;
      case bench::SimMode::Naive:     Boids::stepSimulationNaive(DT); break;
    }
    bench::frameEnd();

    if (visualize) {
      Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
      cudaGLUnmapBufferObject(boidVBO_positions);
      cudaGLUnmapBufferObject(boidVBO_velocities);
    }
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    // Wall-clock frame time, the "application FPS" the assignment asks about.
    // Kept separate from the CUDA-event stage timers on purpose: --bench=fps
    // runs carry no instrumentation at all.
    static bench::Stage frameWall("frame_wall", false);
    double lastFrameTime = glfwGetTime();

    Boids::unitTest(); // LOOK-1.2 We run some basic example code to make sure
                       // your CUDA development setup is ready to go.

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;

        // INSTRUCTION.md asks to read the framerate off the window title
        // with visualization disabled, so the title is updated in both modes.
        // `fps` only changes once per second, so updating the title here
        // rather than every frame displays exactly the same thing without
        // paying for a Win32 call on every iteration.
        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());
      }

      runCUDA();

      if (visualize) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program[PROG_BOID]);
        glBindVertexArray(boidVAO);
        glPointSize((GLfloat)pointSize);
        glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
      }

      const double now = glfwGetTime();
      bench::submitHostSample(frameWall, (float)((now - lastFrameTime) * 1000.0));
      lastFrameTime = now;

      // Exit once the measurement window is filled so a sweep script can move
      // on to the next configuration.
      if (bench::complete()) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
    }

    bench::report();

    glfwDestroyWindow(window);
    glfwTerminate();
  }


  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }
