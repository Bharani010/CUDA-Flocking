/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1

// Default values (can be overridden by command-line arguments)
int N_FOR_VIS = 5000;
int simulationMethod = 0; // 0 = Naive, 1 = Scattered Grid, 2 = Coherent Grid
const float DT = 0.2f;
bool enableVisualization = true;
bool perfTestMode = false;

// Print usage information for command-line arguments
void printUsage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  ./build/bin/cis565_boids" << std::endl;
  std::cout << "      Run with visualization using default parameters (5000 boids, Naive method)" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./build/bin/cis565_boids --perf-test [mode] [boid_count]" << std::endl;
  std::cout << "      Run performance test without visualization" << std::endl;
  std::cout << "      [mode]: Optional integer (0 = Naive, 1 = Scattered Grid, 2 = Coherent Grid)" << std::endl;
  std::cout << "              If not provided, defaults to 0 (Naive)" << std::endl;
  std::cout << "      [boid_count]: Optional integer specifying number of boids" << std::endl;
  std::cout << "                    If not provided, defaults to 5000" << std::endl;
  std::cout << std::endl;
  std::cout << "  Examples:" << std::endl;
  std::cout << "    ./build/bin/cis565_boids --perf-test         # Naive with 5000 boids" << std::endl;
  std::cout << "    ./build/bin/cis565_boids --perf-test 1       # Scattered Grid with 5000 boids" << std::endl;
  std::cout << "    ./build/bin/cis565_boids --perf-test 2 10000 # Coherent Grid with 10000 boids" << std::endl;
}

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "565 CUDA Intro: Boids";

  // Parse command-line arguments
  if (argc == 1) {
    // Default behavior - visualization mode with default parameters
    enableVisualization = true;
    perfTestMode = false;
  } else if (argc >= 2 && std::string(argv[1]) == "--perf-test") {
    // Performance test mode
    enableVisualization = false;
    perfTestMode = true;
    
    // Set default values first
    simulationMethod = 0; // Default to Naive
    N_FOR_VIS = 5000;     // Default to 5000 boids
    
    // Check for optional mode parameter
    if (argc >= 3) {
      try {
        simulationMethod = std::stoi(argv[2]);
        if (simulationMethod < 0 || simulationMethod > 2) {
          std::cerr << "Error: Invalid simulation method. Valid values are 0, 1, or 2." << std::endl;
          printUsage();
          return 1;
        }
      } catch (const std::invalid_argument&) {
        std::cerr << "Error: Invalid simulation method format" << std::endl;
        printUsage();
        return 1;
      } catch (const std::out_of_range&) {
        std::cerr << "Error: Simulation method out of range" << std::endl;
        return 1;
      }
    }
    
    // Check for optional boid count parameter
    if (argc >= 4) {
      try {
        N_FOR_VIS = std::stoi(argv[3]);
        if (N_FOR_VIS <= 0) {
          std::cerr << "Error: Boid count must be positive" << std::endl;
          printUsage();
          return 1;
        }
      } catch (const std::invalid_argument&) {
        std::cerr << "Error: Invalid boid count format" << std::endl;
        printUsage();
        return 1;
      } catch (const std::out_of_range&) {
        std::cerr << "Error: Boid count out of range" << std::endl;
        return 1;
      }
    }
  } else {
    std::cerr << "Error: Invalid arguments" << std::endl;
    printUsage();
    return 1;
  }
  
  // Display simulation configuration
  std::string methodName;
  if (simulationMethod == 0) methodName = "Naive";
  else if (simulationMethod == 1) methodName = "Scattered Grid";
  else methodName = "Coherent Grid";
  
  std::cout << "Running simulation with " << N_FOR_VIS << " boids, method: " 
            << methodName << (enableVisualization ? " (with visualization)" : " (performance test)") << std::endl;

  if (init(argc, argv)) {
    mainLoop();
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
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
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
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

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

    // execute the kernel
    if (simulationMethod == 2) {
      Boids::stepSimulationCoherentGrid(DT);
    } else if (simulationMethod == 1) {
      Boids::stepSimulationScatteredGrid(DT);
    } else {
      Boids::stepSimulationNaive(DT);
    }

    // Only copy to VBO if visualization is enabled
    if (enableVisualization) {
      Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    }
    
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
    
    // Performance metrics
    const int numFramesToMeasure = 100;
    double totalSimTime = 0.0;
    int failedLaunches = 0;

    Boids::unitTest(); // Run the unit test first
    
    if (perfTestMode) {
      // Performance test mode - no visualization
      std::cout << "Starting performance test for " << numFramesToMeasure << " frames..." << std::endl;
      
      try {
        for (int i = 0; i < numFramesToMeasure; i++) {
          double startTime = glfwGetTime();
          
          runCUDA();
          cudaDeviceSynchronize();
          
          double endTime = glfwGetTime();
          totalSimTime += (endTime - startTime);
          
          // Show progress every 10 frames
          if (i % 10 == 0) {
            std::cout << "Progress: " << i << "/" << numFramesToMeasure << " frames" << std::endl;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        failedLaunches++;
      } catch (...) {
        std::cerr << "Unknown error during simulation!" << std::endl;
        failedLaunches++;
      }
      
      // Calculate and output performance metrics
      double avgSimTime = totalSimTime / (numFramesToMeasure - failedLaunches);
      double avgFPS = (numFramesToMeasure - failedLaunches) / totalSimTime;
      
      std::string methodName;
      if (simulationMethod == 0) methodName = "Naive";
      else if (simulationMethod == 1) methodName = "Scattered Grid";
      else methodName = "Coherent Grid";
      
      std::cout << "\n=========== PERFORMANCE METRICS ===========" << std::endl;
      std::cout << "Boid Count: " << N_FOR_VIS << std::endl;
      std::cout << "Method: " << methodName << std::endl;
      std::cout << "Average FPS: " << avgFPS << std::endl;
      std::cout << "Average Simulation Time per Step (ms): " << (avgSimTime * 1000.0) << std::endl;
      std::cout << "Failed Launches: " << failedLaunches << std::endl;
      std::cout << "=========================================" << std::endl;
    } else {
      // Interactive visualization mode
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
          fps = frame / (time - timebase);
          timebase = time;
          frame = 0;
        }

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Only draw if visualization is enabled
        if (enableVisualization) {
          glUseProgram(program[PROG_BOID]);
          glBindVertexArray(boidVAO);
          glPointSize((GLfloat)pointSize);
          glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
          glPointSize(1.0f);

          glUseProgram(0);
          glBindVertexArray(0);

          glfwSwapBuffers(window);
        }
      }
    }
    
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