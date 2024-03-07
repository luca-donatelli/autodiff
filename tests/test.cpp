#include "../src/autodiff.cpp"
#include <cassert>

// Test Solver constructor
void testConstructor() {
    // Create a dummy function for testing
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) { return x[0] * x[0]; };

    // Create a solver object
    Solver solver(f);

    // Assertion: Solver object should be created successfully
    assert(true);
}

// Test computeGradientForward method
void testComputeGradientForward() {
    // Create a dummy function for testing
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) { return x[0] * x[0]; };

    // Create a solver object
    Solver solver(f);

    // Test input vector
    std::vector<double> x = {1.0};

    // Compute gradient using forward automatic differentiation
    auto grad = solver.computeGradientForward(x);

    // Assertion: Gradient should be computed correctly
    assert(std::abs((*grad)[0] - 2.0) < 1e-6);
}

// Test computeGradientBackward method
void testComputeGradientBackward() {
    // Create a dummy function for testing
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) { return x[0] * x[0]; };

    // Create a solver object
    Solver solver(f);

    // Test input vector
    std::vector<double> x = {1.0};

    // Compute gradient using backward automatic differentiation
    auto grad = solver.computeGradientBackward(x);

    // Assertion: Gradient should be computed correctly
    assert(std::abs((*grad)[0] - 2.0) < 1e-6);
}

// Test gradientDescent method
void testGradientDescent() {
    // Create a dummy function for testing (a simple quadratic function)
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) { return x[0] * x[0]; };

    // Create a solver object
    Solver solver(f);

    // Test initial guess
    std::vector<double> initialGuess = {2.0};

    // Perform gradient descent optimization
    double learningRate = 0.1;
    int numIterations = 100;
    auto result = solver.gradientDescent(initialGuess, learningRate, numIterations);

    // Assertion: Result should converge to zero
    assert(std::abs((*result)[0]) < 1e-6);
}

int main() {
    testConstructor();
    testComputeGradientForward();
    testComputeGradientBackward();
    testGradientDescent();

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
