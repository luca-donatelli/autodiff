#include "../src/autodiff.cpp"

int main() {
    // Define a function involving exponential and trigonometric functions using a lambda function
    std::function<double(const std::vector<double>&)> my_func = [](const std::vector<double>& x) -> double {
        // Example function: f(x, y) = x^2 + sin(y) +y^2 +1
        return (x[0] * x[0]) + sin(x[1]) + (x[1] * x[1]) + 1;
    };

    // Create a solver object
    Solver solver(my_func);

    // Example usage of gradient descent
    std::vector<double> initialGuess = {0.5, 1.0}; // Initial guess for optimization
    double learningRate = 0.01; // Learning rate for gradient descent
    int numIterations = 1000; // Number of iterations for gradient descent
    auto result = solver.gradientDescent(initialGuess, learningRate, numIterations);

    // Output the result of gradient descent
    std::cout << "Result of gradient descent:" << std::endl;
    for (size_t i = 0; i < result->size(); ++i) {
        std::cout << "x[" << i << "]: " << (*result)[i] << std::endl;
    }
}