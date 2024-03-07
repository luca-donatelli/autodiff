#include <vector>
#include <functional>
#include <cmath>
#include <iostream>
#include <memory>

class Solver {
private:
    std::function<double(const std::vector<double>&)> func;

public:
    // Constructor
    Solver(std::function<double(const std::vector<double>&)> f) : func(f) {}

    // Method to compute gradient using forward automatic differentiation
    std::unique_ptr<std::vector<double>> computeGradientForward(const std::vector<double>& x, double epsilon = 1e-6) {
        auto grad = std::make_unique<std::vector<double>>();
        grad->reserve(x.size());

        // Compute function value at the current point
        double fx = func(x);

        // Compute gradient using forward automatic differentiation
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> x_plus = x;
            x_plus[i] += epsilon; // Perturb the current variable
            double fx_plus = func(x_plus); // Compute function value at perturbed point
            grad->push_back((fx_plus - fx) / epsilon); // Compute numerical gradient
        }

        return grad;
    }

    // Method to compute gradient using backward automatic differentiation
    std::unique_ptr<std::vector<double>> computeGradientBackward(const std::vector<double>& x, double epsilon = 1e-6) {
        auto grad = std::make_unique<std::vector<double>>();
        grad->reserve(x.size());

        // Compute function value at the current point
        double fx = func(x);

        // Compute gradient using backward automatic differentiation
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> x_plus = x;
            x_plus[i] += epsilon; // Perturb the current variable
            double fx_plus = func(x_plus); // Compute function value at perturbed point
            double derivative = (fx_plus - fx) / epsilon; // Compute derivative
            grad->push_back(derivative); // Push derivative to gradient vector
        }

        return grad;
    }

    // Method to compute higher-order derivative using central finite difference method
    double computeHigherOrderDerivative(const std::vector<double>& x, const std::vector<int>& orders, double epsilon = 1e-6) {
        double result = 0.0;

        // Compute higher-order derivative using central finite difference method
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> x_plus = x;
            x_plus[i] += epsilon; // Perturb the current variable
            double fx_plus = func(x_plus); // Compute function value at perturbed point
            double derivative = (fx_plus - func(x)) / epsilon; // Compute first derivative
            result += derivative * orders[i]; // Accumulate derivative with respect to each variable
        }

        return result;
    }

    // Method to perform gradient descent optimization
    std::unique_ptr<std::vector<double>> gradientDescent(const std::vector<double>& initialGuess, double learningRate, int numIterations) {
        auto currentPoint = std::make_unique<std::vector<double>>(initialGuess);

        // Perform gradient descent iterations
        for (int iter = 0; iter < numIterations; ++iter) {
            auto gradient = computeGradientBackward(*currentPoint); // Compute gradient
            for (size_t i = 0; i < currentPoint->size(); ++i) {
                (*currentPoint)[i] -= learningRate * (*gradient)[i]; // Update current point using gradient descent
            }
        }

        return currentPoint;
    }
};

int main() {
    // Define a function involving exponential and trigonometric functions using a lambda function
    std::function<double(const std::vector<double>&)> my_func = [](const std::vector<double>& x) -> double {
        // Example function: f(x, y) = e^x + sin(y)
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

    // Calculate gradients using both forward and backward differentiation methods
    auto grad_forward = solver.computeGradientForward(initialGuess);
    std::cout << "\nGradient using forward differentiation:" << std::endl;
    for (size_t i = 0; i < grad_forward->size(); ++i) {
        std::cout << "Gradient w.r.t x[" << i << "]: " << (*grad_forward)[i] << std::endl;
    }

    auto grad_backward = solver.computeGradientBackward(initialGuess);
    std::cout << "\nGradient using backward differentiation:" << std::endl;
    for (size_t i = 0; i < grad_backward->size(); ++i) {
        std::cout << "Gradient w.r.t x[" << i << "]: " << (*grad_backward)[i] << std::endl;
    }

    return 0;
}
