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
    std::vector<double> x = {0.5, 1.0}; // Initial guess for optimization

    auto grad_backward = solver.computeGradientBackward(x);
    std::cout << "Gradient using backward differentiation:" << std::endl;
    for (size_t i = 0; i < grad_backward->size(); ++i) {
        std::cout << "Gradient w.r.t x[" << i << "]: " << (*grad_backward)[i] << std::endl;
    }

    return 0;
}
