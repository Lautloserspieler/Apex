#include "apex/engine.h"

#include <iostream>

int main(int argc, char** argv) {
    apex::Engine engine;

    if (!engine.loadModel("dummy_model.gguf")) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    std::string prompt = (argc > 1) ? argv[1] : "Hello, Apex!";
    std::string output = engine.generate(prompt, 64);

    std::cout << output << std::endl;
    return 0;
}
