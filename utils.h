#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

/*
This is a helper function that reads the kernel function from a file
and returns it as a char array
*/
std::string read_kernel(const std::string& filename) {
    std::string kernel_text;

    std::ifstream kernel_reader;
    kernel_reader.open(filename, std::ios::in);

    std::string line;
    while (std::getline(kernel_reader, line)) {
        kernel_text.append(line);
        kernel_text.append("\n");
    }
    kernel_reader.close();

    return kernel_text;
}
