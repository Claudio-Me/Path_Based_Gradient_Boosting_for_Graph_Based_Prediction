/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/


#ifndef WLFAST_AUXILIARYMETHODS_H
#define WLFAST_AUXILIARYMETHODS_H

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "Graph.h"

using Eigen::IOFormat;
using Eigen::MatrixXd;
using namespace std;
using namespace GraphLibrary;

// Include paths assume the eigen3 headers are on the include path; build_kernels.sh
// passes -I for them, so the eigen3/ prefix used upstream is no longer needed.
//
// Upstream included <unsupported/Eigen/src/SparseExtra/MarketIO.h> directly.
// Eigen 5 rejects includes that reach into src/, and the Matrix Market readers
// are not used anywhere in this code, so the include is taken through the
// supported public header instead.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

namespace AuxiliaryMethods {
    // Simple function for converting a comma separated string into a vector of integers.
    vector<int> split_string(string s);

    // Reading a graph database from txt file.
    GraphDatabase read_graph_txt_file(string data_set_name);

    vector<int> read_classes(string data_set_name);
       vector<float> read_targets(string data_set_name);

    // Pairing function to map a pair of Labels to a single label.
    Label pairing(const Label a, const Label b);
    
    // Pairing function to map a vector of Labels to a single label.
    Label pairing(const vector<Label> labels);
}

#endif // WLFAST_AUXILIARYMETHODS_H
