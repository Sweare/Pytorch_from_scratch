#include "../Tensor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

// ─────────────────────────────────────────────
//  Tiny test harness
// ─────────────────────────────────────────────
struct TestRunner {
    int passed = 0;
    int failed = 0;
    bool save  = false;
    std::ofstream outFile;
    std::string   outPath = "test_results.txt";

    TestRunner(bool saveToFile, const std::string& path = "test_results.txt")
        : save(saveToFile), outPath(path)
    {
        if (save) {
            outFile.open(outPath);
            if (!outFile) {
                std::cerr << "[WARN] Could not open " << outPath << " for writing.\n";
                save = false;
            } else {
                outFile << "========================================\n";
                outFile << "  Tensor Test Results\n";
                outFile << "========================================\n\n";
            }
        }
    }

    ~TestRunner() {
        std::string summary =
            "\n========================================\n"
            "  Results: " + std::to_string(passed) + " passed, " +
            std::to_string(failed) + " failed\n"
            "========================================\n";
        std::cout << summary;
        if (save && outFile.is_open()) {
            outFile << summary;
        }
    }

    // Write a line to console (and optionally to file)
    void log(const std::string& msg) {
        std::cout << msg << "\n";
        if (save && outFile.is_open()) outFile << msg << "\n";
    }

    // Run a named test; fn returns true on pass
    void run(const std::string& name, std::function<bool(TestRunner&)> fn) {
        log("\n---- " + name + " ----");
        bool ok = false;
        try {
            ok = fn(*this);
        } catch (const std::exception& e) {
            log("  [EXCEPTION] " + std::string(e.what()));
        }
        if (ok) { ++passed; log("  [PASS]"); }
        else    { ++failed; log("  [FAIL]"); }
    }

    // Expect-equal helper (prints diff on failure)
    template<typename A, typename B>
    bool expect(const A& got, const B& expected, const std::string& label = "") {
        bool ok = (got == expected);
        if (!ok) {
            std::ostringstream ss;
            ss << "  MISMATCH";
            if (!label.empty()) ss << " (" << label << ")";
            ss << ": got " << got << ", expected " << expected;
            log(ss.str());
        }
        return ok;
    }

    bool expectTrue(bool cond, const std::string& label = "") {
        if (!cond) log("  CONDITION FALSE" + (label.empty() ? "" : " (" + label + ")"));
        return cond;
    }
};

// ─────────────────────────────────────────────
//  Tensor pretty-printer
// ─────────────────────────────────────────────
template<typename T>
std::string tensorToString(Tensor<T>& t, const std::vector<int64_t>& shape) {
    std::ostringstream ss;
    if (shape.size() == 1) {
        ss << "[ ";
        for (int64_t c = 0; c < shape[0]; ++c) {
            ss << t(c);
            if (c + 1 < shape[0]) ss << ", ";
        }
        ss << " ]";
    } else if (shape.size() == 2) {
        int64_t rows = shape[0], cols = shape[1];
        for (int64_t r = 0; r < rows; ++r) {
            ss << "  [ ";
            for (int64_t c = 0; c < cols; ++c) {
                ss << t(r, c);
                if (c + 1 < cols) ss << ", ";
            }
            ss << " ]\n";
        }
    } else if (shape.size() == 3) {
        int64_t d0 = shape[0], d1 = shape[1], d2 = shape[2];
        for (int64_t i = 0; i < d0; ++i) {
            ss << "  [slice " << i << "]\n";
            for (int64_t j = 0; j < d1; ++j) {
                ss << "    [ ";
                for (int64_t k = 0; k < d2; ++k) {
                    ss << t(i, j, k);
                    if (k + 1 < d2) ss << ", ";
                }
                ss << " ]\n";
            }
        }
    } else {
        ss << "  (display not supported for rank " << shape.size() << ")";
    }
    return ss.str();
}

// Helper to fill a 2D tensor with sequential values starting at `start`
template<typename T>
void fill2D(Tensor<T>& t, int64_t rows, int64_t cols, T start = 0) {
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            t(r, c) = start + static_cast<T>(r * cols + c);
}

// ─────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────

bool test_constructor(TestRunner& tr) {
    Tensor<float> t({3, 4});
    bool ok = tr.expect(t.getSize(), (int64_t)12, "getSize 3x4");

    Tensor<float> t2({2, 3, 4});
    ok &= tr.expect(t2.getSize(), (int64_t)24, "getSize 2x3x4");

    tr.log("  Shape 3x4 => size " + std::to_string(t.getSize()));
    tr.log("  Shape 2x3x4 => size " + std::to_string(t2.getSize()));
    return ok;
}

bool test_element_access(TestRunner& tr) {
    Tensor<int> t({3, 4});
    fill2D(t, 3, 4, 1);  // 1..12

    tr.log("  3x4 tensor (values 1..12):");
    tr.log(tensorToString(t, {3, 4}));

    bool ok = tr.expect(t(0, 0), 1,  "t(0,0)");
    ok &= tr.expect(t(0, 3), 4,  "t(0,3)");
    ok &= tr.expect(t(2, 3), 12, "t(2,3)");

    // getElement variadic version
    ok &= tr.expect(t.getElement(1, 2), 7, "getElement(1,2)");
    return ok;
}

bool test_transpose(TestRunner& tr) {
    Tensor<int> t({3, 4});
    fill2D(t, 3, 4, 1);

    tr.log("  Original 3x4:");
    tr.log(tensorToString(t, {3, 4}));

    Tensor<int> tp = t.transpose();   // default: dims 0 & 1 => 4x3

    tr.log("  Transposed 4x3:");
    tr.log(tensorToString(tp, {4, 3}));

    // t(r,c) == tp(c,r)
    bool ok = true;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            if (t(r, c) != tp(c, r)) {
                tr.log("  Mismatch at t(" + std::to_string(r) + "," +
                       std::to_string(c) + ")");
                ok = false;
            }
        }
    return ok;
}

bool test_isContiguous(TestRunner& tr) {
    Tensor<int> t({3, 4});
    fill2D(t, 3, 4, 0);

    bool before = t.isContiguous();
    Tensor<int> tp = t.transpose();
    bool after = tp.isContiguous();

    tr.log("  isContiguous before transpose: " + std::string(before ? "true" : "false"));
    tr.log("  isContiguous after  transpose: " + std::string(after  ? "true" : "false"));

    return tr.expectTrue(before,  "original contiguous") &&
           tr.expectTrue(!after,  "transposed non-contiguous");
}

bool test_makeContiguous(TestRunner& tr) {
    Tensor<int> t({3, 4});
    fill2D(t, 3, 4, 1);

    Tensor<int> tp   = t.transpose();
    Tensor<int> cont = tp.makeContiguous();

    tr.log("  Transposed (logical 4x3) raw data:");
    tr.log(tensorToString(tp, {4, 3}));

    tr.log("  After makeContiguous (physical 4x3):");
    tr.log(tensorToString(cont, {4, 3}));

    bool ok = cont.isContiguous();
    ok &= tr.expectTrue(ok, "makeContiguous => isContiguous");

    // Values must still match the transposed view
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            ok &= tr.expect(cont(r, c), tp(r, c), "cont==tp at (" +
                            std::to_string(r) + "," + std::to_string(c) + ")");
    return ok;
}

bool test_view(TestRunner& tr) {
    Tensor<int> t({2, 6});
    fill2D(t, 2, 6, 0);

    tr.log("  Original 2x6:");
    tr.log(tensorToString(t, {2, 6}));

    std::vector<int64_t> ns{3, 4};
    Tensor<int> v = t.view(std::span<const int64_t>(ns));

    tr.log("  View as 3x4:");
    tr.log(tensorToString(v, {3, 4}));

    // Shares data — modifying v changes t
    v(0, 0) = 99;
    bool sharedData = (t(0, 0) == 99);
    tr.log("  After v(0,0)=99, t(0,0) = " + std::to_string(t(0,0)) +
           (sharedData ? "  (shared ✓)" : "  (NOT shared ✗)"));

    return tr.expectTrue(sharedData, "view shares data");
}

bool test_reshape(TestRunner& tr) {
    Tensor<int> t({3, 4});
    fill2D(t, 3, 4, 1);

    // Contiguous path: reshape == view
    std::vector<int64_t> ns1{2, 6};
    Tensor<int> r1 = t.reshape(std::span<const int64_t>(ns1));
    tr.log("  Reshape 3x4 -> 2x6 (contiguous):");
    tr.log(tensorToString(r1, {2, 6}));
    bool ok = tr.expect(r1.getSize(), (int64_t)12, "size preserved");

    // Non-contiguous path: transpose then reshape
    Tensor<int> tp = t.transpose();   // 4x3, non-contiguous
    std::vector<int64_t> ns2{2, 6};
    Tensor<int> r2 = tp.reshape(std::span<const int64_t>(ns2));
    tr.log("  Reshape (transposed 4x3) -> 2x6 (non-contiguous path):");
    tr.log(tensorToString(r2, {2, 6}));
    ok &= tr.expect(r2.getSize(), (int64_t)12, "size preserved after non-cont reshape");
    return ok;
}

bool test_addition(TestRunner& tr) {
    Tensor<float> a({2, 3});
    Tensor<float> b({2, 3});

    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
            a(r, c) = static_cast<float>(r * 3 + c + 1);
            b(r, c) = 10.0f;
        }

    tr.log("  A (2x3):");
    tr.log(tensorToString(a, {2, 3}));
    tr.log("  B (2x3, all 10):");
    tr.log(tensorToString(b, {2, 3}));

    a += b;
    tr.log("  A += B:");
    tr.log(tensorToString(a, {2, 3}));

    bool ok = true;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
            float expected = static_cast<float>(r * 3 + c + 1) + 10.0f;
            ok &= tr.expect(a(r, c), expected,
                            "a(" + std::to_string(r) + "," + std::to_string(c) + ")");
        }
    return ok;
}

bool test_addition_size_mismatch(TestRunner& tr) {
    Tensor<float> a({2, 3});
    Tensor<float> b({3, 2});
    bool threw = false;
    try { a += b; }
    catch (const std::invalid_argument&) { threw = true; }
    tr.log("  operator+= with mismatched shapes threw: " + std::string(threw ? "yes ✓" : "no ✗"));
    return tr.expectTrue(threw, "size mismatch throws");
}

bool test_rank_mismatch_getElement(TestRunner& tr) {
    Tensor<int> t({3, 4});
    bool threw = false;
    try { t.getElement(1, 2, 3); }  // 3 indices for rank-2 tensor
    catch (const std::invalid_argument&) { threw = true; }
    tr.log("  getElement with wrong rank threw: " + std::string(threw ? "yes ✓" : "no ✗"));
    return tr.expectTrue(threw, "rank mismatch throws");
}

bool test_3d_tensor(TestRunner& tr) {
    Tensor<int> t({2, 3, 4});
    int val = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                t(i, j, k) = val++;

    tr.log("  3D tensor 2x3x4:");
    tr.log(tensorToString(t, {2, 3, 4}));

    // Transpose dims 0 and 2 => shape 4x3x2
    Tensor<int> tp = t.transpose(0, 2);
    tr.log("  Transposed (dim0<->dim2) => 4x3x2:");
    tr.log(tensorToString(tp, {4, 3, 2}));

    bool ok = true;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                ok &= (t(i, j, k) == tp(k, j, i));
    tr.log("  Values match original (transposed indices): " + std::string(ok ? "yes ✓" : "no ✗"));
    return ok;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    bool saveOutput = false;
    std::string outPath = "test_results.txt";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--save" || arg == "-s") {
            saveOutput = true;
        } else if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
            outPath = argv[++i];
            saveOutput = true;
        }
    }

    if (saveOutput)
        std::cout << "[INFO] Writing results to: " << outPath << "\n";

    TestRunner tr(saveOutput, outPath);

    tr.run("Constructor & getSize",          test_constructor);
    tr.run("Element Access (operator() & getElement)", test_element_access);
    tr.run("Transpose 2D",                   test_transpose);
    tr.run("isContiguous",                   test_isContiguous);
    tr.run("makeContiguous",                 test_makeContiguous);
    tr.run("View",                           test_view);
    tr.run("Reshape",                        test_reshape);
    tr.run("operator+=",                     test_addition);
    tr.run("operator+= size mismatch (error handling)", test_addition_size_mismatch);
    tr.run("getElement rank mismatch (error handling)",  test_rank_mismatch_getElement);
    tr.run("3D Tensor & Transpose",          test_3d_tensor);

    return tr.failed == 0 ? 0 : 1;
}
