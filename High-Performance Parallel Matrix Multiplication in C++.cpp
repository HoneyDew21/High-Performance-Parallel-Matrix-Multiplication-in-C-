#include <iostream>
#include <vector>
#include <tbb/tbb.h>
#include <cmath>

using namespace std;

// Function to add two matrices
void addMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    tbb::parallel_for(0, n, [&](int i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    });
}

// Function to subtract two matrices
void subtractMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    tbb::parallel_for(0, n, [&](int i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    });
}

// Naive matrix multiplication with cache blocking
void blockedMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    int blockSize = 32; // Block size for cache optimization
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); ++i) {
                    for (int j = jj; j < min(jj + blockSize, n); ++j) {
                        for (int k = kk; k < min(kk + blockSize, n); ++k) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// Strassen's recursive multiplication
void strassenMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    if (n <= 64) { // Base case for blocked multiplication
        blockedMultiply(A, B, C, n);
        return;
    }

    int newSize = n / 2;
    vector<vector<int>> A11(newSize, vector<int>(newSize));
    vector<vector<int>> A12(newSize, vector<int>(newSize));
    vector<vector<int>> A21(newSize, vector<int>(newSize));
    vector<vector<int>> A22(newSize, vector<int>(newSize));
    vector<vector<int>> B11(newSize, vector<int>(newSize));
    vector<vector<int>> B12(newSize, vector<int>(newSize));
    vector<vector<int>> B21(newSize, vector<int>(newSize));
    vector<vector<int>> B22(newSize, vector<int>(newSize));

    // Divide matrices into 4 sub-matrices each
    tbb::parallel_for(0, newSize, [&](int i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    });

    // Allocate matrices for Strassen's computations
    vector<vector<int>> M1(newSize, vector<int>(newSize));
    vector<vector<int>> M2(newSize, vector<int>(newSize));
    vector<vector<int>> M3(newSize, vector<int>(newSize));
    vector<vector<int>> M4(newSize, vector<int>(newSize));
    vector<vector<int>> M5(newSize, vector<int>(newSize));
    vector<vector<int>> M6(newSize, vector<int>(newSize));
    vector<vector<int>> M7(newSize, vector<int>(newSize));

    vector<vector<int>> temp1(newSize, vector<int>(newSize));
    vector<vector<int>> temp2(newSize, vector<int>(newSize));

    // M1 = (A11 + A22) * (B11 + B22)
    addMatrices(A11, A22, temp1, newSize);
    addMatrices(B11, B22, temp2, newSize);
    strassenMultiply(temp1, temp2, M1, newSize);

    // M2 = (A21 + A22) * B11
    addMatrices(A21, A22, temp1, newSize);
    strassenMultiply(temp1, B11, M2, newSize);

    // M3 = A11 * (B12 - B22)
    subtractMatrices(B12, B22, temp1, newSize);
    strassenMultiply(A11, temp1, M3, newSize);

    // M4 = A22 * (B21 - B11)
    subtractMatrices(B21, B11, temp1, newSize);
    strassenMultiply(A22, temp1, M4, newSize);

    // M5 = (A11 + A12) * B22
    addMatrices(A11, A12, temp1, newSize);
    strassenMultiply(temp1, B22, M5, newSize);

    // M6 = (A21 - A11) * (B11 + B12)
    subtractMatrices(A21, A11, temp1, newSize);
    addMatrices(B11, B12, temp2, newSize);
    strassenMultiply(temp1, temp2, M6, newSize);

    // M7 = (A12 - A22) * (B21 + B22)
    subtractMatrices(A12, A22, temp1, newSize);
    addMatrices(B21, B22, temp2, newSize);
    strassenMultiply(temp1, temp2, M7, newSize);

    // Combine results into C
    tbb::parallel_for(0, newSize, [&](int i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    });
}

// Main function
int main() {
    int n = 256; // Matrix size (must be a power of 2 for simplicity)
    vector<vector<int>> A(n, vector<int>(n, 1));
    vector<vector<int>> B(n, vector<int>(n, 2));
    vector<vector<int>> C(n, vector<int>(n, 0));

    // Perform matrix multiplication
    strassenMultiply(A, B, C, n);

    // Display the result matrix (optional for large matrices)
    cout << "Result Matrix:" << endl;
    for (const auto& row : C) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
