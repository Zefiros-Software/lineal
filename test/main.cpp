/**
 * @cond ___LICENSE___
 *
 * Copyright (c) 2016-2019 Zefiros Software.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @endcond
 */
#include "lineal/lineal.h"

#include <armadillo>

#include <cstdlib>
#include <iostream>
#include <chrono>

template<typename tRow, typename tCol>
void bench_op(const tRow &row, const tCol &col)
{
    auto row_op = (row * 2.0) / 3.0;
    auto col_op = col * 2.0;

    auto start = std::chrono::high_resolution_clock::now();

    double alpha = 0.0;

    constexpr size_t iterations = 2000000;

    std::cout << lineal::is_scalar_addition<decltype(col_op)> << std::endl;

    for (size_t i = 0; i < iterations; ++i)
    {
        alpha += arma::as_scalar(row_op * col_op);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << alpha << std::endl;
    std::cout << "Took "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.0 / iterations
              << "ns per dot-product" << std::endl;
}

void main(int argc, char **argv)
{
    {
        lineal::Row<double> row(1024, lineal::fill::ones);
        lineal::Col<double> col(1024, lineal::fill::ones);
        lineal::ConstCol<double> const_col(col.begin(), col.size());

        bench_op(row, const_col);
    }
    {
        arma::rowvec row(1024, arma::fill::ones);
        arma::vec col(1024, arma::fill::ones);
        bench_op(row, col);
    }
    // system("pause");
    //     auto row_op0 = 1.0 / (4.0 + row + 3.0);
    //     auto row_op = 2.0 / row_op0;
    //     auto col_op = (5.0 - (col + 2.0) - 1.0) * 2.0;

    //     auto row_op = row + 2.0;
    //     auto col_op = const_col / 2.0;
    //
    //     auto start = std::chrono::high_resolution_clock::now();
    //
    //     double alpha = 0.0;
    //
    //     constexpr size_t iterations = 2000000;
    //
    //     std::cout << lineal::is_scalar_addition<decltype(col_op)> << std::endl;
    //
    //     for (size_t i = 0; i < iterations; ++i)
    //     {
    //         alpha += row_op * col_op;
    //     }
    //
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::cout << alpha << std::endl;
    //     std::cout << "Took "
    //               << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.0 / iterations
    //               << "ns per dot-product" << std::endl;
}