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
#pragma once
#include "lineal/vec_scalar_op.h"
#include "lineal/types.h"

#include <mkl.h>

#include <numeric>

namespace lineal
{
    namespace operations
    {
        template<typename tVec0, typename tVec1>
        struct VecVecOp
        {
            const tVec0 &vec0;
            const tVec1 &vec1;

            using value_type = typename PreciseType<typename tVec0::value_type, typename tVec1::value_type>;

            template<typename = std::enable_if_t<is_vec_op<tVec0>>, typename = std::enable_if_t<is_vec_op<tVec1>>>
                                                                                                VecVecOp(const tVec0 &v0, const tVec1 &v1)
                                                                                                    : vec0_holder(v0),
                                                                                                      vec0(vec0_holder),
                                                                                                      vec1_holder(v1),
                                                                                                      vec1(vec1_holder)
            {}

            template<typename = std::enable_if_t<is_raw_vec<tVec0>>, typename = std::enable_if_t<is_vec_op<tVec1>>, typename = bool>
            VecVecOp(const tVec0 &v0, const tVec1 &v1)
                : vec0(v0),
                  vec1_holder(v1),
                  vec1(vec1_holder)
            {}

            template<typename = std::enable_if_t<is_vec_op<tVec0>>, typename = std::enable_if_t<is_raw_vec<tVec1>>, typename = bool, typename = bool>
            VecVecOp(const tVec0 &v0, const tVec1 &v1)
                : vec0_holder(v0),
                  vec0(vec0_holder),
                  vec1(v1)
            {}

            template<typename = std::enable_if_t<is_raw_vec<tVec0>>, typename = std::enable_if_t<is_raw_vec<tVec1>>, typename = bool, typename = bool, typename = bool>
            VecVecOp(const tVec0 &v0, const tVec1 &v1)
                : vec0(v0),
                  vec1(v1)
            {}

            size_t size() const
            {
                return vec0.size();
            }

        protected:

            typename std::conditional_t<is_vec_op<tVec0>, tVec0, constexpr bool> vec0_holder;
            typename std::conditional_t<is_vec_op<tVec1>, tVec1, constexpr bool> vec1_holder;
        };

        template<typename tRow, typename tCol>
    struct InProd : VecVecOp<tRow, tCol>
        {
        public:

            using tParent = VecVecOp<tRow, tCol>;
            using tParent::VecVecOp;
            using tPackedHelper = typename impl::PackedTypeHelper<value_type>;
            using tPacked = typename tPackedHelper::type;

            //             operator value_type() const
            //             {
            //                 value_type tmp = 0;
            //
            //                 for (size_t i = 0, end = vec0.size(); i < end; ++i)
            //                 {
            //                     tmp += vec0[i] * vec1[i];
            //                 }
            //
            //                 return tmp;
            //             }

            template < typename = std::enable_if_t < is_vec_op<tRow> || is_vec_op<tCol >> >
            value_type eval() const
            {
                //                 size_t i = 0;
                //
                //                 value_type init = 0;
                //                 tPacked tmp_inprod = ::simdpp::load_splat(&init);
                //
                //                 impl::WrapSIMD<tRow> v0(vec0);
                //                 impl::WrapSIMD<tCol> v1(vec1);
                //
                //                 for (size_t end = vec0.size() / tPackedHelper::count; i < end; ++i)
                //                 {
                //                     tmp_inprod = ::simdpp::add(::simdpp::mul(v0.load_packed(i), v1.load_packed(i)), tmp_inprod);
                //                     // tmp_inprod = ::simdpp::fmadd(v0.load_packed(i), v1.load_packed(i), tmp_inprod);
                //                 }
                //
                //                 value_type tmp = ::simdpp::reduce_add(tmp_inprod);
                //                 i *= tPackedHelper::count;
                //
                //                 for (size_t end = vec0.size(); i < end; ++i)
                //                 {
                //                     tmp += vec0[i] * vec1[i];
                //                 }

                if constexpr(is_vec_op<tCol>)
                {
                    lineal::Col<value_type> tmp_col(vec1.size());

                    for (size_t i = 0; i < vec0.size(); ++i)
                    {
                        tmp_col[i] = vec1[i];
                    }

                    return vec0 * tmp_col;
                }
                else
                {
                    lineal::Row<value_type> tmp_row(vec0.size());

                    for (size_t i = 0; i < vec0.size(); ++i)
                    {
                        tmp_row[i] = vec0[i];
                    }

                    return tmp_row * vec1;
                }
            }

            template < typename = std::enable_if_t < is_raw_vec<tRow> &&is_raw_vec<tCol >>, typename = bool >
            value_type eval() const
            {
                const int N = static_cast<int>(vec0.size());
                return cblas_ddot(N, vec0.data(), 1, vec1.data(), 1);
            }

            operator value_type() const
            {
                return eval();
            }
        };

        template<typename tRow, typename tCol>
        constexpr bool valid_for_inproduct = is_row<tRow> &&is_col<tCol>;
    }
}

template<typename tRow, typename tCol, std::enable_if_t<::lineal::operations::valid_for_inproduct<tRow, tCol>, int> = 0>
auto operator*(const tRow &row, const tCol &col)
{
    return ::lineal::operations::InProd<tRow, tCol>(row, col).eval();
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_col<tCol>, int> = 0>
auto operator*(const ::lineal::operations::VecTimesScalar<tRow, tT> &op, const tCol &col)
{
    return (op.vec * col) * op.scalar;
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const tRow &row, const ::lineal::operations::VecTimesScalar<tCol, tT> &op)
{
    return (row * op.vec) * op.scalar;
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_col<tCol>, int> = 0>
auto operator*(const ::lineal::operations::VecDivScalar<tRow, tT> &op, const tCol &col)
{
    return (op.vec * col) / op.scalar;
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const tRow &row, const ::lineal::operations::VecDivScalar<tCol, tT> &op)
{
    return (row * op.vec) / op.scalar;
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const ::lineal::operations::VecTimesScalar<tRow, tT> &row,
               const ::lineal::operations::VecTimesScalar<tCol, tT> &col)
{
    return (row.vec * col.vec) * (row.scalar * col.scalar);
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const ::lineal::operations::VecTimesScalar<tRow, tT> &row,
               const ::lineal::operations::VecDivScalar<tCol, tT> &col)
{
    return (row.vec * col.vec) * (row.scalar / col.scalar);
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const ::lineal::operations::VecDivScalar<tRow, tT> &row,
               const ::lineal::operations::VecTimesScalar<tCol, tT> &col)
{
    return (row.vec * col.vec) * (col.scalar / row.scalar);
}

template<typename tRow, typename tT, typename tCol, std::enable_if_t<::lineal::is_row<tRow>, int> = 0>
auto operator*(const ::lineal::operations::VecDivScalar<tRow, tT> &row,
               const ::lineal::operations::VecDivScalar<tCol, tT> &col)
{
    return (row.vec * col.vec) / (row.scalar * col.scalar);
}

template<typename tOp0, typename tOp1, typename std::enable_if_t<::lineal::is_scalar_addition<tOp0>, int> = 0>
auto operator+(const tOp0 &op0, const tOp1 &op1)
{
    return op0.apply_additive(op0 + op1);
}

// template < typename tOp0, typename tOp1, typename std::enable_if_t < !::lineal::is_scalar_addition<tOp0>, int > = 0 >
// {
//     return
// }