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
#include "lineal/types.h"
#include "lineal/simd.h"

#include <type_traits>


namespace lineal
{
    namespace impl
    {
        template<typename...>
        struct PreciseTypeImpl;

        template<typename tT, typename tS, typename... tOthers>
        struct PreciseTypeImpl<tT, tS, tOthers...>
        {
            using type = typename PreciseTypeImpl<typename PreciseTypeImpl<tT, tS>::type, tOthers...>::type;
        };

        template<typename tT, typename tS>
        struct PreciseTypeImpl<tT, tS>
        {
            using tSelf = typename PreciseTypeImpl<tT, tS>;
            static constexpr bool is_tT_float = std::is_floating_point_v<tT>;
            static constexpr bool is_tS_float = std::is_floating_point_v<tS>;
            static constexpr bool is_only_one_float = tSelf::is_tT_float != tSelf::is_tS_float;
            using tFloatType = std::conditional_t<tSelf::is_tT_float, tT, tS>;
            static constexpr bool is_tT_larger = sizeof(tT) > sizeof(tS);
            using tLargeType = std::conditional_t<tSelf::is_tT_larger, tT, tS>;

            using type = std::conditional_t<tSelf::is_only_one_float, typename tSelf::tFloatType, typename tSelf::tLargeType>;
        };

        template<typename tT>
        struct PreciseTypeImpl<tT>
        {
            using type = tT;
        };
    }
}

namespace lineal
{
    enum class fill
    {
        ones,
        zeros,
        none
    };

    template<typename>
    class Vec;
    template<typename>
    class Row;
    template<typename>
    class Col;
    template<typename>
    class ConstVec;
    template<typename>
    class ConstRow;
    template<typename>
    class ConstCol;

    namespace operations
    {
        template<typename, typename>
        struct VecScalarOp;
        template<typename, typename, typename>
        struct VecFMABase;
    }

    template<typename... tTypes>
    using PreciseType = typename impl::PreciseTypeImpl<tTypes...>::type;

    enum class Orientation
    {
        OrientationRow,
        OrientationCol
    };

    template<Orientation, typename...>
    struct VecOrientationHelper
    {
        constexpr static bool check()
        {
            return false;
        }
    };

    template<Orientation tOrient, template<typename...> typename tVec, typename tT>
    struct VecOrientationHelper<tOrient, tVec<tT>>
    {
        constexpr static bool check()
        {
            if constexpr(std::is_base_of_v<Col<tT>, tVec<tT>> || std::is_base_of_v<ConstCol<tT>, tVec<tT>>)
            {
                return tOrient == Orientation::OrientationCol;
            }

            if constexpr(std::is_base_of_v<Row<tT>, tVec<tT>> || std::is_base_of_v<ConstRow<tT>, tVec<tT>>)
            {
                return tOrient == Orientation::OrientationRow;
            }

            return false;
        }
    };

    template<Orientation tOrient, template<typename...> typename tOp, typename tVec, typename tT>
    struct VecOrientationHelper<tOrient, tOp<tVec, tT>>
    {
        constexpr static bool check()
        {
            if constexpr(std::is_base_of_v<operations::VecScalarOp<tVec, tT>, tOp<tVec, tT>>)
            {
                return VecOrientationHelper<tOrient, tVec>::check();
            }

            return false;
        }
    };

    template<Orientation tOrient, template<typename...> typename tOp, typename tVec, typename tM, typename tA>
    struct VecOrientationHelper<tOrient, tOp<tVec, tM, tA>>
    {
        constexpr static bool check()
        {
            if constexpr(std::is_base_of_v<operations::VecFMABase<tVec, tT, tA>, tOp<tVec, tT, tA>>)
            {
                return VecOrientationHelper<tOrient, tVec>::check();
            }

            return false;
        }
    };

    template<typename tVec>
    constexpr bool is_row = VecOrientationHelper<Orientation::OrientationRow, tVec>::check();

    template<typename tVec>
    constexpr bool is_col = VecOrientationHelper<Orientation::OrientationCol, tVec>::check();

    //     template<typename tT>
    //     constexpr bool is_row<Row<tT>> = true;
    //     template<typename tT>
    //     constexpr bool is_row<ConstRow<tT>> = true;
    //     template<typename tT>
    //     constexpr bool is_row<Col<tT>> = false;
    //     template<typename tT>
    //     constexpr bool is_row<ConstCol<tT>> = false;
    //
    //     template<template <typename...> typename tOp, typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<tOp<tVec, tM, tA>> = std::is_base_of_v<operations::VecFMABase<tVec, tM, tA>, tOp<tVec, tM, tA>> ?
    //                                                is_row<tVec> : false;
    //
    //     template<template <typename...> typename tOp, typename tVec, typename tT>
    //     constexpr bool is_row<tOp<tVec, tT>> = std::is_base_of_v<operations::VecScalarOp<tVec, tT>, tOp<tVec, tT>> ? is_row<tVec> : false;

    //template<typename tVec, typename tT>
    //     constexpr bool is_row<operations::VecScalarOp<tVec, tT>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFMABase<tVec, tM, tA>> = is_row<tVec>;
    //
    //     template<typename tOp>
    //     struct ParentHelper
    //     {
    //         constexpr static bool check()
    //         {
    //             if constexpr(!std::is_same_v<void, typename tOp::tParent>)
    //             {
    //                 return is_row<typename tOp::tParent>;
    //             }
    //
    //             return false;
    //         }
    //     };
    //
    //     template<typename tOp>
    //     constexpr bool is_row<tOp> = ParentHelper<tOp>::check();

    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::VecPlusScalar<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::VecMinusScalar<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::ScalarMinusVec<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::VecTimesScalar<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::VecDivScalar<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_row<operations::ScalarDivVec<tVec, tScalar>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFMA<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFMAVecMinusScalar<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFMAScalarMinusVec<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDA<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDAVecMinusScalar<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDAScalarMinusVec<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDAInv<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDAInvVecMinusScalar<tVec, tM, tA>> = is_row<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_row<operations::VecFDAInvScalarMinusVec<tVec, tM, tA>> = is_row<tVec>;

    //     template<typename...>
    //     constexpr bool is_col = false;
    //
    //     template<typename tT>
    //     constexpr bool is_col<Row<tT>> = false;
    //     template<typename tT>
    //     constexpr bool is_col<ConstRow<tT>> = false;
    //     template<typename tT>
    //     constexpr bool is_col<Col<tT>> = true;
    //     template<typename tT>
    //     constexpr bool is_col<ConstCol<tT>> = true;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::VecPlusScalar<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::VecMinusScalar<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::ScalarMinusVec<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::VecTimesScalar<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::VecDivScalar<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tScalar>
    //     constexpr bool is_col<operations::ScalarDivVec<tVec, tScalar>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFMA<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFMAVecMinusScalar<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFMAScalarMinusVec<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDA<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDAVecMinusScalar<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDAScalarMinusVec<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDAInv<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDAInvVecMinusScalar<tVec, tM, tA>> = is_col<tVec>;
    //     template<typename tVec, typename tM, typename tA>
    //     constexpr bool is_col<operations::VecFDAInvScalarMinusVec<tVec, tM, tA>> = is_col<tVec>;

    template<typename tVec>
    constexpr bool is_vec = is_row<tVec> || is_col<tVec>;

    template<typename>
    constexpr bool is_raw_vec = false;
    template<typename tT>
    constexpr bool is_raw_vec<Row<tT>> = true;
    template<typename tT>
    constexpr bool is_raw_vec<ConstRow<tT>> = true;
    template<typename tT>
    constexpr bool is_raw_vec<Col<tT>> = true;
    template<typename tT>
    constexpr bool is_raw_vec<ConstCol<tT>> = true;

    template<typename tT>
    constexpr bool is_numeric = std::is_integral_v<tT> || std::is_floating_point_v<tT>;

    template<typename tVec>
    constexpr bool is_vec_op = is_vec<tVec> &&!is_raw_vec<tVec>;
}
