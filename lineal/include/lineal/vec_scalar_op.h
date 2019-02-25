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

#include<optional>
#include<variant>

namespace lineal
{
    template<typename tVec, typename tT>
    constexpr bool vec_scalar_type = is_raw_vec<tVec> &&is_numeric<tT>;

    template<typename tVec, typename tT>
    constexpr bool scalar_vec_type = is_vec<tVec> &&is_numeric<tT>;

    template<typename tVec, typename tM, typename tA>
    constexpr bool vec_fma_type = is_raw_vec<tVec> &&is_numeric<tM> &&is_numeric<tA>;

    //     template<typename, typename = bool>
    //     constexpr bool is_scalar_addition = false;

    template<typename tOp>
    struct IsScalarAdditionHelper
    {
        template<typename tC>
        static auto test(double p) -> decltype(std::declval<tC>().apply_additive(p), std::true_type());

        template<typename>
        static std::false_type test(...);

        static const bool value = std::is_same<std::true_type, decltype(test<tOp>(0.0))>::value;
    };

    template<typename tOp>
    struct IsScalarMultiplicationHelper
    {
        template<typename tC>
        static auto test(double p) -> decltype(std::declval<tC>().apply_multiplicative(p), std::true_type());

        template<typename>
        static std::false_type test(...);

        static const bool value = std::is_same<std::true_type, decltype(test<tOp>(0.0))>::value;
    };

    template<typename tOp>
    constexpr bool is_scalar_addition = IsScalarAdditionHelper<tOp>::value;

    template<typename tOp>
    constexpr bool is_scalar_multiplicative = IsScalarMultiplicationHelper<tOp>::value;

    namespace impl
    {
        template<typename tVec>
        struct WrapRawSIMD
        {
            using tPackedHelper = PackedTypeHelper<typename tVec::value_type>;
            using tPacked = typename tPackedHelper::type;

            const tVec &vec;

            WrapRawSIMD(const tVec &v)
                : vec(v)
            {}

            auto &load_packed(size_t i)
            {
                m_packed = simdpp::load(vec.data() + i * tPackedHelper::count);
                return m_packed;
            }

        private:

            tPacked m_packed;
        };

        template<typename tVec>
        struct WrapOpSIMD
        {
            using tPackedHelper = PackedTypeHelper<typename tVec::value_type>;
            using tPacked = typename tPackedHelper::type;

            const tVec vec;

            WrapOpSIMD(const tVec &v)
                : vec(v)
            {
                vec.prepare_simd(m_scalar, m_mul);
            }

            auto &load_packed(size_t i)
            {
                vec.load_packed(i * tPackedHelper::count, m_packed, m_scalar, m_mul);
                return m_packed;
            }

        private:

            tPacked m_packed;
            tPacked m_scalar;
            tPacked m_mul;

        };

        template<typename tVec>
        using WrapSIMD = typename
                         std::conditional_t<is_raw_vec<tVec>, WrapRawSIMD<tVec>, typename std::conditional_t<is_vec_op<tVec>, WrapOpSIMD<tVec>, void>>;
    }

    namespace operations
    {
        template<typename tVec, typename tT>
        struct VecScalarOp
        {
            template<typename>
            friend struct impl::WrapOpSIMD;

            const tVec &vec;
            const tT scalar;
            impl::PackedType<tT> simd_scalar;

            using value_type = PreciseType<typename tVec::value_type, tT>;

            template<typename = std::enable_if_t<is_vec_op<tVec>>, typename = bool>
            VecScalarOp(const tVec &v, const tT &t)
                : vec_holder(v),
                  vec(vec_holder),
                  scalar(t)
            {}

            template<typename = std::enable_if_t<is_raw_vec<tVec>>>
                                                 VecScalarOp(const tVec &v, const tT &t)
                                                     : vec(v),
                                                       scalar(t)
            {}

            size_t size() const
            {
                return vec.size();
            }

        protected:

            typename std::conditional_t<is_vec_op<tVec>, tVec, constexpr bool> vec_holder;

            template<typename tPacked>
            void prepare_simd(tPacked &s, tPacked &) const
            {
                s = simdpp::load_splat(&scalar);
            }
        };

        template<typename tVec, typename tT>
    struct VecPlusScalar : VecScalarOp<tVec, tT>
        {
            template<typename>
            friend struct impl::WrapOpSIMD;

            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return v + scalar;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }

        private:

            template<typename tPacked>
            void load_packed(size_t i, tPacked &v, tPacked &s, tPacked &) const
            {
                v = simdpp::load(vec.data() + i);
                v = simdpp::add(v, s);
            }
        };

        template<typename tVec, typename tT>
    struct VecMinusScalar : VecScalarOp<tVec, tT>
        {
            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return v - scalar;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tT>
    struct ScalarMinusVec : VecScalarOp<tVec, tT>
        {
            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return scalar - v;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tT>
    struct VecTimesScalar : VecScalarOp<tVec, tT>
        {
            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return scalar * v;
            }

            template<typename tV>
            auto apply_multiplicative(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tT>
    struct VecDivScalar : VecScalarOp<tVec, tT>
        {
            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return v / scalar;
            }

            template<typename tV>
            auto apply_multiplicative(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tT>
    struct ScalarDivVec : VecScalarOp<tVec, tT>
        {
            using tParent = typename VecScalarOp<tVec, tT>;
            using tParent::VecScalarOp;

            template<typename tV>
            auto apply(const tV &v) const
            {
                return scalar / v;
            }

            template<typename tV>
            auto apply_multiplicative(const tV &v) const
            {
                return apply(v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };


        template<typename tVec, typename tM, typename tA>
        struct VecFMABase
        {
            const tVec &vec;
            const tA scalar;
            const tM mul;

            VecFMABase(const tVec &v, const tM &m, const tA &a)
                : vec(v),
                  scalar(a),
                  mul(m)
            {}

            using value_type = typename PreciseType<typename tVec::value_type, tM, tA>;

            size_t size() const
            {
                return vec.size();
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFMA : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul * vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar + v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul * v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFMAVecMinusScalar : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul * vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return v - scalar;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul * v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFMAScalarMinusVec : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul * vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar - v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul * v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDA : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return vec / mul;;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar + v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(v / mul);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDAVecMinusScalar : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return vec / mul;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return v - scalar
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(v / mul);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDAScalarMinusVec : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return vec / mul;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar - v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(v / mul);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDAInv : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul / vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar + v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul / v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDAInvVecMinusScalar : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul / vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return v  - scalar;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul / v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tM, typename tA>
    struct VecFDAInvScalarMinusVec : VecFMABase<tVec, tM, tA>
        {
            using tParent = VecFMABase<tVec, tM, tA>;
            using tParent::VecFMABase;
            using value_type = typename tParent::value_type;

            auto sub_operation() const
            {
                return mul / vec;
            }

            template<typename tV>
            auto apply_additive(const tV &v) const
            {
                return scalar - v;
            }

            template<typename tV>
            auto apply(const tV &v) const
            {
                return apply_additive(mul / v);
            }

            value_type operator[](size_t i) const
            {
                return apply(vec[i]);
            }
        };

        template<typename tVec, typename tT>
        constexpr bool valid_for_vec_scalar_op = ::lineal::is_raw_vec<tVec> &&::lineal::is_numeric<tT>;
    }
}

template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
auto operator+(const tVec &v, const tT &t)
{
    return ::lineal::operations::VecPlusScalar<tVec, tT>(v, t);
}

template<typename tVec, typename tT, typename std::enable_if_t<::lineal::scalar_vec_type<tVec, tT>, int> = 0>
auto operator+(const tT &t, const tVec &v)
{
    return v + t;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec + (op.scalar + t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec + (t - op.scalar);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar + t) - op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFMA<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFDA<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFDAInv<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * op.mul) + (t + op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / op.mul) + (t + op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.mul / op.vec) + (t + op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * op.mul) + (t - op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / op.mul) + (t - op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.mul / op.vec) + (t - op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (t + op.scalar) - (op.vec * op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (t + op.scalar) - (op.vec / op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator+(const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (t + op.scalar) - (op.mul / op.vec);
}



template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
auto operator-(const tVec &v, const tT &t)
{
    return ::lineal::operations::VecMinusScalar<tVec, tT>(v, t);
}

template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
auto operator-(const tT &t, const tVec &v)
{
    return ::lineal::operations::ScalarMinusVec<tVec, tT>(v, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec + (op.scalar - t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
{
    return (t - op.scalar) - op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec - (op.scalar + t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
{
    return (t + op.scalar) - op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar - t) - op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
{
    return (t - op.scalar) + op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFMAVecMinusScalar<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
{
    return ::lineal::operations::VecFMAScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFDAVecMinusScalar<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
{
    return ::lineal::operations::VecFDAScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
{
    return ::lineal::operations::VecFDAInvVecMinusScalar(op.vec, op.scalar, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
{
    return ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * op.mul) + (op.scalar - t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / op.mul) + (op.scalar - t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.mul / op.vec) + (op.scalar - t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) - (op.vec * op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) - (op.vec / op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) - (op.mul / op.vec);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * op.mul) - (op.scalar + t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / op.mul) - (op.scalar + t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.mul / op.vec) - (op.scalar + t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return (t + op.scalar) - (op.vec * op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return (t + op.scalar) - (op.vec / op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return (t + op.scalar) - (op.mul / op.vec);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar - t) - (op.vec * op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar - t) - (op.vec / op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar - t) - (op.mul / op.vec);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) + (op.vec * op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) + (op.vec / op.mul);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator-(const tT &t, const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return (t - op.scalar) + (op.mul / op.vec);
}






template<typename tVec, typename tT, typename std::enable_if_t<::lineal::scalar_vec_type<tVec, tT>, int> = 0>
auto operator*(const tVec &v, const tT &t)
{
    return ::lineal::operations::VecTimesScalar<tVec, tT>(v, t);
}

template<typename tVec, typename tT, typename std::enable_if_t<::lineal::scalar_vec_type<tVec, tT>, int> = 0>
auto operator*(const tT &t, const tVec &v)
{
    return v * t;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
{
    return (op.vec * t) + (op.scalar * t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
{
    return (op.vec * t) - (op.scalar * t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar * t) - (op.vec * t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec * (op.scalar * t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec * (t / op.scalar);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar * t) / op.vec;
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (op.mul * t)) + (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (t / op.mul)) + (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op, const tT &t)
{
    return ((op.mul * t) / op.vec) + (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (op.mul * t)) - (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (t / op.mul)) - (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return ((op.mul * t) / op.vec) - (op.scalar * t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar * t) - (op.vec * (op.mul * t));
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar * t) - (op.vec * (t / op.mul));
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator*(const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar * t) - ((op.scalar * t) / op.vec);
}






template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
auto operator/(const tVec &v, const tT &t)
{
    return ::lineal::operations::VecDivScalar<tVec, tT>(v, t);
}

template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
auto operator/(const tT &t, const tVec &v)
{
    return ::lineal::operations::ScalarDivVec<tVec, tT>(v, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
{
    return (op.vec / t) + (op.scalar / t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecPlusScalar<tVec0, tT0>, tT>(op, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
{
    return (op.vec / t) - (op.scalar / t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecMinusScalar<tVec0, tT0>, tT>(op, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar / t) - (op.vec / t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::ScalarMinusVec<tVec0, tT0>, tT0>(op, t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec * (op.scalar / t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
{
    return (t / op.scalar) / op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
{
    return op.vec / (op.scalar * t);
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
{
    return (op.scalar * t) / op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
{
    return (op.scalar / t) / op.vec;
}

template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
{
    return op.vec * (t / op.scalar);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (op.mul / t)) + (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / (op.mul * t)) + (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op, const tT &t)
{
    return ((op.mul / t) / op.vec) + (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFMA<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFMA<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDA<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDA<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDAInv<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDAInv<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec * (op.mul / t)) - (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.vec / (op.mul * t)) - (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op, const tT &t)
{
    return ((op.mul / t) / op.vec) - (op.scalar / t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFMAVecMinusScalar<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDAVecMinusScalar<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar / t) - (op.vec * (op.mul / t));
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar / t) - (op.vec / (op.mul * t));
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op, const tT &t)
{
    return (op.scalar / t) - ((op.mul / t) / op.vec);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFMAScalarMinusVec<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0>, tT>(op, t);
}

template<typename tVec0, typename tM0, typename tA0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
auto operator/(const tT &t, const ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tM0, tA0> &op)
{
    return ::lineal::operations::ScalarDivVec<::lineal::operations::VecFDAScalarMinusVec<tVec0, tM0, tA0>, tT>(op, t);
}

// Vec and scalar
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator+(const tVec &v, const tT &t)
// {
//     return ::lineal::operations::VecPlusScalar<tVec, tT>(v, t);
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator+(const tT &t, const tVec &v)
// {
//     return v + t;
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator-(const tVec &v, const tT &t)
// {
//     return ::lineal::operations::VecMinusScalar<tVec, tT>(v, t);
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator-(const tT &t, const tVec &v)
// {
//     return ::lineal::operations::ScalarMinusVec<tVec, tT>(v, t);
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator*(const tVec &v, const tT &t)
// {
//     return ::lineal::operations::VecTimesScalar<tVec, tT>(v, t);
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator*(const tT &t, const tVec &v)
// {
//     return v * t;
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator/(const tVec &v, const tT &t)
// {
//     return ::lineal::operations::VecDivScalar<tVec, tT>(v, t);
// }
//
// template<typename tVec, typename tT, typename std::enable_if_t<::lineal::vec_scalar_type<tVec, tT>, int> = 0>
// auto operator/(const tT &t, const tVec &v)
// {
//     return ::lineal::operations::ScalarDivVec<tVec, tT>(v, t);
// }
//
// // VecPlusScalar and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec + (op.scalar + t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec + (op.scalar - t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     return (t - op.scalar) - op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return (op.vec * t) + (op.scalar * t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return (op.vec / t) + (op.scalar / t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// // VecPlusScalar and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tVec &v)
// {
//     return (op.vec + v) + op.scalar;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     return op + v;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op, const tVec &v)
// {
//     return (op.vec - v) + op.scalar;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::VecPlusScalar<tVec0, tT0> &op)
// {
//     return (v - op.vec) - op.scalar;
// }
//
//
//
//
// // VecMinusScalar and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec + (t - op.scalar);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec - (op.scalar + t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     return (t + op.scalar) - op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return (op.vec * t) - (op.scalar * t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tT &t)
// {
//     return (op.vec / t) - (op.scalar / t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// // VecMinusScalar and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tVec &v)
// {
//     return (op.vec + v) - op.scalar;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     return op + v;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op, const tVec &v)
// {
//     return (op.vec - v) - op.scalar;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::VecMinusScalar<tVec0, tT0> &op)
// {
//     return (v - op.vec) + op.scalar;
// }
//
//
//
//
//
//
// // ScalarMinusVec and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
// {
//     return (op.scalar + t) - op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
// {
//     return (op.scalar - t) - op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     return op.vec - (op.scalar - t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
// {
//     return (op.scalar * t) - (op.vec * t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tT &t)
// {
//     return (op.scalar / t) - (op.vec / t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// // ScalarMinusVec and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tVec &v)
// {
//     return (v - op.vec) + op.scalar;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     return op + v;
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op, const tVec &v)
// {
//     return op.scalar - (op.vec + v);
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::ScalarMinusVec<tVec0, tT0> &op)
// {
//     return (v + op.vec) - op.scalar;
// }
//
//
//
//
//
//
// // VecTimesScalar and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFMA<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFMAVecMinusScalar<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     return ::lineal::operations::VecFMAScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec * (op.scalar * t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec * (op.scalar / t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     return (t / op.scalar) / op.vec;
// }
//
// // VecTimesScalar and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::VecTimesScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
//
//
//
//
//
// // VecDivScalar and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFDA<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFDAVecMinusScalar<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     return ::lineal::operations::VecFDAScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec * (t / op.scalar);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tT &t)
// {
//     return op.vec / (op.scalar * t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     return (t * op.scalar) / op.vec;
// }
//
// // VecDivScalar and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::VecDivScalar<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::VecDivScalar<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
//
//
//
//
//
// // ScalarDivVec and scalar
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFDAInv<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator+(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     return op + t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
// {
//     return ::lineal::operations::VecFDAInvVecMinusScalar<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator-(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     return ::lineal::operations::VecFDAInvScalarMinusVec<tVec0, tT0, tT>(op.vec, op.scalar, t);
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
// {
//     return (t * op.scalar) / op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator*(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     return op * t;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tT &t)
// {
//     return (op.scalar / t) / op.vec;
// }
//
// template<typename tVec0, typename tT0, typename tT, typename std::enable_if_t<::lineal::is_numeric<tT>, int> = 0>
// auto operator/(const tT &t, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     return op.vec * (t / op.scalar);
// }
//
// // ScalarDivVec and Vec
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator+(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator+(const tVec &v, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_vec<tVec>, int> = 0>
// auto operator-(const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op, const tVec &v)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
//
// template<typename tVec0, typename tT0, typename tVec, typename std::enable_if_t<::lineal::is_raw_vec<tVec>, int> = 0>
// auto operator-(const tVec &v, const ::lineal::operations::ScalarDivVec<tVec0, tT0> &op)
// {
//     static_assert(false, "needs to be implemented with extra operation class");
// }
