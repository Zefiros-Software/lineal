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
#include "simdpp/simd.h"

#include <cstdint>

namespace lineal
{
    namespace simd
    {
#if defined(SIMDPP_HAS_AVX_SUPPORT)
        constexpr bool has_sse = true;
        constexpr bool has_avx = true;
        constexpr bool has_simd = true;
        constexpr size_t register_bits = 256;
#define LINEAL_HAS_SIMD
#elif defined(SIMDPP_HAS_SSE_SUPPORT)
        constexpr bool has_sse = true;
        constexpr bool has_avx = true;
        constexpr bool has_simd = true;
        constexpr size_t register_bits = 128;
#define LINEAL_HAS_SIMD
#else
        constexpr bool has_sse = false;
        constexpr bool has_avx = false;
        constexpr bool has_simd = false;
#endif
    }


    namespace impl
    {
        template<typename>
        struct PackedTypeImpl;

        template<>
        struct PackedTypeImpl<double>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 64;
            using type = simdpp::float64<count>;
        };

        template<>
        struct PackedTypeImpl<float>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 32;
            using type = simdpp::float32<count>;
        };

        template<>
        struct PackedTypeImpl<uint64_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 64;
            using type = simdpp::uint64<count>;
        };

        template<>
        struct PackedTypeImpl<uint32_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 32;
            using type = simdpp::uint32<count>;
        };

        template<>
        struct PackedTypeImpl<uint16_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 16;
            using type = simdpp::uint16<count>;
        };

        template<>
        struct PackedTypeImpl<uint8_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 8;
            using type = simdpp::uint8<count>;
        };

        template<>
        struct PackedTypeImpl<int64_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 64;
            using type = simdpp::int64<count>;
        };

        template<>
        struct PackedTypeImpl<int32_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 32;
            using type = simdpp::int32<count>;
        };

        template<>
        struct PackedTypeImpl<int16_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 16;
            using type = simdpp::int16<count>;
        };

        template<>
        struct PackedTypeImpl<int8_t>
        {
            constexpr static size_t count = ::lineal::simd::register_bits / 8;
            using type = simdpp::int8<count>;
        };

        template<typename tT>
        struct PackedTypeHelperImpl
        {
            using type = PackedTypeImpl<tT>;
        };

        template<>
        struct PackedTypeHelperImpl<size_t>
        {
            using type = typename std::conditional_t<(sizeof(size_t) == 8), PackedTypeImpl<uint64_t>, PackedTypeImpl<uint32_t>>;
        };

        template<typename tT>
        using PackedTypeHelper = typename PackedTypeHelperImpl<tT>::type;

        template<typename tT>
        using PackedType = typename PackedTypeHelper<tT>::type;
    }
}