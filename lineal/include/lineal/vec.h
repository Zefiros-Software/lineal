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
#include "lineal/memory.h"

namespace lineal
{
    template<typename tT>
    class Vec
    {
    public:

        using value_type = tT;

        Vec(size_t size = 0, const ::lineal::fill &fill = ::lineal::fill::none)
            : m_memory(size, fill),
              m_raw(m_memory.raw())
        {
        }

        Vec(tT *auxiliary, size_t size)
            : m_memory(auxiliary, size),
              m_raw(m_memory.raw())
        {
        }

        Vec(const Vec &) = delete;

        tT &operator[](const size_t i)
        {
            return m_raw[i];
        }

        const tT &operator[](const size_t i) const
        {
            return m_raw[i];
        }

        tT *begin()
        {
            return m_raw;
        }

        tT *end()
        {
            return m_raw + m_memory.size();
        }

        size_t size() const
        {
            return m_memory.size();
        }

        const tT *data() const
        {
            return m_raw;
        }

    private:

        lineal::impl::Memory<tT> m_memory;
        tT *m_raw;
    };

    template<typename tT>
    class ConstVec
    {
    public:

        using value_type = tT;

        //      Vec(size_t size = 0, const ::lineal::fill &fill = ::lineal::fill::none)
        //          : m_memory(size, fill),
        //          m_raw(m_memory.raw())
        //      {
        //      }

        ConstVec(const tT *auxiliary, size_t size)
            : m_raw(auxiliary),
              m_size(size)
        {
        }

        ConstVec(const ConstVec &) = delete;

        const tT &operator[](const size_t i) const
        {
            return m_raw[i];
        }

        const tT *begin() const
        {
            return m_raw;
        }

        const tT *end() const
        {
            return m_raw + m_size;
        }

        size_t size() const
        {
            return m_size;
        }

        const tT *data() const
        {
            return m_raw;
        }

    private:

        const tT *m_raw;
        const size_t m_size;
    };

    template<typename tT>
    class Col : public Vec<tT>
    {
    public:

        using tParent = Vec<tT>;

        using Vec::Vec;

        Col(const Col &) = delete;
    };

    template<typename tT>
    class ConstCol : public ConstVec<tT>
    {
    public:

        using tParent = ConstVec<tT>;

        using ConstVec::ConstVec;

        ConstCol(const ConstCol &) = delete;
    };

    template<typename tT>
    class Row : public Vec<tT>
    {
    public:

        using tParent = Vec<tT>;

        using Vec::Vec;

        Row(const Row &) = delete;
    };

    template<typename tT>
    class ConstRow : public ConstVec<tT>
    {
    public:

        using tParent = ConstVec<tT>;

        using ConstVec::ConstVec;

        ConstRow(const ConstRow &) = delete;
    };

    template<typename tVec>
    typename tVec::value_type sum(const tVec &v)
    {
        using value_type = typename tVec::value_type;

        impl::Memory<value_type> tmp_simd(4, fill::zeros);
        impl::PackedType<value_type> tmp_sum = ::simdpp::load(tmp_simd.raw());

        size_t i = 0;

        for (size_t iEnd = v.size() >> 2; i < iEnd; ++i)
        {
            impl::PackedType<value_type> tmp_vec = v.load_packed(i << 2);
            tmp_sum = simdpp::add(tmp_sum, tmp_vec);
        }

        simdpp::store(tmp_simd.raw(), tmp_sum);
        value_type res = lineal::sum(tmp_simd);

        i <<= 2;

        for (size_t iEnd = v.size(); i < iEnd; ++i)
        {
            res += v[i];
        }

        return res;
    }

    template<typename tT>
    tT sum(const impl::Memory<tT> &m)
    {
        tT init = 0;
        return std::accumulate(m.begin(), m.end(), init);
    }
}
