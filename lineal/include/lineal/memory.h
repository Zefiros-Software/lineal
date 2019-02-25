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

namespace lineal
{
    namespace impl
    {
        template<typename tT>
        struct AlignedAllocator
        {
#if   defined(LINEAL_USE_TBB_ALLOC)
            static tT *aligned_malloc(size_t size)
            {
                return (tT *)scalable_malloc(sizeof(tT) * size);
            }

            static void aligned_free(tT *mem)
            {
                scalable_free(mem);
            }
#elif defined(LINEAL_USE_MKL_ALLOC)
            static tT *aligned_malloc(size_t size)
            {
                return (tT *)mkl_malloc(sizeof(tT) * size, 128);
            }

            static void aligned_free(tT *mem)
            {
                mkl_free(mem);
            }
#elif defined(LINEAL_HAVE_POSIX_MEMALIGN)
            static tT *aligned_malloc(size_t size)
            {
                tT *mem = NULL;

                const size_t byte_count = sizeof(tT) * size_t(size);
                const size_t alignment = (byte_count >= size_t(1024)) ? size_t(64) : size_t(16);

                int status = posix_memalign((void **)&mem, ((alignment >= sizeof(void *)) ? alignment : sizeof(void *)), byte_count);

                return (status == 0) ? mem : NULL;
            }

            static void aligned_free(tT *mem)
            {
                free(mem);
            }
#elif defined(_MSC_VER)
            static tT *aligned_malloc(size_t size)
            {
                const size_t byte_count = sizeof(tT) * size_t(size);
                const size_t alignment = (byte_count >= size_t(1024)) ? size_t(64) : size_t(16);

                return (tT *)_aligned_malloc(byte_count, alignment);
            }

            static void aligned_free(tT *mem)
            {
                _aligned_free(mem);
            }
#else
            static tT *aligned_malloc(size_t size)
            {
                return (tT *)malloc(sizeof(tT) * size);
            }

            static void aligned_free(tT *mem)
            {
                free(mem);
            }
#endif
        };

        template<typename tT>
        class Memory
        {
        public:

            Memory(size_t size, const ::lineal::fill &fill)
                : m_memory(AlignedAllocator<tT>::aligned_malloc(size)),
                  m_size(size),
                  m_owned(true)
            {
                switch (fill)
                {
                case ::lineal::fill::ones:
                    std::fill_n(m_memory, size, 1);
                    break;

                case ::lineal::fill::zeros:
                    std::fill_n(m_memory, size, 0);
                    break;

                default:
                    break;
                }
            }

            Memory(const Memory &) = delete;

            Memory(tT *auxiliary, size_t size)
                : m_memory(auxiliary),
                  m_size(size),
                  m_owned(false)
            {
            }

            Memory(Memory &&other)
                : m_memory(other.steal()),
                  m_size(other.m_size),
                  m_owned(true)
            {
            }

            ~Memory()
            {
                if (m_owned)
                {
                    AlignedAllocator<tT>::aligned_free(m_memory);
                }
            }

            tT *steal()
            {
                m_owned = false;
                return m_memory;
            }

            tT *raw()
            {
                return m_memory;
            }

            size_t size() const
            {
                return m_size;
            }

            const tT *begin() const
            {
                return m_memory;
            }

            const tT *end() const
            {
                return m_memory + m_size;
            }

            tT &operator[](size_t i)
            {
                return m_memory[i];
            }

        private:

            tT *m_memory;
            size_t m_size;
            bool m_owned;
        };
    }
}
