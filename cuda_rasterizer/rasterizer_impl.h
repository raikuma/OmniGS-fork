/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/**
	 * 从一块裸字节指针创建并分配T类型指针
	 * 使用时alignment = 2^n，作用是把chunk加上alignment-1后低n位置零，即以alignment为单位无条件进一，
	 * 从而align内存分配，使得ptr和chunk以alignment个byte为单位分块
	 */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset); // 输出结果：给成员指针分配内存起始位置
		chunk = reinterpret_cast<char*>(ptr + count); // 先ptr+count算下一个成员应当起始的内存位置，再转成通用过渡类型的指针
	}

	/**
	 * 3D点云状态
	 */
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * 图像像素状态
	 */
	struct ImageState
	{
		uint2* ranges; // 一个tile对应的第一个Gaus instance的序号（存放在.x）和下一个tile的第一个Gaus instance的序号（存放在.y）（排序后）；预留了像素数那么多的空间，但实际只需要tile数那么多？
		uint32_t* n_contrib;
		float* accum_alpha; // 一个tile的所有像素累积alpha值都为1时该tile渲染完成

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/**
	 * 
	 */
	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * 获取T（三种state）具有P个元素时所需的存放空间大小
	 * size = nullptr相当于从零开始，所以结果是直接累积量（加上alignment）
	 */
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};