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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 查找最高有效位的位置，位数从1开始（例如若第18位为1，再高位都是0，则返回18）（第18位即2^17比指数多1所以叫next-highest？？？）
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2; // 二分查找
		if (n >> msb) // 右移msb位，如果n的后msb位之前没有有效数字，则msb-=step即往低位查找，否则往高位查找
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void markAllVisible(int P,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	present[idx] = true;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,                   // 输入：3D Gaus数
	const float2* points_xy, // 输入：图像空间 像素坐标
	const float* depths,     // 输入：相机坐标 深度
	const uint32_t* offsets, // 输入：覆盖的tile点次（累积值）
	uint64_t* gaussian_keys_unsorted,   // 输出：（未排序）instance的key
	uint32_t* gaussian_values_unsorted, // 输出：（未排序）instance的Gaus idx
	int* radii,              // 输入：图像空间 半径
	dim3 grid /*输入：tile grid的维度shape(格子数目)*/)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0) // 只有preprocess中未被跳过的Gaus会参与处理、产生key
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 每个Gaus预留了覆盖tile数个空位给它的key
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 图像空间中根据中心和半径，计算Gaus对应的范围方框（用方框两个对角顶点落在的tile的X序号和Y序号表示，因此下面遍历这个结果相当于直接按序号遍历方框覆盖的tile）
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// 每个Gaus拥有其覆盖tile数个64位key，即每个Gaus instance拥有1个key
		// 每个key的高32位是tile的一维序号（先行后列），低32位是相机坐标深度，目的是这些instance排序时先按tile排序，再按深度排序
		// key对应的value是Gaus的idx，用于从instance反向查找对应的Gaus
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// // Generates one key/value pair for all Gaussian / tile overlaps. 
// // Run once per Gaussian (1:N mapping).
// __global__ void duplicateWithKeys(
// 	int P,                   // 输入：3D Gaus数
// 	const float2* points_xy, // 输入：图像空间 像素坐标
// 	const float* depths,     // 输入：相机坐标 深度
// 	const uint32_t* offsets, // 输入：覆盖的tile点次（累积值）
// 	uint64_t* gaussian_keys_unsorted,   // 输出：（未排序）instance的key
// 	uint32_t* gaussian_values_unsorted, // 输出：（未排序）instance的Gaus idx
// 	int* radii,              // 输入：图像空间 半径
// 	int* radii_x,            // 输入：图像空间 x方向半径
// 	dim3 grid /*输入：tile grid的维度shape(格子数目)*/)
// {
// 	auto idx = cg::this_grid().thread_rank();
// 	if (idx >= P)
// 		return;

// 	// Generate no key/value pair for invisible Gaussians
// 	if (radii[idx] > 0) // 只有preprocess中未被跳过的Gaus会参与处理、产生key
// 	{
// 		// Find this Gaussian's offset in buffer for writing keys/values.
// 		// 每个Gaus预留了覆盖tile数个空位给它的key
// 		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
// // int2 rect_min, rect_max;
// 		uint2 rect_min, rect_max;
// 		// 图像空间中根据中心和半径，计算Gaus对应的范围方框（用方框两个对角顶点落在的tile的X序号和Y序号表示，因此下面遍历这个结果相当于直接按序号遍历方框覆盖的tile）
// // getRectCyclic(points_xy[idx], radii_x[idx], radii[idx], rect_min, rect_max, grid);
// 		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

// 		// For each tile that the bounding rect overlaps, emit a 
// 		// key/value pair. The key is |  tile ID  |      depth      |,
// 		// and the value is the ID of the Gaussian. Sorting the values 
// 		// with this key yields Gaussian IDs in a list, such that they
// 		// are first sorted by tile and then by depth. 
// 		// 每个Gaus拥有其覆盖tile数个64位key，即每个Gaus instance拥有1个key
// 		// 每个key的高32位是tile的一维序号（先行后列），低32位是相机坐标深度，目的是这些instance排序时先按tile排序，再按深度排序
// 		// key对应的value是Gaus的idx，用于从instance反向查找对应的Gaus
// // int gx = (int)grid.x;
// // int instance_x_idx;
// 		for (int y = rect_min.y; y < rect_max.y; y++)
// 		{
// // instance_x_idx = 0;
// 			for (int x = rect_min.x; x < rect_max.x; x++)
// 			{
// // if (instance_x_idx >= gx) continue;
// // int x_actual = x;
// // if (x_actual < 0) x_actual += gx;
// // else if (x_actual >= gx) x_actual -= gx;
// // uint64_t key = y * grid.x + x_actual;
// 				uint64_t key = y * grid.x + x;
// 				key <<= 32;
// 				key |= *((uint32_t*)&depths[idx]);
// 				gaussian_keys_unsorted[off] = key;
// 				gaussian_values_unsorted[off] = idx;
// 				off++;
// // instance_x_idx++;
// 			}
// 		}
// 	}
// }

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::LonlatRasterizer::markVisible(
	int P,
	bool* present)
{
	markAllVisible << <(P + 255) / 256, 256 >> > (
		P,
		present);
}

/**
 * 创建一个新GeometryState对象，存放P个3D点的数据
 * 内层：所有点的同种数据放在一个指针下（连续）；外层：不同数据占用的内存块也连续
 */
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

/**
 * 创建一个新ImageState对象，存放N个像素的数据
 * 内层：所有像素的同种数据放在一个指针下（连续）；外层：不同数据占用的内存块也连续
 */
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

/**
 * 创建一个新BinningState对象，存放num_rendered个实际等效渲染点的数据
 * 内层：所有点的同种数据放在一个指针下（连续）；外层：不同数据占用的内存块也连续
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// 只有out_color（渲染结果图像）和radii（gaus半径，用于剔除gaus、确定更新哪些gaus的梯度等，即控制densify过程）是真正输出，其余都是输入或上下文
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P/*3D Gaus数目*/, int D/*active_sh_degree_，仅用于preprocess*/, int M/*dc和rest谐波之和(1+15)*/,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	const bool render_depth)
{
	// 传进来的实际上是tan(fov/2)；fov转换成焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 先计算P个3D点所需的空间（char数，即byte数），再通过传入的torch Tensor的resize_接口分配空间，最后在这段空间上划分State各成员指针的位置
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 如果没有radii则使用内置（但实际上总是会在外界弄好一个tensor传进来）
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 划分tile：每个维度上tile数都需要进1——不能去尾
	// perspective相机的tile是直接画方格
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 先计算width * height个像素所需的空间（char数，即byte数），再通过传入的torch Tensor的resize_接口分配空间，最后在这段空间上划分State各成员指针的位置
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// 这个Gaussian rasterizer只能自主计算RGB颜色，其余色彩空间需要外界预计算
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 计算每个3D Gaus点的2D图像空间的协方差、范围框，并且SH转RGB（计算图像空间的颜色）
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 输入：tiles_touched：每个Gaus点覆盖的tile数
	// 输出：point_offsets: 到每个Gaus点为止，所有Gaus累积覆盖tile次数（“人次“的概念，点次）
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	//num_rendered = 截至最后一个Gaus，所有Gaus累积覆盖tile点次（全渲染总共需要计算的等效的点数目）
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	// binning存放参与渲染的等效点云
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 对于每个Gaus，若可视，则生成其实际参与渲染次数（即覆盖tile个数）个instance，instance按所在tile序号和对应Gaus深度生成key用于排序
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);

	// tile格子总数以2为底的对数，向上取整，例如有100格，则bit=7（2^6=64,2^7=128）
	// 用于计算key的最高有效位，提供给cuda排序（有利于性能？）
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 升序排序Gaus instances，先深度后tile序号（即同一tile的放一起，内部按深度）
	// 排序是为了能够生成后面的索引ranges，使每个tile能轻易定位自己包含哪些instance
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit/*低32位是深度，完整比较；高32位的tile序号可根据tile总数确定最大序号从而确定比较其低bit位即可*/);

	// 各个像素的ranges初始化为(0,0)
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	// 计算ranges，即每个tile包含的Gaus instance序号
	// ranges.x：instance序号起点
	// ranges.y：下一个tile的instance序号起点，即当前tile的instance序号终点+1
	// 并行遍历排序后的每个instance，
	// 若其为第一个，则其对应tile的ranges.x=0
	// 若其为分界点（即其前一个instance属于前一个tile，与它所属不同），则前一个tile的ranges.y=它所属的tile的ranges.x=它的idx
	// 若其为最后一个，则其对应tile的ranges.y=instance总数（点次）
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	// Let each tile blend its range of Gaussians independently in parallel
	if (render_depth)
	{
		FORWARD::renderDepth(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			// means3D,
			// viewmatrix,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color);
	}
	else
	{
		const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
		FORWARD::render(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color);
	}

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix, // grad_outputs[0]，即d{loss}_d{forward输出}[0]，由torch自动计算，以下backward定义的是d{forward输出}_d{forward输入}
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R); // 通过ctx上下文获取Gaus instance数目R
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 从ctx传进来的和forward一样，实际上是tan(fov/2)；fov转换成焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 划分tile：每个维度上tile数都需要进1——不能去尾
	// 在screen-space划分
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// 协方差矩阵可能是预计算的完整输入的，也可能是输入scale和rot在rasterizer内部计算的，注意两种情况处理不同
	// 一般而言用的是第二种
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot);
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// 只有out_color（渲染结果图像）和radii（gaus半径，用于剔除gaus、确定更新哪些gaus的梯度等，即控制densify过程）是真正输出，其余都是输入或上下文
int CudaRasterizer::LonlatRasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P/*3D Gaus数目*/, int D/*active_sh_degree_，仅用于preprocess*/, int M/*dc和rest谐波之和(1+15)*/,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* cam_pos,
	const bool prefiltered,
	float* out_color,
	int* radii)
// int* radii_x)
{
	// 先计算P个3D点所需的空间（char数，即byte数），再通过传入的torch Tensor的resize_接口分配空间，最后在这段空间上划分State各成员指针的位置
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 如果没有radii则使用内置（但实际上总是会在外界弄好一个tensor传进来）
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 划分tile：每个维度上tile数都需要进1——不能去尾
	// 【相机模型】perspective相机的tile是直接画方格
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 先计算width * height个像素所需的空间（char数，即byte数），再通过传入的torch Tensor的resize_接口分配空间，最后在这段空间上划分State各成员指针的位置
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// 这个Gaussian rasterizer只能自主计算RGB颜色，其余色彩空间需要外界预计算
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 计算每个3D Gaus点的2D图像空间的协方差、范围框，并且SH转RGB（计算图像空间的颜色）
	FORWARD::preprocessLonlat(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		radii,
// radii_x,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 输入：tiles_touched：每个Gaus点覆盖的tile数
	// 输出：point_offsets: 到每个Gaus点为止，所有Gaus累积覆盖tile次数（“人次“的概念，点次）
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	//num_rendered = 截至最后一个Gaus，所有Gaus累积覆盖tile点次（全渲染总共需要计算的等效的点数目）
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	// binning存放参与渲染的等效点云
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 对于每个Gaus，若可视，则生成其实际参与渲染次数（即覆盖tile个数）个instance，instance按所在tile序号和对应Gaus深度生成key用于排序
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
// radii_x,
		tile_grid
		);

	// tile格子总数以2为底的对数，向上取整，例如有100格，则bit=7（2^6=64,2^7=128）
	// 用于计算key的最高有效位，提供给cuda排序（有利于性能？）
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 升序排序Gaus instances，先深度后tile序号（即同一tile的放一起，内部按深度）
	// 排序是为了能够生成后面的索引ranges，使每个tile能轻易定位自己包含哪些instance
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit/*低32位是深度，完整比较；高32位的tile序号可根据tile总数确定最大序号从而确定比较其低bit位即可*/);

	// 各个像素的ranges初始化为(0,0)
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	// 计算ranges，即每个tile包含的Gaus instance序号
	// ranges.x：instance序号起点
	// ranges.y：下一个tile的instance序号起点，即当前tile的instance序号终点+1
	// 并行遍历排序后的每个instance，
	// 若其为第一个，则其对应tile的ranges.x=0
	// 若其为分界点（即其前一个instance属于前一个tile，与它所属不同），则前一个tile的ranges.y=它所属的tile的ranges.x=它的idx
	// 若其为最后一个，则其对应tile的ranges.y=instance总数（点次）
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::LonlatRasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* campos,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix, // grad_outputs[0]，即d{loss}_d{forward输出}[0]，由torch自动计算，以下backward定义的是d{forward输出}_d{forward输入}
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dpx_dt,
	float* dpy_dt)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R); // 通过ctx上下文获取Gaus instance数目R
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 划分tile：每个维度上tile数都需要进1——不能去尾
	// 在screen-space划分
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// 协方差矩阵可能是预计算的完整输入的，也可能是输入scale和rot在rasterizer内部计算的，注意两种情况处理不同
	// 一般而言用的是第二种
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocessLonlat(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		width, height,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		(float3*)dpx_dt,
		(float3*)dpy_dt);
}
