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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx]; // 世界坐标
	glm::vec3 dir = pos - campos; // 世界坐标下相对相机的方向向量
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		// active_sh_degree到1时坐标就参与颜色计算了，等级越高坐标的阶次越高
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// 结果颜色clamp到非负。记录这一操作并告知梯度计算
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	// t是相机坐标，在perspective下有限制，在panoramic下不应有/或只裁掉近点
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 相机坐标到像素坐标的雅可比
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2DLonlat(const float3& mean, const int width, const int height, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 【相机模型】t是相机坐标，在perspective下有限制，在panoramic下不应有/或只裁掉近点
	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

	// 【相机模型】相机坐标到像素坐标的雅可比
	float trxztrxz = t.x * t.x + t.z * t.z;
	float trxztrxz_inv = 1.0f / (trxztrxz + 0.0000001f);
	float trxz = sqrtf(trxztrxz);
	float trxz_inv = 1.0f / (trxz + 0.0000001f);
	float trtr = trxztrxz + t.y * t.y;
	float trtr_inv = 1.0f / (trtr + 0.0000001f);

	float W_div_2pi = width * 0.5f * M_1_PIf32;
	float H_div_pi = height * M_1_PIf32;

	float dpx_dtx = W_div_2pi * t.z * trxztrxz_inv;
	float dpx_dtz = -W_div_2pi * t.x * trxztrxz_inv;

	float dpy_dtx = -H_div_pi * t.x * t.y * trxz_inv * trtr_inv;
	float dpy_dty = H_div_pi * trxz * trtr_inv;
	float dpy_dtz = -H_div_pi * t.z * t.y * trxz_inv * trtr_inv;

	glm::mat3 J = glm::mat3(
		dpx_dtx, 0.0f, dpx_dtz,
		dpy_dtx, dpy_dty, dpy_dtz,
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
// float inv_cos_lat = trxz_inv * sqrtf(trtr);
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P/*3D Gaus数目*/, int D/*active_sh_degree_，值域{0,1,2,3}*/, int M/*dc和rest谐波之和(1+15)*/,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs, /*features_dc和features_rest*/
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix, /*Tcw*/
	const float* projmatrix, /*full transform*/
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// preprocessCUDA对每个3D Gaus进行预处理
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 统计Gaus的半径和覆盖tile数，预处理之后若仍为零，则其不参与渲染处理
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// 剔除不可视点；实际上只剔除了相机坐标下z<=0.2的点，即前方的点，不涉及相机模型
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // screen-space的坐标

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 一般没有precompute的cov3D，传入是nullptr，因此使用scale和rot计算3D协方差
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 计算投影到2D screen-space的协方差
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// 求conic：2D协方差的逆矩阵
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 计算2D协方差的特征值->计算Gaus的半径->计算Gaus的范围框
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 着色：SH转RGB
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // 相机坐标 深度
	radii[idx] = my_radius; // 图像空间 半径
	points_xy_image[idx] = point_image; // 图像空间 像素坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; // 协方差、透明度 打包
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // 用方框粗略估计的覆盖tile数
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 每个thread处理一个像素，同一个tile内的像素thread在同一个block（线程块，GPU的feature）里
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block(); // 当前block即当前tile
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 水平方向上block总数（tile总数，=tile_grid.x）
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前tile最靠近原点的像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };        // 当前tile最远离原点的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前像素坐标 = 最靠近原点像素 + block.thread_index（二维）
	uint32_t pix_id = W * pix.y + pix.x; // 当前像素的总序号（图像铺开到一维时像素的序号，先行后列）
	float2 pixf = { (float)pix.x, (float)pix.y };  // 当前像素坐标

	// 图像外的像素为invalid，直接done=true，不参与重采样、不累积alpha，但参与协作从全局获取Gaus数据到block
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // 之前预计算的当前tile的instance序号范围
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 将当前tile的instance分为BLOCK_SIZE（=每个tile最大像素数）个一组来处理
	int toDo = range.y - range.x; // 当前像素还未用于渲染的instance数，初始等于当前tile用于渲染的instance总数

	// Allocate storage for batches of collectively fetched data.
	// 用于一个tile内所有像素存放取得的Gaus数据。数组长度 = maxThreadsPerBlock
	// 共享内存是按线程块分配的，因此块中的所有线程都可以访问同一共享内存 https://developer.nvidia.com/zh-cn/blog/using-shared-memory-cuda-cc/
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; // 论文(2)透射比
	uint32_t contributor = 0; // 计数器：参与构成当前像素的Gaus instance数（当实际上continue跳过了一个instance但没有done时，也会计数）
	uint32_t last_contributor = 0; // 辅助记录，最后一个实际参与构成当前像素（没有跳过）的instance是该像素所有instance中的第几个
	float C[CHANNELS] = { 0 }; // 辅助记录，当前像素的累积颜色

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count统计block内done为true的线程总数，所有像素都done之后，该block的所有线程一起break退出
		// 所有像素done之前，已经done的像素仍参与共享数据获取，但自己不再新增contributor，颜色不再改变
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 一轮中，即对同一个i，整个block共同并行取出最多BLOCK_SIZE个instance，block内每个thread取1个，同步：等到这些instance都取好
		int progress = i * BLOCK_SIZE + block.thread_rank(); // 是当前block内的第几个instance
		if (range.x + progress < range.y) // range.x + progress = 是总的第几个instance，只有当前tile范围内的会取出
		{
			int coll_id = point_list[range.x + progress]; // instance对应的Gaus的id
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // instance对应的Gaus的像素坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // instance对应的Gaus的协方差、透明度
		}
		block.sync();

		// Iterate over current batch
		// 一轮中，即对同一个i，每个像素分别利用刚刚取出的最多BLOCK_SIZE个instance累积自身的渲染结果颜色
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j]; // {x=conic[0][0], y=conic[0][1], z=conic[1][1], w=opacity}
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // 指数项，定义参考EWA splatting (20)
			if (power > 0.0f) // 数值稳健性：一个Gaus分布函数的指数不应该大于0
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 这里的alpha不是用(2)式中的变量定义的，而是直接定义为sigmoid激活后的opacity属性 * Gaus分布取值
			float alpha = min(0.99f, con_o.w * exp(power)); // 数值稳健性1：防除以0，大alpha clamp到0.99
			if (alpha < 1.0f / 255.0f) // 数值稳健性2：防除以0，跳过小alpha
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) // 数值稳健性3：alpha累积停在0.9999时而非1，即 新T<1-0.9999 时
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T; // 累乘T

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 背景色也是等效为一个混合项累加到式(3)上去的
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 每个thread处理一个像素，同一个tile内的像素thread在同一个block（线程块，GPU的feature）里
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/BLOCK_X * BLOCK_Y)
renderDepthCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	// const float* __restrict__ orig_points,
	// const float* __restrict__ viewmatrix,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block(); // 当前block即当前tile
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 水平方向上block总数（tile总数，=tile_grid.x）
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前tile最靠近原点的像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };        // 当前tile最远离原点的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前像素坐标 = 最靠近原点像素 + block.thread_index（二维）
	uint32_t pix_id = W * pix.y + pix.x; // 当前像素的总序号（图像铺开到一维时像素的序号，先行后列）
	float2 pixf = { (float)pix.x, (float)pix.y };  // 当前像素坐标

	// 图像外的像素为invalid，直接done=true，不参与重采样、不累积alpha，但参与协作从全局获取Gaus数据到block
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // 之前预计算的当前tile的instance序号范围
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 将当前tile的instance分为BLOCK_SIZE（=每个tile最大像素数）个一组来处理
	int toDo = range.y - range.x; // 当前像素还未用于渲染的instance数，初始等于当前tile用于渲染的instance总数

	// Allocate storage for batches of collectively fetched data.
	// 用于一个tile内所有像素存放取得的Gaus数据。数组长度 = maxThreadsPerBlock
	// 共享内存是按线程块分配的，因此块中的所有线程都可以访问同一共享内存 https://developer.nvidia.com/zh-cn/blog/using-shared-memory-cuda-cc/
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; // 论文(2)透射比
	uint32_t contributor = 0; // 计数器：参与构成当前像素的Gaus instance数（当实际上continue跳过了一个instance但没有done时，也会计数）
	uint32_t last_contributor = 0; // 辅助记录，最后一个实际参与构成当前像素（没有跳过）的instance是该像素所有instance中的第几个
	float C[CHANNELS] = { 0 }; // 辅助记录，当前像素的累积颜色

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count统计block内done为true的线程总数，所有像素都done之后，该block的所有线程一起break退出
		// 所有像素done之前，已经done的像素仍参与共享数据获取，但自己不再新增contributor，颜色不再改变
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 一轮中，即对同一个i，整个block共同并行取出最多BLOCK_SIZE个instance，block内每个thread取1个，同步：等到这些instance都取好
		int progress = i * BLOCK_SIZE + block.thread_rank(); // 是当前block内的第几个instance
		if (range.x + progress < range.y) // range.x + progress = 是总的第几个instance，只有当前tile范围内的会取出
		{
			int coll_id = point_list[range.x + progress]; // instance对应的Gaus的id
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // instance对应的Gaus的像素坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // instance对应的Gaus的协方差、透明度
		}
		block.sync();

		// Iterate over current batch
		// 一轮中，即对同一个i，每个像素分别利用刚刚取出的最多BLOCK_SIZE个instance累积自身的渲染结果颜色
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j]; // {x=conic[0][0], y=conic[0][1], z=conic[1][1], w=opacity}
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // 指数项，定义参考EWA splatting (20)
			if (power > 0.0f) // 数值稳健性：一个Gaus分布函数的指数不应该大于0
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 这里的alpha不是用(2)式中的变量定义的，而是直接定义为sigmoid激活后的opacity属性 * Gaus分布取值
			float alpha = min(0.99f, con_o.w * exp(power)); // 数值稳健性1：防除以0，大alpha clamp到0.99
			if (alpha < 1.0f / 255.0f) // 数值稳健性2：防除以0，跳过小alpha
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) // 数值稳健性3：alpha累积停在0.9999时而非1，即 新T<1-0.9999 时
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// int idx = collected_id[j] * 3;
			// float3 p_orig = { orig_points[idx], orig_points[idx + 1], orig_points[idx + 2] };
			// float3 p_view = transformPoint4x3(p_orig, viewmatrix);
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += depths[collected_id[j]] * alpha * T;

			T = test_T; // 累乘T

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 背景色也是等效为一个混合项累加到式(3)上去的
	}
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessLonlatCUDA(int P/*3D Gaus数目*/, int D/*active_sh_degree_，值域{0,1,2,3}*/, int M/*dc和rest谐波之和(1+15)*/,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs, /*features_dc和features_rest*/
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix, /*Tcw*/
	const glm::vec3* cam_pos,
	const int W, int H,
	int* radii,
// int* radii_x,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// preprocessCUDA对每个3D Gaus进行预处理
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 统计Gaus的半径和覆盖tile数，预处理之后若仍为零，则其不参与渲染处理
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// 剔除不可视点；实际上只剔除了相机坐标下r<=0.2的点，不涉及proj
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_view;
	if (too_close(p_orig, viewmatrix, p_view))
		return;

	// Transform point by projecting
	// 【相机模型】计算模型点投影到screen-space的坐标
	float2 p_proj = point3ToLonlatScreen(p_view); // screen-space的坐标

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 一般没有precompute的cov3D，传入是nullptr，因此使用scale和rot计算3D协方差
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 【相机模型】计算投影到2D screen-space的协方差
	float3 cov = computeCov2DLonlat(p_orig, W, H, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// 求conic：2D协方差的逆矩阵
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 计算2D协方差的特征值->计算Gaus的半径->计算Gaus的范围框
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
// float my_radius_y = ceil(3.f * sqrt(min(lambda1, lambda2)));
// 【相机模型】TODO: 同样的圆，纬度越高，影响经度范围越大，对应范围框x方向越长
// float my_radius_x = my_radius * cov.w;
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
// int2 rect_min, rect_max;
	uint2 rect_min, rect_max;
// getRectCyclic(point_image, my_radius_x, my_radius, rect_min, rect_max, grid);
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 着色：SH转RGB
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.w; // 相机坐标 深度
	radii[idx] = my_radius; // 图像空间 半径
// radii_x[idx] = my_radius_x; // 图像空间 x方向（经度）半径
	points_xy_image[idx] = point_image; // 图像空间 像素坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; // 协方差、透明度 打包
// uint tiles_touched_x = (uint)min((int)grid.x, rect_max.x - rect_min.x);
// tiles_touched[idx] = (uint)(rect_max.y - rect_min.y) * tiles_touched_x; // 用方框粗略估计的覆盖tile数
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // 用方框粗略估计的覆盖tile数
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::renderDepth(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	// const float* orig_points,
	// const float* viewmatrix,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderDepthCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		// orig_points,
		// viewmatrix,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

// void FORWARD::renderLonlat(
// 	const dim3 grid, dim3 block,
// 	const uint2* ranges,
// 	const uint32_t* point_list,
// 	int W, int H,
// 	const float2* means2D,
// 	const float* colors,
// 	const float4* conic_opacity,
// 	float* final_T,
// 	uint32_t* n_contrib,
// 	const float* bg_color,
// 	float* out_color)
// {
// 	renderLonlatCUDA<NUM_CHANNELS> << <grid, block >> > (
// 		ranges,
// 		point_list,
// 		W, H,
// 		means2D,
// 		colors,
// 		conic_opacity,
// 		final_T,
// 		n_contrib,
// 		bg_color,
// 		out_color);
// }

void FORWARD::preprocessLonlat(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	int* radii,
// int* radii_x,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessLonlatCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		cam_pos,
		W, H,
		radii,
// radii_x,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}