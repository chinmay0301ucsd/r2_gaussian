// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//TODO: Clean-up shared-memory logic, too complicated, could be simplified.

import utils;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared Splat_3D_AlphaBlend collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

[Differentiable]
float5 update_pixel_state(float5 pixel_state_t_nm1, float5 gauss_rgbad_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb() + gauss_rgbad_t_n.rgb() * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (1 - gauss_rgbad_t_n.a);
    float depth_t_n = pixel_state_t_nm1.d + gauss_rgbad_t_n.d * pixel_state_t_nm1.a;
    return float5(color_t_n.x, color_t_n.y, color_t_n.z, transmittance_t_n, depth_t_n);
}

float5 undo_pixel_state(float5 pixel_state_t_n, float5 gauss_rgbad_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.a / (1 - gauss_rgbad_t_n.a);
    float3 color_t_nm1 = pixel_state_t_n.rgb() - gauss_rgbad_t_n.rgb() * transmittance_t_nm1;
    float depth_t_nm1 = pixel_state_t_n.d - gauss_rgbad_t_n.d * transmittance_t_nm1;
    return float5(color_t_nm1.x, color_t_nm1.y, color_t_nm1.z, transmittance_t_nm1, depth_t_nm1);
}

[BackwardDerivative(bwd_alpha_blend_volr)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
float5 alpha_blend_volr(TensorView<int32_t> sorted_gauss_idx,
                   DiffTensorView xyz3d_vs,
                   DiffTensorView inv_cov3d_vs,
                   DiffTensorView opacity,
                   DiffTensorView opacity_volr,
                   DiffTensorView rgb,
                   DiffTensorView final_pixel_state,
                   TensorView<float> final_pixel_depth,
                   TensorView<int32_t> n_contributors,
                   TensorView<float> grad_abs,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   no_diff float fx,
                   no_diff float fy,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    float3 dir = {(center_pix_coord.x - W/2.0f + 0.5f)/fx, (center_pix_coord.y - H/2.0f + 0.5f)/fy, 1.0f };
    dir = normalize(dir);
    float5 curr_pixel_state = float5(0.f, 0.f, 0.f, 1.f, 0.f);
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_splat3d_alphablend(coll_id, xyz3d_vs, inv_cov3d_vs, opacity_volr, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                Splat_3D_AlphaBlend g = collected_splats[j];
                float5 gauss_rgbad = evaluate_splat_volr(g, dir, H, W, fx, fy);
                
                // TODO: Uncomment this!!
                //Skip Splats that have a tiny contribution.
                if (gauss_rgbad.a < 1.0f / 255.0f)
                    continue;

                float5 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgbad);
                if (new_pixel_state.a < 0.0001f) {
                    // This Splat never registred so we subtract it before we break.
                    local_n_contrib--;
                    thread_active = false;
                    break;
                }
                curr_pixel_state = new_pixel_state;
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return curr_pixel_state;
}

void bwd_alpha_blend_volr(TensorView<int32_t> sorted_gauss_idx,
                     DiffTensorView xyz3d_vs,
                     DiffTensorView inv_cov3d_vs,
                     DiffTensorView opacity,
                     DiffTensorView opacity_volr,
                     DiffTensorView rgb,
                     DiffTensorView final_pixel_state,
                     TensorView<float> final_pixel_depth,
                     TensorView<int32_t> n_contributors,
                     TensorView<float> grad_abs,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     no_diff float fx,
                     no_diff float fy,
                     uint32_t H,
                     uint32_t W,
                     float5 d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float5 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = float5(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
                                     final_pixel_depth[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    float2 center_pix_coord = pix_coord;
    float3 dir = {(center_pix_coord.x - W/2.0f + 0.5f)/fx, (center_pix_coord.y - H/2.0f + 0.5f)/fy, 1.0f };
    dir = normalize(dir);
    DifferentialPair<float3> dp_dir = diffPair(dir);


    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat3d_alphablend(coll_id, xyz3d_vs, inv_cov3d_vs, opacity_volr, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd)
                    continue;
                uint32_t g_idx = collected_idx[j];
                Splat_3D_AlphaBlend g = collected_splats[j];

                float5 gauss_rgbad = evaluate_splat_volr(g, dir, H, W, fx, fy);

                if (gauss_rgbad.a < 1.0f / 255.0f)
                    continue;

                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgbad);

                // Back-prop automatically through blending and gaussian evaluation.
                DifferentialPair<Splat_3D_AlphaBlend> dp_g = diffPair(g);
                DifferentialPair<float5> dp_gauss_rgbad = diffPair(gauss_rgbad);
                DifferentialPair<float5> dp_current_pixel_state = diffPair(current_pixel_state);
                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgbad, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat_volr)(dp_g, dp_dir, H, W, fx, fy, dp_gauss_rgbad.d);
                bwd_diff(load_splat3d_alphablend)(g_idx, xyz3d_vs, inv_cov3d_vs, opacity_volr, rgb, dp_g.d);
                // Accumulate absolute gradients for xyz
                // float3 abs_xyz_grad = abs(dp_g.d.xyz3d_vs);
                // float3 total_abs_xyz_grad = WaveActiveSum(abs_xyz_grad);
                // if (WaveIsFirstLane()) {
                //     float oldVal;
                //     grad_abs.InterlockedAdd(uint2(g_idx, 0), total_abs_xyz_grad.x, oldVal);
                //     grad_abs.InterlockedAdd(uint2(g_idx, 1), total_abs_xyz_grad.y, oldVal);
                //     grad_abs.InterlockedAdd(uint2(g_idx, 2), total_abs_xyz_grad.z, oldVal);
                // }
                float oldvalx, oldvaly, oldvalz;
                grad_abs.InterlockedAdd(uint2(g_idx, 0), abs(dp_g.d.xyz3d_vs.x), oldvalx);
                grad_abs.InterlockedAdd(uint2(g_idx, 1), abs(dp_g.d.xyz3d_vs.y), oldvaly);
                grad_abs.InterlockedAdd(uint2(g_idx, 2), abs(dp_g.d.xyz3d_vs.z), oldvalz);
            }
        }
        splats_left_to_process -= block_size;
    }
}


[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_volr_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz3d_vs,
                 DiffTensorView inv_cov3d_vs,
                 DiffTensorView opacity,
                 DiffTensorView opacity_volr,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 TensorView<float> output_depth,
                 TensorView<int32_t> n_contributors,
                 TensorView<float> grad_abs,
                 no_diff float fx,
                 no_diff float fy,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float5 pixel_state = alpha_blend_volr(sorted_gauss_idx,
                                     xyz3d_vs,
                                     inv_cov3d_vs,
                                     opacity,
                                     opacity_volr,
                                     rgb,
                                     output_img,
                                     output_depth,
                                     n_contributors,
                                     grad_abs,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     fx,
                                     fy,
                                     output_img.size(0),
                                     output_img.size(1));

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_state.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_state.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_state.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
      output_depth[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = pixel_state.d;
    }
}

// This function is essentially a visualizer, that returns :
// 1. The first Gaussian idx with alpha > 0.5, and the value of alpha along the ray.
// 2. value of alpha of the Gaussian. 
// 3. Value of sigma_d for this gaussian. 
float5 vis_volr(TensorView<int32_t> sorted_gauss_idx,
                   DiffTensorView xyz3d_vs,
                   DiffTensorView inv_cov3d_vs,
                   DiffTensorView opacity,
                   DiffTensorView opacity_volr,
                   DiffTensorView rgb,
                   DiffTensorView final_pixel_state,
                   TensorView<int32_t> n_contributors,
                   TensorView<float> grad_abs,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   no_diff float fx,
                   no_diff float fy,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    float3 dir = {(center_pix_coord.x - W/2.0f + 0.5f)/fx, (center_pix_coord.y - H/2.0f + 0.5f)/fy, 1.0f };
    dir = normalize(dir);
    float5 curr_pixel_state = float5(0.f, 0.f, 0.f, 1.f, 0.f);
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat3d_alphablend(coll_id, xyz3d_vs, inv_cov3d_vs, opacity_volr, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                Splat_3D_AlphaBlend g = collected_splats[j];
                float g_idx = float(collected_idx[j]);
                float5 gauss_rgbad = evaluate_splat_volr(g, dir, H, W, fx, fy);
                float denominator = max(eps, dot(dir, mul(g.inv_cov3d_vs, dir)));
                float sigma_d = sqrt(1.0f / (denominator + eps));
                // TODO: Uncomment this!!
                //Skip Splats that have a tiny contribution.
                if (gauss_rgbad.a > 0.5f) {
                    curr_pixel_state = float5(g_idx, gauss_rgbad.a, gauss_rgbad.d, 0.f, 0.f);
                    if (is_inside)
                        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;
                    return curr_pixel_state;
                }
            }
        }
        splats_left_to_process -= block_size;
    }
     if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return curr_pixel_state;
}



[AutoPyBindCUDA]
[CUDAKernel]
void visualize_volr_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz3d_vs,
                 DiffTensorView inv_cov3d_vs,
                 DiffTensorView opacity,
                 DiffTensorView opacity_volr,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 TensorView<float> output_depth,
                 TensorView<int32_t> n_contributors,
                 TensorView<float> grad_abs,
                 no_diff float fx,
                 no_diff float fy,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float5 pixel_state = vis_volr(sorted_gauss_idx,
                                     xyz3d_vs,
                                     inv_cov3d_vs,
                                     opacity,
                                     opacity_volr,
                                     rgb,
                                     output_img,
                                     n_contributors,
                                     grad_abs,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     fx,
                                     fy,
                                     output_img.size(0),
                                     output_img.size(1));

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_state.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_state.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_state.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
      output_depth[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = pixel_state.d;
    }
}