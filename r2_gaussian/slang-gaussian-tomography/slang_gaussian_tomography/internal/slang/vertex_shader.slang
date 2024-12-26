import utils;
import spherical_harmonics;

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float radius,
    uint grid_height,
    uint grid_width,
    uint tile_height,
    uint tile_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = int32_t(floor(clip((ndc_xy.x - radius) / tile_width, 0, grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip((ndc_xy.y - radius) / tile_height, 0, grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip((ndc_xy.x + radius) / tile_width, 0, grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip((ndc_xy.y + radius) / tile_height, 0, grid_height)));

    //TODO: Clean this up, unintuivie math and code
    //rect_tile_space.max_x = clip((ndc_xy.x + radius + tile_width - 1)/tile_width, 0.0, grid_width);
    //rect_tile_space.max_y = clip((ndc_xy.y + radius + tile_height - 1)/tile_height, 0.0, grid_height);

    return rect_tile_space;
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void vertex_shader(DiffTensorView xyz_ws,
                   DiffTensorView rotations,
                   DiffTensorView scales,
                   DiffTensorView scale3d_factor,
                   TensorView<float> world_view_transform,
                   TensorView<float> proj_mat,
                   TensorView<float> cam_pos,
                   TensorView<int32_t> out_tiles_touched,
                   TensorView<int32_t> out_rect_tile_space,
                   TensorView<int32_t> out_radii,
                   DiffTensorView out_xyz_vs,
                   DiffTensorView out_xyz3d_cam,
                   DiffTensorView out_inv_cov_vs,
                   DiffTensorView out_inv_cov3d_vs,
                   DiffTensorView out_rgb,
                   no_diff float fovy,
                   no_diff float fovx,
                   uint image_height,
                   uint image_width,
                   uint grid_height,
                   uint grid_width,
                   uint tile_height,
                   uint tile_width)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= xyz_ws.size(0))
        return;

    Camera cam = no_diff load_camera(world_view_transform, proj_mat, cam_pos, fovy, fovx, image_height, image_width);
    // TODO: Add the opacity based scaling factor here to gauss, for calculating correct tile sizes
    Gaussian_3D gauss = load_gaussian(g_idx, xyz_ws, rotations, scales);
    // Apply scale3d_factor to the Gaussian scales, to incorporate effect of opacity in computing tile size
    float scalef = read_t1_float(g_idx, scale3d_factor);
    if (scalef <= 0.0) {
        return;
    }
    // TODO: Add code here to compute scaling factor in alpha volr for each vertex.
    gauss.scales = gauss.scales * scalef;
    Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam);
    if (splat.xyz_vs.z <= 0.2) {
        return;
    }
    float det = compute_det(splat.cov_vs);
    
    if (det == 0.0f)
        return;
    // Computing 3D covariance as part of the vertex shader and exporting it for future use. 
    // Removing the opacity based scaling factor before these steps, so that actual covariance isn't affected. 
    gauss.scales = gauss.scales / scalef;
    float3x3 cov3d_ws = get_covariance_from_quat_scales(gauss.rotations, gauss.scales);
    float3x3 inv_cov3d_ws = get_inverse_covariance_from_quat_scales(gauss.rotations, gauss.scales);
    float3x3 cov3d_vs = covariance_3dw_to_3dv(cam, cov3d_ws);

    // float invdet3d = 1.0f / compute_det3x3(cov3d_vs);
    // if (isnan(invdet3d)) {
    //     return;
    // }

    // printf("invdet3d: %f det: %f\n", 1.0f/det3d, det);
    float3x3 inv_cov3d_vs = covariance_3dw_to_3dv(cam, inv_cov3d_ws);
    // float3x3 inv_cov3d_vs = inverse3x3(cov3d_vs);
    
    float3 xyz3d_cam = geom_transform_points2(gauss.xyz_ws, cam.world_view_transform);
    float radius = splat_radius(splat.cov_vs, det);


    float2 pixelspace_xy = { ndc2pix(splat.xyz_vs.x, image_width), ndc2pix(splat.xyz_vs.y, image_height) };
    rectangle rect_tile_space = get_rectangle_tile_space(pixelspace_xy,
                                                         radius, grid_height, grid_width, tile_height, tile_width);
    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }

    float2x2 g_inv_cov_vs = float2x2(splat.cov_vs[1][1], -splat.cov_vs[0][1], -splat.cov_vs[1][0], splat.cov_vs[0][0]) / det;

    out_radii[g_idx] = (uint32_t)radius;
    out_tiles_touched[g_idx] = n_tiles;
    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    out_xyz_vs.storeOnce(uint2(g_idx, 0), splat.xyz_vs.x);
    out_xyz_vs.storeOnce(uint2(g_idx, 1), splat.xyz_vs.y);
    out_xyz_vs.storeOnce(uint2(g_idx, 2), splat.xyz_vs.z);
    out_xyz3d_cam.storeOnce(uint2(g_idx, 0), xyz3d_cam.x);
    out_xyz3d_cam.storeOnce(uint2(g_idx, 1), xyz3d_cam.y);
    out_xyz3d_cam.storeOnce(uint2(g_idx, 2), xyz3d_cam.z);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 0), g_inv_cov_vs[0][0]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 1), g_inv_cov_vs[0][1]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 0), g_inv_cov_vs[1][0]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 1), g_inv_cov_vs[1][1]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 0, 0), inv_cov3d_vs[0][0]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 0, 1), inv_cov3d_vs[0][1]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 0, 2), inv_cov3d_vs[0][2]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 1, 0), inv_cov3d_vs[1][0]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 1, 1), inv_cov3d_vs[1][1]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 1, 2), inv_cov3d_vs[1][2]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 2, 0), inv_cov3d_vs[2][0]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 2, 1), inv_cov3d_vs[2][1]);
    out_inv_cov3d_vs.storeOnce(uint3(g_idx, 2, 2), inv_cov3d_vs[2][2]);
    out_rgb.storeOnce(uint2(g_idx, 0), splat.rgb.r);
    out_rgb.storeOnce(uint2(g_idx, 1), splat.rgb.g);
    out_rgb.storeOnce(uint2(g_idx, 2), splat.rgb.b);
}
