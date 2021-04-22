#include <cmath>
//#include <omp.h>
#include <algorithm>
#include <iostream>
#define PI 3.14159265358979323846

inline void lookAt(double* camera_param, double* extrinsic) {
    double camera[3] = { camera_param[0], camera_param[1], camera_param[2] };
    double lookat[3] = { camera_param[3], camera_param[4], camera_param[5] };
    double up[3] = { camera_param[6], camera_param[7], camera_param[8] };

    double tmp1[3] = { camera[0] - lookat[0],camera[1] - lookat[1],camera[2] - lookat[2] };
    double len = sqrt(tmp1[0] * tmp1[0] + tmp1[1] * tmp1[1] + tmp1[2] * tmp1[2]);
    double zp[3] = { tmp1[0] / len, tmp1[1] / len, tmp1[2] / len };
    double tmp2[3] = { zp[1] * up[2] - zp[2] * up[1], zp[2] * up[0] - zp[0] * up[2], zp[0] * up[1] - zp[1] * up[0] };
    len = sqrt(tmp2[0] * tmp2[0] + tmp2[1] * tmp2[1] + tmp2[2] * tmp2[2]);
    double xp[3] = { tmp2[0] / len, tmp2[1] / len, tmp2[2] / len };
    double tmp3[3] = { zp[1] * xp[2] - zp[2] * xp[1], zp[2] * xp[0] - zp[0] * xp[2], zp[0] * xp[1] - zp[1] * xp[0] };
    len = sqrt(tmp3[0] * tmp3[0] + tmp3[1] * tmp3[1] + tmp3[2] * tmp3[2]);
    double yp[3] = { tmp3[0] / len, tmp3[1] / len, tmp3[2] / len };

    extrinsic[0] = xp[0];
    extrinsic[1] = xp[1];
    extrinsic[2] = xp[2];
    extrinsic[3] = -xp[0] * camera[0] - xp[1] * camera[1] - xp[2] * camera[2];
    extrinsic[4] = yp[0];
    extrinsic[5] = yp[1];
    extrinsic[6] = yp[2];
    extrinsic[7] = -yp[0] * camera[0] - yp[1] * camera[1] - yp[2] * camera[2];
    extrinsic[8] = zp[0];
    extrinsic[9] = zp[1];
    extrinsic[10] = zp[2];
    extrinsic[11] = -zp[0] * camera[0] - zp[1] * camera[1] - zp[2] * camera[2];
    extrinsic[12] = 0;
    extrinsic[13] = 0;
    extrinsic[14] = 0;
    extrinsic[15] = 1;
}

extern "C" void calculate_flow(double* camera_params, double fovRadian, float* flow_array) {
    int res = 1024;
    double f = 0.5 * res / std::tan(fovRadian / 2.0);
    double c = 0.5 * res;
    double intrinsic[12] = { f,0,c,0,0,f,c,0,0,0,1,0 };

    int fm_list[] = {0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140};

    //omp_set_num_threads(8);
    //#pragma omp parallel for
    double extrinsic_A[16];
    lookAt(camera_params, extrinsic_A);
    /*for (int i = 0; i < 16; ++i) {
        std::cout << extrinsic_A[i] << " ";
    }
    std::cout << std::endl;*/
    for (int k = 0; k < 29; ++k) {
        double extrinsic_B[16];
        lookAt(camera_params + fm_list[k] * 9, extrinsic_B);
        for (int i = 0; i < res; ++i) {
            for (int j = 0; j < res; ++j) {
                double coef[4] = { f * extrinsic_A[0] + (c - j) * extrinsic_A[8], f * extrinsic_A[1] + (c - j) * extrinsic_A[9], f * extrinsic_A[4] + (c - i) * extrinsic_A[8], f * extrinsic_A[5] + (c - i) * extrinsic_A[9] };
                double ordi[2] = { (j - c) * extrinsic_A[11] - f * extrinsic_A[3], (i - c) * extrinsic_A[11] - f * extrinsic_A[7] };
                double v = coef[0] * coef[3] - coef[1] * coef[2];
                double XYZ1[4] = { (ordi[0] * coef[3] - ordi[1] * coef[1]) / v, (ordi[1] * coef[0] - ordi[0] * coef[2]) / v, 0 , 1 };
                /*for (int l = 0; l < 4; ++l) {
                    std::cout << XYZ1[l] << " ";
                }
                std::cout << std::endl;*/
                double camspace[4] = { extrinsic_B[0] * XYZ1[0] + extrinsic_B[1] * XYZ1[1] + extrinsic_B[2] * XYZ1[2] + extrinsic_B[3] * XYZ1[3], extrinsic_B[4] * XYZ1[0] + extrinsic_B[5] * XYZ1[1] + extrinsic_B[6] * XYZ1[2] + extrinsic_B[7] * XYZ1[3], extrinsic_B[8] * XYZ1[0] + extrinsic_B[9] * XYZ1[1] + extrinsic_B[10] * XYZ1[2] + extrinsic_B[11] * XYZ1[3], extrinsic_B[12] * XYZ1[0] + extrinsic_B[13] * XYZ1[1] + extrinsic_B[14] * XYZ1[2] + extrinsic_B[15] * XYZ1[3] };
                double uv[3] = { intrinsic[0] * camspace[0] + intrinsic[1] * camspace[1] + intrinsic[2] * camspace[2] + intrinsic[3] * camspace[3], intrinsic[4] * camspace[0] + intrinsic[5] * camspace[1] + intrinsic[6] * camspace[2] + intrinsic[7] * camspace[3], intrinsic[8] * camspace[0] + intrinsic[9] * camspace[1] + intrinsic[10] * camspace[2] + intrinsic[11] * camspace[3] };

                /*for (int l = 0; l < 3; ++l) {
                    std::cout << uv[l] << " ";
                }
                std::cout << std::endl;*/
                flow_array[k * res * res * 2 + i * res * 2 + j * 2] = j - uv[0] / uv[2];
                flow_array[k * res * res * 2 + i * res * 2 + j * 2 + 1] = i - uv[1] / uv[2];
                //std::cin >> i;
            }
        }
    }
}
