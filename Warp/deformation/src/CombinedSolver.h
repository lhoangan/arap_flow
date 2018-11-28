#pragma once
#include "../../shared/SolverIteration.h"
#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "Configure.h"
#include <fstream>



static bool
PointInTriangleBarycentric(float x0, float y0, float w0,
float x1, float y1, float w1,
float x2, float y2, float w2,
float sx, float sy,
float *wt0, float *wt1, float *wt2) {
    x0 /= w0;
    y0 /= w0;
    x1 /= w1;
    y1 /= w1;
    x2 /= w2;
    y2 /= w2;

    float v0x = x2 - x0, v0y = y2 - y0;
    float v1x = x1 - x0, v1y = y1 - y0;
    float v2x = sx - x0, v2y = sy - y0;

    float area = 0.5f * (v1x * v0y - v1y * v0x);
    if (area <= 0.) {
        // backfacing
        return false;
    }

#define DOT2(a,b) ((a##x)*(b##x)+(a##y)*(b##y))
    float dot00 = DOT2(v0, v0);
    float dot01 = DOT2(v0, v1);
    float dot11 = DOT2(v1, v1);
    float denom = (dot00 * dot11 - dot01 * dot01);
    if (denom == 0)
        return false;
    float invDenom = 1.f / denom;

    float dot02 = DOT2(v0, v2);
    float dot12 = DOT2(v1, v2);

    // Compute barycentric coordinates
    float b2 = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float b1 = (dot00 * dot12 - dot01 * dot02) * invDenom;
    float b0 = 1.f - b1 - b2;

    *wt0 = b0;
    *wt1 = b1;
    *wt2 = b2;

    return (b0 > 0. && b1 > 0 && b2 > 0);
}

inline bool
PointInTriangleLK(float x0, float y0, float w0,
float x1, float y1, float w1,
float x2, float y2, float w2,
float sx, float sy,
float *wt0, float *wt1, float *wt2) {
    float X[3], Y[3];

    X[0] = x0 - sx*w0;
    X[1] = x1 - sx*w1;
    X[2] = x2 - sx*w2;

    Y[0] = y0 - sy*w0;
    Y[1] = y1 - sy*w1;
    Y[2] = y2 - sy*w2;

    float d01 = X[0] * Y[1] - Y[0] * X[1];
    float d12 = X[1] * Y[2] - Y[1] * X[2];
    float d20 = X[2] * Y[0] - Y[2] * X[0];

    if ((d01 < 0) & (d12 < 0) & (d20 < 0)) {
        //printf("Backfacing\n");
        // backfacing
        return false;
    }

    float OneOverD = 1.f / (d01 + d12 + d20);
    d01 *= OneOverD;
    d12 *= OneOverD;
    d20 *= OneOverD;

    *wt0 = d12;
    *wt1 = d20;
    *wt2 = d01;

    return (d01 >= 0 && d12 >= 0 && d20 >= 0);
}

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(CombinedSolverParameters params): m_dims() {
        m_combinedSolverParameters = params;
    }
    CombinedSolver( unsigned int width, unsigned int height,
            CombinedSolverParameters params) {

        m_dims = { width, height };
        m_combinedSolverParameters = params;

        addOptSolvers(m_dims, "image_warping.t", m_combinedSolverParameters.optDoublePrecision);
    }

    CombinedSolver(const ColorImageR8G8B8& imageColor,
            const ColorImageR8G8B8& imageMask,
            std::vector<std::vector<int>> constraints,
            CombinedSolverParameters params) {

        m_orgRGB = imageColor;
        m_orgMask = imageMask;
        m_combinedSolverParameters = params;
        m_constraints = constraints;

        m_urshape           = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_warpField         = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_warpAngles        = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);
        m_constraintImage   = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_mask              = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);

        resetGPU();

        m_dims = {  (unsigned int) m_orgRGB.getWidth(),
                    (unsigned int) m_orgRGB.getHeight() };

        addOptSolvers(m_dims, "image_warping.t", m_combinedSolverParameters.optDoublePrecision);
    }

 
    void addImage(const ColorImageR8G8B8& imageColor,
            const ColorImageR8G8B8& imageMask,
            std::vector<std::vector<int>> constraints,
            CombinedSolverParameters params) {

        m_orgRGB = imageColor;
        m_orgMask = imageMask;
        m_combinedSolverParameters = params;
        m_constraints = constraints;

        if (m_dims[0] != m_orgRGB.getWidth() || m_dims[1] != m_orgRGB.getHeight()) {

            printf("Warning: Input image has different size to the prebuilt plan.\n"
                    "To avoid re-building plan and save time, put images of the "
                    "same size in the same list.\nStarting to re-build plan...\n");

            if (m_solverInfo.size() > 0)
                m_solverInfo.clear();
            m_dims = {  (unsigned int) m_orgRGB.getWidth(),
                        (unsigned int) m_orgRGB.getHeight() };
            addOptSolvers(m_dims, "image_warping.t", m_combinedSolverParameters.optDoublePrecision);
        }

        m_urshape           = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_warpField         = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_warpAngles        = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);
        m_constraintImage   = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::GPU, true);
        m_mask              = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);

        resetGPU();

    }

    virtual void combinedSolveInit() override {
        float weightFit = 100.0f;
        float weightReg = 0.01f;

        m_weightFitSqrt = sqrtf(weightFit);
        m_weightRegSqrt = sqrtf(weightReg);

        m_problemParams.set("Offset",       m_warpField);
        m_problemParams.set("Angle",        m_warpAngles);
        m_problemParams.set("UrShape",      m_urshape);
        m_problemParams.set("Constraints",  m_constraintImage);
        m_problemParams.set("Mask",         m_mask);
        m_problemParams.set("w_fitSqrt",    &m_weightFitSqrt);
        m_problemParams.set("w_regSqrt",    &m_weightRegSqrt);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preSingleSolve() override {
        resetGPU();
    }

    virtual void postSingleSolve() override {
        copyResultToCPU();
    }

    virtual void preNonlinearSolve(int i) override {
        setConstraintImage((float)(i+1) / (float)m_combinedSolverParameters.numIter);
    }

    virtual void postNonlinearSolve(int) override{}

    virtual void combinedSolveFinalize() override {}

    void resetGPU() {
        std::vector<float2> h_urshape(m_dims[0] * m_dims[1]);
        std::vector<float>  h_mask(m_dims[0] * m_dims[1]);
        for (unsigned int y = 0; y < m_dims[1]; y++)
            for (unsigned int x = 0; x < m_dims[0]; x++) {
                h_urshape[y*m_dims[0] + x] = { (float)x, (float)y };
                h_mask[y*m_dims[0] + x] = (float)m_orgMask(x, y).x;
            }

        setConstraintImage(1.0f);
        m_urshape->update(h_urshape);
        m_warpField->update(h_urshape);
        m_mask->update(h_mask);
        cudaSafeCall(cudaMemset(m_warpAngles->data(), 0, sizeof(float)*m_dims[0]*m_dims[1]));
    }

    void setConstraintImage(float alpha) {
        std::vector<float2> h_constraints(m_dims[0]*m_dims[1]);
        for (unsigned int y = 0; y < m_dims[1]; y++)
            for (unsigned int x = 0; x < m_dims[0]; x++) {
                h_constraints[y*m_dims[0] + x] = { -1.0f, -1.0f };
            }

        for (unsigned int k = 0; k < m_constraints.size(); k++) {
            int x = m_constraints[k][0];
            int y = m_constraints[k][1];

            if (m_orgMask(x, y).x == 0) {
                float newX = (1.0f - alpha)*(float)x + alpha*(float)m_constraints[k][2];
                float newY = (1.0f - alpha)*(float)y + alpha*(float)m_constraints[k][3];

                h_constraints[y*m_dims[0] + x] = { newX, newY };
            }
        }
        m_constraintImage->update(h_constraints);
    }

    vec2f toVec2(float2 p) {
        return vec2f(p.x, p.y);
    }

    void rasterizeTriangle(ColorImageR8G8B8 &result,
        float2 p0, float2 p1, float2 p2,
        vec3f c0, vec3f c1, vec3f c2,
        bool ismask=false) {

        vec2f t0 = toVec2(p0);
        vec2f t1 = toVec2(p1);
        vec2f t2 = toVec2(p2);


        int W = m_warpedRGB.getWidth();
        int H = m_warpedRGB.getHeight();

        vec2f minBound = math::floor(math::min(t0, math::min(t1, t2)));
        vec2f maxBound = math::ceil(math::max(t0, math::max(t1, t2)));
        for (int x = (int)minBound.x; x <= maxBound.x; ++x)
            for (int y = (int)minBound.y; y <= maxBound.y; ++y)
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    float b0, b1, b2;
                    if (PointInTriangleLK(t0.x, t0.y, 1.0f,
                        t1.x, t1.y, 1.0f,
                        t2.x, t2.y, 1.0f, (float)x, (float)y, &b0, &b1, &b2)) {

                        vec3f val = c0*b0 + c1*b1 + c2*b2;
                        if (ismask)
                            result(x, y) = int(val > 0) * 255;
                        else
                            result(x, y) = val;
                    }
                }
    }

    void copyResultToCPU() {
        m_warpedRGB = ColorImageR8G8B8((unsigned int)(m_dims[0]), (unsigned int)(m_dims[1]));
        m_warpedRGB.setPixels(vec3uc(0, 0, 0));

        m_warpedMask = ColorImageR8G8B8((unsigned int)(m_dims[0]), (unsigned int)(m_dims[1]));
        m_warpedMask.setPixels(vec3uc(0, 0, 0));

        std::vector<float2> h_warpField(m_dims[0]*m_dims[1]);
        m_warpField->copyTo(h_warpField);

        // Rasterize the results
        unsigned int c = 3;
        for (unsigned int y = 0; y < m_dims[1]; y++)
            for (unsigned int x = 0; x < m_dims[0]; x++)
                if (y + 1 < m_dims[1] && x + 1 < m_dims[0])
                    if (m_orgMask(x, y).x == 0)
                    {
                        float2 pos00 = h_warpField[y*m_dims[0] + x];
                        float2 pos01 = h_warpField[y*m_dims[0] + (x + 1)];
                        float2 pos10 = h_warpField[(y + 1)*m_dims[0] + x];
                        float2 pos11 = h_warpField[(y + 1)*m_dims[0] + (x + 1)];

                        vec3f v00 = m_orgRGB(x, y);
                        vec3f v01 = m_orgRGB(x + 1, y);
                        vec3f v10 = m_orgRGB(x, y + 1);
                        vec3f v11 = m_orgRGB(x + 1, y + 1);

                        bool valid00 = (m_orgMask(x, y).x == 0);
                        bool valid01 = (m_orgMask(x, y + 1).x == 0);
                        bool valid10 = (m_orgMask(x + 1, y).x == 0);
                        bool valid11 = (m_orgMask(x + 1, y + 1).x == 0);

                        if (valid00 && valid01 && valid10 && valid11) {
                            rasterizeTriangle(m_warpedRGB,
                                    pos00, pos01, pos10,
                                    v00, v01, v10);
                            rasterizeTriangle(m_warpedRGB,
                                    pos10, pos01, pos11,
                                    v10, v01, v11);
                        }

                        v00 = 1-m_orgMask(x, y).x;
                        v01 = 1-m_orgMask(x + 1, y).x;
                        v10 = 1-m_orgMask(x, y + 1).x;
                        v11 = 1-m_orgMask(x + 1, y + 1).x;

                        valid00 = (m_orgMask(x, y).x == 0);
                        valid01 = (m_orgMask(x, y + 1).x == 0);
                        valid10 = (m_orgMask(x + 1, y).x == 0);
                        valid11 = (m_orgMask(x + 1, y + 1).x == 0);

                        if (valid00 && valid01 && valid10 && valid11) {
                            rasterizeTriangle(m_warpedMask,
                                    pos00, pos01, pos10,
                                    v00, v01, v10,
                                    true);
                            rasterizeTriangle(m_warpedMask,
                                    pos10, pos01, pos11,
                                    v10, v01, v11,
                                    true);
                        }
                    }
    }

    ColorImageR8G8B8* result() {
        return &m_warpedRGB;
    }

    ColorImageR8G8B8* resultSeg() {
        return &m_warpedMask;
    }

    std::vector<float>* warpField() {
        std::vector<float>* warpField = new std::vector<float>(m_dims[0]*m_dims[1]*2);
        m_warpField->copyTo(*warpField);
        return warpField;
    }

private:
    ColorImageR8G8B8 m_orgRGB;
    ColorImageR8G8B8 m_orgMask;

    float m_weightFitSqrt;
    float m_weightRegSqrt;

    ColorImageR8G8B8 m_warpedRGB;
    ColorImageR8G8B8 m_warpedMask;

    std::vector<unsigned int> m_dims;

    std::shared_ptr<OptImage> m_urshape;
    std::shared_ptr<OptImage> m_warpField;
    std::shared_ptr<OptImage> m_constraintImage;
    std::shared_ptr<OptImage> m_warpAngles;
    std::shared_ptr<OptImage> m_mask;


    std::vector<std::vector<int>> m_constraints;

};
