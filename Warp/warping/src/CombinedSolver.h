#pragma once
#include <cuda_runtime.h>

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

class CombinedSolver {
public:
    CombinedSolver(
            const ColorImageR32G32B32& imageColor,
            const ColorImageR32& imageMask,
            std::vector<float>* flow){

        ColorImageR32G32B32 m_imageColor;
        ColorImageR32 m_imageMask;

        ColorImageR32G32B32 m_resultColor;
        ColorImageR32G32B32 m_resultSeg;

        std::vector<unsigned int> m_dims;

        std::vector<float2>* h_warpField; // new std::vector<float>(m_imageColor.getWidth()*m_imageColor.getHeight()*2);

		m_imageColor = imageColor;
		m_imageMask = imageMask;

        m_dims = { m_imageColor.getWidth(), m_imageColor.getHeight() };
        h_warpField = new std::vector<float2>(m_imageColor.getWidth()*m_imageColor.getHeight());

        int n = m_imageColor.getWidth() * 2;
		for (unsigned int y = 0; y < m_imageColor.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_imageColor.getWidth(); x++)
			{
                float fx = (*flow)[y*n + 2*x];
                float fy = (*flow)[y*n + 2*x + 1];
                (*h_warpField)[y*m_imageColor.getWidth() + x] = { (float)x + fx, (float)y + fy  };
			}
		}

        m_resultColor = ColorImageR32G32B32((unsigned int)(m_imageColor.getWidth()), (unsigned int)(m_imageColor.getHeight()));
		m_resultColor.setPixels(vec3f(0.0f, 0.0f, 0.0f));

        m_resultSeg = ColorImageR32G32B32((unsigned int)(m_imageColor.getWidth()), (unsigned int)(m_imageColor.getHeight()));
		m_resultSeg.setPixels(vec3f(0.0f, 0.0f, 0.0f));


        // Rasterize the results
		unsigned int c = 3;
		for (unsigned int y = 0; y < m_imageColor.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_imageColor.getWidth(); x++)
			{
				if (y + 1 < m_imageColor.getHeight() && x + 1 < m_imageColor.getWidth())
				{
					if (m_imageMask(x, y) == 0)
					{
                        float2 pos00 = (*h_warpField)[y*m_imageColor.getWidth() + x];
                        float2 pos01 = (*h_warpField)[y*m_imageColor.getWidth() + (x + 1)];
                        float2 pos10 = (*h_warpField)[(y + 1)*m_imageColor.getWidth() + x];
                        float2 pos11 = (*h_warpField)[(y + 1)*m_imageColor.getWidth() + (x + 1)];

						vec3f v00 = m_imageColor(x, y);
						vec3f v01 = m_imageColor(x + 1, y);
						vec3f v10 = m_imageColor(x, y + 1);
						vec3f v11 = m_imageColor(x + 1, y + 1);

						bool valid00 = (m_imageMask(x, y) == 0);
						bool valid01 = (m_imageMask(x, y + 1) == 0);
						bool valid10 = (m_imageMask(x + 1, y) == 0);
						bool valid11 = (m_imageMask(x + 1, y + 1) == 0);

						if (valid00 && valid01 && valid10 && valid11) {
							rasterizeTriangle(pos00, pos01, pos10,
								v00, v01, v10, m_resultColor
                                );
							rasterizeTriangle(pos10, pos01, pos11,
								v10, v01, v11, m_resultColor);
                        }

						v00 = 1-m_imageMask(x, y);
						v01 = 1-m_imageMask(x + 1, y);
						v10 = 1-m_imageMask(x, y + 1);
						v11 = 1-m_imageMask(x + 1, y + 1);

						valid00 = (m_imageMask(x, y) == 0);
						valid01 = (m_imageMask(x, y + 1) == 0);
						valid10 = (m_imageMask(x + 1, y) == 0);
						valid11 = (m_imageMask(x + 1, y + 1) == 0);

						if (valid00 && valid01 && valid10 && valid11) {
							rasterizeTriangle(pos00, pos01, pos10,
								v00, v01, v10, m_resultSeg
                                );
							rasterizeTriangle(pos10, pos01, pos11,
								v10, v01, v11, m_resultSeg);
						}
					}
				}
			}
		}
	}
	
    vec2f toVec2(float2 p) {
		return vec2f(p.x, p.y);
	}

    void rasterizeTriangle(float2 p0, float2 p1, float2 p2, vec3f c0, vec3f c1, vec3f c2, ColorImageR32G32B32 &resultColor) {
		vec2f t0 = toVec2(p0);
		vec2f t1 = toVec2(p1);
		vec2f t2 = toVec2(p2);


		int W = resultColor.getWidth();
		int H = resultColor.getHeight();

		vec2f minBound = math::floor(math::min(t0, math::min(t1, t2)));
		vec2f maxBound = math::ceil(math::max(t0, math::max(t1, t2)));
		for (int x = (int)minBound.x; x <= maxBound.x; ++x) {
			for (int y = (int)minBound.y; y <= maxBound.y; ++y) {
				if (x >= 0 && x < W && y >= 0 && y < H) {
					float b0, b1, b2;
					if (PointInTriangleLK(t0.x, t0.y, 1.0f,
						t1.x, t1.y, 1.0f,
						t2.x, t2.y, 1.0f, (float)x, (float)y, &b0, &b1, &b2)) {
						vec3f color = c0*b0 + c1*b1 + c2*b2;
						resultColor(x, y) = color;
					}

				}
			}
		}
    }

    ColorImageR32G32B32* result() {
        return &m_resultColor;
    }

    ColorImageR32G32B32* resultSeg() {
        return &m_resultSeg;
    }

private:
};
