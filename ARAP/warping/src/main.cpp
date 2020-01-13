#include "main.h"

#include <cuda_runtime.h>

void usage() {
#define p(msg)  printf(msg "\n");
    p("Usage:");
    p("./warp_image image mask flow warped_image warped_mask");
    p("Mask and warp image using the provided optical flow field.")
    p("\timage: path to image with png extension")
    p("\tmask: path to mask image with png extension, 0 for object, 1 for background")
    p("\tflo: path to optical flow image with flo extension")
    p("\twarped_image: path to output warped image (.png), all intermediate directories must exist")
    p("\twarped_mask: path to output warped mask (.png), all intermediate directories must exist")
}


#define DOT2(a,b) ((a##x)*(b##x)+(a##y)*(b##y))

static 
bool PointInTriangleBarycentric(
        float x0, float y0, float w0,
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

inline 
bool PointInTriangleLK (
        float x0, float y0, float w0,
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


    int W = result.getWidth();
    int H = result.getHeight();

    vec2f minBound = math::floor(math::min(t0, math::min(t1, t2)));
    vec2f maxBound = math::ceil(math::max(t0, math::max(t1, t2)));
    for (int x = (int)minBound.x; x <= maxBound.x; ++x) {
        for (int y = (int)minBound.y; y <= maxBound.y; ++y) {
            if (x >= 0 && x < W && y >= 0 && y < H) {
                float b0, b1, b2;
                if (PointInTriangleLK(t0.x, t0.y, 1.0f,
                    t1.x, t1.y, 1.0f,
                    t2.x, t2.y, 1.0f, (float)x, (float)y, &b0, &b1, &b2)) {
                    vec3uc val = c0*b0 + c1*b1 + c2*b2;
                    if (ismask)
                        result(x, y) = int(val > 0) * 255;
                    else
                        result(x, y) = val;
                }

            }
        }
    }
}


void Warp (
    const ColorImageR8G8B8 orgRGB,
    const ColorImageR8G8B8 orgMask,
    std::vector<float>* flowField, 
    ColorImageR8G8B8& warpedRGB,
    ColorImageR8G8B8& warpedMask) {

    std::vector<unsigned int> dims;

    std::vector<float2>* warpField;

    dims = { orgRGB.getWidth(), orgRGB.getHeight() };
    warpField = new std::vector<float2>(dims[0]*dims[1]);

    // Converting flow field to warp field, by adding the grid index
    int n = dims[0] * 2;
    for (unsigned int y = 0; y < dims[1]; y++)
        for (unsigned int x = 0; x < dims[0]; x++) {
            float fx = (*flowField)[y*n + 2*x];
            float fy = (*flowField)[y*n + 2*x + 1];
            (*warpField)[y*dims[0] + x] = { (float)x + fx, (float)y + fy  };
        }

    warpedRGB = ColorImageR8G8B8(dims[0], dims[1]);
    warpedRGB.setPixels(vec3uc(0, 0, 0));

    warpedMask = ColorImageR8G8B8(dims[0], dims[1]);
    warpedMask.setPixels(vec3uc(0, 0, 0));


    // Rasterize the results
    for (unsigned int y = 0; y < dims[1]; y++)
        for (unsigned int x = 0; x < dims[0]; x++)
            if (y + 1 < dims[1] && x + 1 < dims[0])
                if (orgMask(x, y).x == 0) {
                    float2 pos00 = (*warpField)[y*dims[0] + x];
                    float2 pos01 = (*warpField)[y*dims[0] + (x + 1)];
                    float2 pos10 = (*warpField)[(y + 1)*dims[0] + x];
                    float2 pos11 = (*warpField)[(y + 1)*dims[0] + (x + 1)];

                    vec3f v00 = orgRGB(x, y);
                    vec3f v01 = orgRGB(x + 1, y);
                    vec3f v10 = orgRGB(x, y + 1);
                    vec3f v11 = orgRGB(x + 1, y + 1);

                    bool valid00 = (orgMask(x, y).x == 0);
                    bool valid01 = (orgMask(x, y + 1).x == 0);
                    bool valid10 = (orgMask(x + 1, y).x == 0);
                    bool valid11 = (orgMask(x + 1, y + 1).x == 0);

                    if (valid00 && valid01 && valid10 && valid11) {
                        rasterizeTriangle (warpedRGB,
                            pos00, pos01, pos10,
                            v00, v01, v10);
                        rasterizeTriangle (warpedRGB,
                            pos10, pos01, pos11,
                            v10, v01, v11);
                    }

                    v00 = 1-orgMask(x, y).x;
                    v01 = 1-orgMask(x + 1, y).x;
                    v10 = 1-orgMask(x, y + 1).x;
                    v11 = 1-orgMask(x + 1, y + 1).x;

                    valid00 = (orgMask(x, y).x == 0);
                    valid01 = (orgMask(x, y + 1).x == 0);
                    valid10 = (orgMask(x + 1, y).x == 0);
                    valid11 = (orgMask(x + 1, y + 1).x == 0);

                    if (valid00 && valid01 && valid10 && valid11) {
                        rasterizeTriangle(warpedMask,
                            pos00, pos01, pos10,
                            v00, v01, v10,
                            true);
                        rasterizeTriangle(warpedMask,
                            pos10, pos01, pos11,
                            v10, v01, v11,
                            true);
                    }
                }
}

// read a flowImg file into 2-band orgRGB
void ReadFlowFile(std::vector<float>*& img, const char* inp_imgPath)
{
    if (inp_imgPath == NULL)
        printf("ReadFlowFile: empty inp_imgPath");

    const char *dot = strrchr(inp_imgPath, '.');
    if (strcmp(dot, ".flo") != 0)
        printf("ReadFlowFile (%s): extension .flo expected", inp_imgPath);

    FILE *stream = fopen(inp_imgPath, "rb");
    if (stream == 0)
        printf("ReadFlowFile: could not open %s", inp_imgPath);

    int width, height;
    float tag;

    if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
    (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
    (int)fread(&height, sizeof(int),   1, stream) != 1)
        printf("ReadFlowFile: problem reading file %s", inp_imgPath);

    if (tag != TAG_FLOAT) // simple test for correct endian-ness
    printf("ReadFlowFile(%s): wrong tag (possibly due to big-endian machine?)", inp_imgPath);

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999)
        printf("ReadFlowFile(%s): illegal width %d", inp_imgPath, width);

    if (height < 1 || height > 99999)
        printf("ReadFlowFile(%s): illegal height %d", inp_imgPath, height);

    int nBands = 2;
    img = new std::vector<float>(width * height * 2);

    int n = nBands * width;
    float* ptr = &(*img)[0]; //&(*img)[y*n]; //.Pixel(0, y, 0);
    for (int y = 0; y < height; y++) {
        if ((int)fread(ptr, sizeof(float), n, stream) != n)
            printf("ReadFlowFile(%s): file is too short", inp_imgPath);
        ptr += n;
    }

    if (fgetc(stream) != EOF)
        printf("ReadFlowFile(%s): file is too long", inp_imgPath);

    fclose(stream);
}


// write a 2-band orgRGB into flowImg file 
void WriteFlowFile(std::vector<float2>* img, const char* inp_imgPath, int width, int height)
{
    std::vector<float> v(10);
    float *p = &v[0];
    FILE *stream = fopen(inp_imgPath, "wb");
    //
    // write the header
    fprintf(stream, TAG_STRING);
    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
    (int)fwrite(&height, sizeof(int),   1, stream) != 1)
        printf ("WriteFlowFile(%s): problem writing header\n", inp_imgPath); 

    // write the rows
    int n = width;
    float2* ptr = &(*img)[0];
    for (int y = 0; y < height; y++) {
        if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
            printf("WriteFlowFile(%s): problem writing data", inp_imgPath); 
        ptr += n;
   }

    fclose(stream);
}

int main(int argc, const char * argv[]) {

    std::string inp_imgPath, inp_mskPath, inp_floPath;
    std::string out_warpedImgPath, out_warpedSegPath;

    if (argc == 6) {
        inp_imgPath = argv[1];
        inp_mskPath = argv[2];
        inp_floPath = argv[3];
        out_warpedImgPath = argv[4];
        out_warpedSegPath = argv[5];
    }
    else {
        printf("Invalid Input! ");
        usage();
        return 1;
    }

    const ColorImageR8G8B8A8 orgRGB = LodePNG::load(inp_imgPath);
    const ColorImageR8G8B8A8 orgMask = LodePNG::load(inp_mskPath);
    std::vector<float>* flowImg;
    ReadFlowFile(flowImg, inp_floPath.c_str());

    ColorImageR8G8B8 warpedRGB;
    ColorImageR8G8B8 warpedMask;

    Warp(orgRGB, orgMask, flowImg, warpedRGB, warpedMask);

    LodePNG::save(warpedRGB, out_warpedImgPath);
    LodePNG::save(warpedMask, out_warpedSegPath);

    printf("Saved\n");

    return 0;
}
