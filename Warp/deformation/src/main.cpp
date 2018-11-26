#include "main.h"
#include "CombinedSolver.h"

static void loadConstraints(std::vector<std::vector<int> >& constraints, std::string inp_imgPath) {
  std::ifstream in(inp_imgPath, std::fstream::in);

    if(!in.good())
    {
        std::cout << "Could not open marker file " << inp_imgPath << std::endl;
        assert(false);
    }

    unsigned int nMarkers;
    in >> nMarkers;
    constraints.resize(nMarkers);
    for(unsigned int m = 0; m<nMarkers; m++)
    {
        int temp;
        for (int i = 0; i < 4; ++i) {
            in >> temp;
            constraints[m].push_back(temp);
        }

    }

    in.close();
}

// write a 2-band orgRGB into flow file 
void WriteFlowFile(std::vector<float>* img, const char* inp_imgPath, int width, int height)
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
    int n = 2 * width;
    float* ptr = &(*img)[0];
    for (int y = 0; y < height; y++) {
        if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
            printf("WriteFlowFile(%s): problem writing data", inp_imgPath); 
        ptr += n;
   }

    fclose(stream);
}

int main(int argc, const char * argv[]) {

    std::string inp_imgPath;
    std::string inp_mskPath;
    std::string inp_cstrPath;
    std::string segfilename;
    std::string out_imgPath;
    std::string out_mskPath;
    std::string out_floPath;

    if (argc == 7) {
        inp_imgPath = argv[1];
        inp_mskPath = argv[2];
        inp_cstrPath = argv[3];
        out_imgPath = argv[4];
        out_floPath = argv[5];
        out_mskPath = argv[6];
    }
    else {
        printf("Invalid Input!");
        return 1;
    }

    std::vector<std::vector<int>> constraints;
    loadConstraints(constraints, inp_cstrPath);

    ColorImageR8G8B8A8 orgRGB = LodePNG::load(inp_imgPath);
    const ColorImageR8G8B8A8 orgMask = LodePNG::load(inp_mskPath);

    for (unsigned int y = 0; y < orgRGB.getHeight(); y++)
        for (unsigned int x = 0; x < orgRGB.getWidth(); x++)
            if (y == 0 || x == 0 || 
                y == (orgRGB.getHeight() - 1) || x == (orgRGB.getWidth() - 1)) {
                std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
                constraints.push_back(v);
            }

    CombinedSolverParameters params;
    params.numIter = 19;
    params.useCUDA = false;
    params.useOpt = true;
    params.nonLinearIter = 8;
    params.linearIter = 400;
    params.profileSolve = false;

    CombinedSolver solver(orgRGB, orgMask, constraints, params);
    solver.solveAll();

    ColorImageR8G8B8* warpedRGB = solver.result();
    ColorImageR8G8B8* warpedMask = solver.resultSeg();

    LodePNG::save(*warpedRGB, out_imgPath);
    LodePNG::save(*warpedMask, out_mskPath);

    WriteFlowFile(solver.warpField(), out_floPath.c_str(), warpedMask->getWidth(), warpedMask->getHeight());
    printf("Saved\n");

    return 0;
}
