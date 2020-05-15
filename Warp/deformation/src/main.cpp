#include "main.h"
#include "CombinedSolver.h"

struct inputPaths {
    std::string inp_imgPath;
    std::string inp_mskPath;
    std::string inp_cstrPath;
    std::string out_floPath;
    std::string out_imgPath;
    std::string out_mskPath;
};

void usage() {
#define p(msg)  printf(msg "\n");
    p("Usage:\n");
    p("./arap_deform RGB Mask Constraint Flow warped_RGB warped_Mask\n");
    p("Mask and warp image using the provided optical flow field.\n")
    p("RGB \t\t [input]  path to an input RGB image (.png only)")
    p("Mask\t\t [input]  path to an input mask image (.png only) where 0 for object, 1 for background")
    p("Constraint \t [input]  path to list of constraints, text file")
    p("Flow \t\t [output] path to optical flow image with (.flo only)")
    p("warped_RGB \t [output] path to output warped image (.png), all intermediate directories must exist")
    p("warped_Mask \t [output] path to output warped mask (.png), all intermediate directories must exist")
}

static
void loadConstraints(
        std::vector<std::vector<int> >& constraints,
        std::string inp_imgPath) {

    std::ifstream in(inp_imgPath, std::fstream::in);

    if(!in.good()) {
        std::cout << "Could not open marker file " << inp_imgPath << std::endl;
        assert(false);
    }

    unsigned int nMarkers;
    in >> nMarkers;
    constraints.resize(nMarkers);
    for(unsigned int m = 0; m<nMarkers; m++) {
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

void deformSingle(
    std::string inp_imgPath,
    std::string inp_mskPath,
    std::string inp_cstrPath,
    std::string out_floPath,
    std::string out_imgPath,
    std::string out_mskPath,
    CombinedSolver solver,
    CombinedSolverParameters params) {

    std::vector<std::vector<int>> constraints;
    loadConstraints(constraints, inp_cstrPath);

    ColorImageR8G8B8A8 orgRGB = LodePNG::load(inp_imgPath);
    const unsigned int width = orgRGB.getWidth();
    const unsigned int height = orgRGB.getHeight();
    const ColorImageR8G8B8A8 orgMask = LodePNG::load(inp_mskPath);

    for (unsigned int y = 0; y < height; y++)
        for (unsigned int x = 0; x < width; x++)
            if (y == 0 || x == 0 || 
                y == (height - 1) || x == (width - 1)) {
                std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
                constraints.push_back(v);
            }

    solver.addImage(orgRGB, orgMask, constraints, params);
    solver.solveAll();

    ColorImageR8G8B8* warpedRGB = solver.result();
    ColorImageR8G8B8* warpedMask = solver.resultSeg();

    LodePNG::save(*warpedRGB, out_imgPath);
    LodePNG::save(*warpedMask, out_mskPath);

    WriteFlowFile(solver.warpField(), out_floPath.c_str(), warpedMask->getWidth(), warpedMask->getHeight());
    printf("Saved\n");
}

void loadData(inputPaths paths_1frame,
        ColorImageR8G8B8 &orgRGB,
        ColorImageR8G8B8 &orgMask,
        std::vector<std::vector<int>> &constraints) {

    loadConstraints(constraints, paths_1frame.inp_cstrPath);

    orgRGB = LodePNG::load(paths_1frame.inp_imgPath);
    const unsigned int width = orgRGB.getWidth();
    const unsigned int height = orgRGB.getHeight();
    orgMask = LodePNG::load(paths_1frame.inp_mskPath);

    // TODO: sanity check the size of orgMask and orgRGB

    for (unsigned int y = 0; y < height; y++)
        for (unsigned int x = 0; x < width; x++)
            if (y == 0 || x == 0 || 
                y == (height - 1) || x == (width - 1)) {
                std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
                constraints.push_back(v);
            }
    return ;
}

void deformSingle(
    inputPaths paths_1frame,
    ColorImageR8G8B8 orgRGB,
    ColorImageR8G8B8 orgMask,
    std::vector<std::vector<int>> constraints,
    CombinedSolver solver,
    CombinedSolverParameters params) {

    solver.addImage(orgRGB, orgMask, constraints, params);
    solver.solveAll();

    ColorImageR8G8B8* warpedRGB = solver.result();
    ColorImageR8G8B8* warpedMask = solver.resultSeg();

    LodePNG::save(*warpedRGB, paths_1frame.out_imgPath);
    LodePNG::save(*warpedMask, paths_1frame.out_mskPath);

    WriteFlowFile(solver.warpField(), paths_1frame.out_floPath.c_str(),
            warpedMask->getWidth(), warpedMask->getHeight());
    printf("Saved\n");
}

int main(int argc, const char * argv[]) {

    std::vector<inputPaths> lines;
    inputPaths paths_1frame;

    // Read command line arguments
    // receive all the relevant paths for 1 frame and run once
    if (argc == 7) {

        paths_1frame.inp_imgPath = argv[1];
        paths_1frame.inp_mskPath = argv[2];
        paths_1frame.inp_cstrPath = argv[3];
        paths_1frame.out_floPath = argv[4];
        paths_1frame.out_imgPath = argv[5];
        paths_1frame.out_mskPath = argv[6];

        lines.push_back(paths_1frame);
    }
    else if (argc == 2) { // receive a path to list file for multiple frames
        std::ifstream infile(argv[1]);
        std::string line;
        while (getline(infile, line)) {
            std::stringstream s(line);
            s   >> paths_1frame.inp_imgPath
                >> paths_1frame.inp_mskPath
                >> paths_1frame.inp_cstrPath
                >> paths_1frame.out_floPath
                >> paths_1frame.out_imgPath
                >> paths_1frame.out_mskPath;
            lines.push_back(paths_1frame);
        }
    }
    else {
        printf("Invalid Input!\n");
        usage();
        return 1;
    }

    // sanity check
    unsigned int len = static_cast<unsigned int>(lines.size());
    if (len == 0) {
        printf("No file to be processed");
        return 1;
    }
    const char *planPath = getenv("ARAP_PLAN") == NULL ? "arap_plan.t" : getenv("ARAP_PLAN");
    printf("Optimization plan at %s\n", planPath);
    std::ifstream f(planPath);
    if (!f.good()) {
        printf (" Not found! Please run export ARAP_PLAN=/path/to/plan.t or copy "
                "the file to the running folder with name arap_plan.t");
        return 1;
    }

    CombinedSolverParameters params;
    params.numIter = 19;
    params.useCUDA = false;
    params.useOpt = true;
    params.nonLinearIter = 8;
    params.linearIter = 400;
    params.profileSolve = false;

    // Process the first image
    ColorImageR8G8B8 orgRGB;
    ColorImageR8G8B8 orgMask;
    std::vector<std::vector<int>> constraints;
    loadData(lines[0], orgRGB, orgMask, constraints);
    unsigned int width = orgRGB.getWidth();
    unsigned int height = orgRGB.getHeight();

    CombinedSolver solver(width, height, planPath, params);
    deformSingle(lines[0], orgRGB, orgMask, constraints, solver, params);

    for (int i=1; i < len; ++i) {
        constraints.clear();
        loadData(lines[i], orgRGB, orgMask, constraints);
        deformSingle(lines[i], orgRGB, orgMask, constraints, solver, params);
    }

    return 0;
}
