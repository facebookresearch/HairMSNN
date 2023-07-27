// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "scene.h"

void Scene::extractHairData()
{
    int numControlPoints = this->hair.GetHeader().point_count;
    int numStrands = this->hair.GetHeader().hair_count;
    const float* controlPointsData = this->hair.GetPointsArray();

    const unsigned short* segmentsData = this->hair.GetSegmentsArray();
    const unsigned short defaultNumSegments = this->hair.GetHeader().d_segments;

    const float* widthsData = this->hair.GetThicknessArray();
    const float defaultWidth = this->hair.GetHeader().d_thickness;

    int strandIndex = 0;
    int numSegmentsLeft = -1;

    for (int i = 0; i < numControlPoints; i++) {
        const vec3f p = vec3f(controlPointsData[i * 3], controlPointsData[i * 3 + 1], controlPointsData[i * 3 + 2]);
        const float w = widthsData ? widthsData[i] : defaultWidth;

        this->hairModel.maxBound = max(p, this->hairModel.maxBound);
        this->hairModel.minBound = min(p, this->hairModel.minBound);

        bool firstSegment = false;

        if (numSegmentsLeft == -1) {
            if (segmentsData)
                numSegmentsLeft = segmentsData[strandIndex];
            else
                numSegmentsLeft = defaultNumSegments;

            if (strandIndex == numStrands)
                break;

            strandIndex++;
            firstSegment = true;

            this->hairModel.numSgmentsInStrand.push_back(numSegmentsLeft);
        }

        if (firstSegment) {
            const vec3f p_next = vec3f(controlPointsData[(i + 1) * 3], controlPointsData[(i + 1) * 3 + 1], controlPointsData[(i + 1) * 3 + 2]);
            this->hairModel.controlPoints.push_back(p + (p - p_next));
            this->hairModel.widths.push_back(.2f * w);
            this->hairModel.segmentIndices.push_back(this->hairModel.controlPoints.size() - 1);
        }

        this->hairModel.controlPoints.push_back(p);
        this->hairModel.widths.push_back(.2f * w);

        if (numSegmentsLeft > 1) {
            this->hairModel.segmentIndices.push_back(this->hairModel.controlPoints.size() - 1);
        }
        else if (numSegmentsLeft == 0) {
            const vec3f p_prev = vec3f(controlPointsData[(i - 1) * 3], controlPointsData[(i - 1) * 3 + 1], controlPointsData[(i - 1) * 3 + 2]);
            this->hairModel.controlPoints.push_back(p + (p - p_prev));
            this->hairModel.widths.push_back(.2f * w);
        }

        numSegmentsLeft--;
    }

    this->hairModel.numStrands = numStrands;
    this->hairModel.scale = length(this->hairModel.maxBound - this->hairModel.minBound);
}

void LoadCemYuksel(const char* filename, cyHairFile& hairfile)
{
    // Load the hair model
    int result = hairfile.LoadFromFile(filename);
    // Check for errors
    switch (result) {
    case CY_HAIR_FILE_ERROR_CANT_OPEN_FILE:
        printf("Error: Cannot open hair file!\n");
        return;
    case CY_HAIR_FILE_ERROR_CANT_READ_HEADER:
        printf("Error: Cannot read hair file header!\n");
        return;
    case CY_HAIR_FILE_ERROR_WRONG_SIGNATURE:
        printf("Error: File has wrong signature!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_SEGMENTS:
        printf("Error: Cannot read hair segments!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_POINTS:
        printf("Error: Cannot read hair points!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_COLORS:
        printf("Error: Cannot read hair colors!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_THICKNESS:
        printf("Error: Cannot read hair thickness!\n");
        return;
    case CY_HAIR_FILE_ERROR_READING_TRANSPARENCY:
        printf("Error: Cannot read hair transparency!\n");
        return;
    default:
        printf("Hair file \"%s\" loaded.\n", filename);
    }
    int hairCount = hairfile.GetHeader().hair_count;
    int pointCount = hairfile.GetHeader().point_count;
    printf("Number of hair strands = %d\n", hairCount);
    printf("Number of hair points = %d\n", pointCount);
}

/*
Returns:
true - scene loaded
false - scene load failed
*/
bool parseScene(std::string sceneFile, Scene& scene)
{
    nlohmann::json sceneConfig;
    try {
        std::ifstream sceneStream(sceneFile.c_str());
        sceneStream >> sceneConfig;
    }
    catch (std::runtime_error e) {
        LOG("Could not load scene .json file! Exiting...");
        return false;
    }

    scene.json = sceneConfig;
    scene.jsonFilePath = sceneFile;

    // ================================
    // Load either .hair or .obj file
    // ================================
    bool is_obj = false, is_hair = false;
    try {
        std::string objFilePath = sceneConfig["surface"]["geometry"];
        std::cout << objFilePath << std::endl;

        scene.surface = loadOBJ(objFilePath);
        scene.has_surface = true;
        is_obj = true;
    }
    catch (nlohmann::json::exception e) {
        scene.has_surface = false;
        LOG("No surface geometry found");
    }

    try {
        std::string hairFilePath = sceneConfig["hair"]["geometry"];

        // Load Cem Yuksel's hair models
        LoadCemYuksel(hairFilePath.c_str(), scene.hair);
        scene.extractHairData();

        is_hair = true;
        scene.has_hair = true;
    }
    catch (nlohmann::json::exception e) {
        scene.has_hair = false;
        LOG("No hair geometry found");
    }

    if (!is_obj && !is_hair) {
        LOG("Either hair or surface must be defined! Exiting...");
        return false;
    }

    // ================================
    // Camera setup
    // ================================
    try {
        auto camConfig = sceneConfig["camera"];

        SceneCamera cam;
        cam.from = vec3f(camConfig["from"][0], camConfig["from"][1], camConfig["from"][2]);
        cam.at = vec3f(camConfig["to"][0], camConfig["to"][1], camConfig["to"][2]);
        cam.up = vec3f(camConfig["up"][0], camConfig["up"][1], camConfig["up"][2]);
        cam.cosFovy = float(camConfig["cos_fovy"]);

        scene.camera = cam;
    }
    catch (nlohmann::json::exception e) {
        LOG("Camera must be defined! Exiting...");
        return false;
    }

    // ================================
    // Hair BSDF setup
    // ================================
    try {
        auto bsdfConfig = sceneConfig["hair"];

        scene.bsdf = bsdfConfig["type"];

        try { scene.sig_a = vec3f(bsdfConfig["sigma_a"][0], bsdfConfig["sigma_a"][1], bsdfConfig["sigma_a"][2]); }
        catch (nlohmann::json::exception e) { scene.sig_a = vec3f(0.06, 0.1, 0.2); }

        try { scene.beta_m = float(bsdfConfig["beta_m"]); }
        catch (nlohmann::json::exception e) { scene.beta_m = 0.3f; }

        try { scene.beta_n = float(bsdfConfig["beta_n"]); }
        catch (nlohmann::json::exception e) { scene.beta_n = 0.3f; }

        try { scene.alpha = 3.14159f * float(bsdfConfig["alpha"]) / 180.f; }
        catch (nlohmann::json::exception e) { scene.alpha = 0.f; }

        try { scene.R_G = float(bsdfConfig["Gain R"]); }
        catch (nlohmann::json::exception e) { scene.R_G = 1.f; }

        try { scene.TT_G = float(bsdfConfig["Gain TT"]); }
        catch (nlohmann::json::exception e) { scene.TT_G = 1.f; }

        try { scene.TRT_G = float(bsdfConfig["Gain TRT"]); }
        catch (nlohmann::json::exception e) { scene.TRT_G = 1.f; }

        try { scene.TRRT_G = float(bsdfConfig["Gain TRRT"]); }
        catch (nlohmann::json::exception e) { scene.TRRT_G = 1.f; }

        try { scene.pixarFactor = float(bsdfConfig["pixar_factor"]); }
        catch (nlohmann::json::exception e) { scene.pixarFactor = 1.f; }

        // Dual scattering precomp
        try { scene.dualScatterPrecompPath = std::string(bsdfConfig["dual_scatter_precomputation"]); }
        catch (nlohmann::json::exception e) { scene.dualScatterPrecompPath = "C:\\Users\\Projects\\MetaHair\\precomputation"; }
    }
    catch (nlohmann::json::exception e) {
        LOG("Hair material not defined. Using default values...");
    }

    // ====================================
    // Load enviroinment light, if present
    // ====================================
    bool is_env = false;
    scene.hasEnvLight = false;
    try {
        auto envConfig = sceneConfig["lights"]["environment"];

        std::string envTexPath = envConfig["exr"];
        loadEnvTexture(envTexPath, &scene.env);
        is_env = true;

        scene.hasEnvLight = true;
        scene.envScale = (float)envConfig["scale"];

        try { scene.envRotPhi = float(envConfig["rotation"]); }
        catch (nlohmann::json::exception e) { scene.envRotPhi = 0.f; }
    }
    catch (nlohmann::json::exception e) {
        LOG("No environment light found");
    }

    // ====================================
    // Load directional lights
    // ====================================
    bool is_directional = false;
    scene.hasDirectionalLights = false;
    try {
        auto dLights = sceneConfig["lights"]["directional"];

        for (auto d : dLights) {
            vec3f from = normalize(vec3f(d["from"][0], d["from"][1], d["from"][2]));
            vec3f emit = vec3f(d["emit"][0], d["emit"][1], d["emit"][2]);

            scene.dLightFrom.push_back(from);
            scene.dLightEmit.push_back(emit);
        }

        scene.hasDirectionalLights = true;
        is_directional = true;
    }
    catch (nlohmann::json::exception e) {
        LOG("No directional lights found");
    }

    if (!is_directional && !is_env) {
        LOG("Either directional or environment light must be defined! Exiting...");
        return false;
    }

    // ====================================
    // Load integrator parameters
    // ====================================
    try {
        auto integratorConfig = sceneConfig["integrator"];

        scene.spp = integratorConfig["spp"];
        scene.path_v1 = integratorConfig["path_v1"];
        scene.path_v2 = integratorConfig["path_v2"];
        scene.imgWidth = integratorConfig["width"];
        scene.imgHeight = integratorConfig["height"];
        scene.renderOutput = integratorConfig["image_output"];
        scene.renderStatsOutput = integratorConfig["stats_output"];

        scene.MIS = integratorConfig["MIS"];
        scene.envPdfSampling = integratorConfig["ENV_PDF"];

        // First cond: Dictated by TCNN's output buffer size (line 167, hair.h)
        // Second cond: TCNN limitation, batch size should be multiple of 128
        if (scene.imgHeight * scene.imgWidth > 2048 * 2048 ||
            (scene.imgHeight * scene.imgWidth) % 128 != 0) {
            LOG("Image size has a hard limit of 2K*2K!");
            return false;
        }
    }
    catch (nlohmann::json::exception e) {
        LOG("Integrator must be defined! Exiting...");
        return false;
    }

    // ====================================
    // Load NRC parameters
    // ====================================
    try {
        auto nrcConfig = sceneConfig["tcnn"];
        scene.nrcConfig = nrcConfig["config"];

        try {
            scene.nrcConfig2 = nrcConfig["config2"];
        }
        catch (nlohmann::json::exception e) {
            scene.nrcConfig2 = nrcConfig["config"];
        }
        
        try { scene.nrcTrain = nrcConfig["init_train"]; }
        catch (nlohmann::json::exception e) { scene.nrcTrain = false; }

        try { scene.nrcWeights = nrcConfig["init_weights"]; }
        catch (nlohmann::json::exception e) { scene.nrcWeights = ""; }
    }
    catch (nlohmann::json::exception e) {
        LOG("TCNN not defined.");
    }


    return true;
}

Scene parseScene_(std::string sceneFile)
{
    Scene scene;
    parseScene(sceneFile, scene);

    return scene;
}

void generateEnvSamplingTables(vec2i resolution, float* envMap, float** cPdf, float** cCdf, float** mPdf, float** mCdf)
{
    int width = resolution.x;
    int height = resolution.y;

    int cdfWidth = width + 1;
    *cPdf = (float*)malloc(cdfWidth * height * sizeof(float));
    *cCdf = (float*)malloc(cdfWidth * height * sizeof(float));

    auto average = [](const vec3f a) { return (a.x + a.y + a.z) * (1.0f / 3.0f); };

    // Conditional CDFs (rows, U direction)
    for (int y = 0; y < height; y++)
    {
        float sinTheta = sinf(3.14159f * (y + 0.5f) / height);

        vec3f envColor = vec3f(
            envMap[4 * y * width + 0], envMap[4 * y * width + 1], envMap[4 * y * width + 2]);
        float averageLuminance = average(envColor);

        (*cPdf)[y * cdfWidth] = averageLuminance * sinTheta;
        (*cCdf)[y * cdfWidth] = 0.0f;

        for (int x = 1; x < width; x++)
        {
            envColor = vec3f(
                envMap[4 * (y * width + x) + 0],
                envMap[4 * (y * width + x) + 1],
                envMap[4 * (y * width + x) + 2]);
            averageLuminance = average(envColor);

            (*cPdf)[y * cdfWidth + x] = averageLuminance * sinTheta;
            (*cCdf)[y * cdfWidth + x] =
                (*cCdf)[y * cdfWidth + x - 1] + (*cPdf)[y * cdfWidth + x - 1] / width;
        }

        const float cdfTotal =
            (*cCdf)[y * cdfWidth + width - 1] + (*cPdf)[y * cdfWidth + width - 1] / width;

        // stuff the total into the brightness value for the last entry, because
        // we are going to normalize the CDFs to 0.0 to 1.0 afterwards
        (*cPdf)[y * cdfWidth + width] = cdfTotal;

        if (cdfTotal > 0.0f)
        {
            const float cdfTotalInv = 1.0f / cdfTotal;
            for (int x = 1; x < width; x++)
            {
                (*cCdf)[y * cdfWidth + x] *= cdfTotalInv;
            }
        }

        (*cCdf)[y * cdfWidth + width] = 1.0f;
    }

    // marginal
    *mPdf = (float*)malloc((height + 1) * sizeof(float));
    *mCdf = (float*)malloc((height + 1) * sizeof(float));

    // marginal CDFs (column, V direction, sum of rows)
    (*mPdf)[0] = (*cPdf)[width];
    (*mCdf)[0] = 0.0f;

    for (int i = 1; i < height; i++)
    {
        (*mPdf)[i] = (*cPdf)[i * cdfWidth + width];
        (*mCdf)[i] = (*mCdf)[i - 1] + (*mPdf)[i - 1] / height;
    }

    float cdfTotal = (*mCdf)[height - 1] + (*mPdf)[height - 1] / height;
    (*mPdf)[height] = cdfTotal;

    if (cdfTotal > 0.0f)
        for (int i = 1; i < height; i++)
            (*mCdf)[i] /= cdfTotal;
    (*mCdf)[height] = 1.0f;
}