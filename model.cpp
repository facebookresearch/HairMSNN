// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "extern/owl/3rdParty/stb_image/stb/stb_image.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

using namespace owl;

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

vec2f getBarycentrics(vec3f t1, vec3f t2, vec3f t3, vec3f point)
{
    vec3f s1 = t1 - t3;
    vec3f s2 = t2 - t3;
    float triArea = 0.5f * length(cross(s1, s2));

    s1 = t1 - point;
    s2 = t2 - point;
    float area1 = 0.5f * length(cross(s1, s2));

    s1 = t3 - point;
    float area2 = 0.5f * length(cross(s1, s2));

    float u = area1 / triArea;
    float v = area2 / triArea;

    return vec2f(u, v);
}

vec3f Texture::sample(vec2f uv)
{
    vec3f rval(0.f);

    uv.x = int(uv.x * (this->resolution.x - 1));
    uv.y = int(uv.y * (this->resolution.y - 1));

    int idx = uv.x + this->resolution.x * uv.y;
    uint32_t pixel = this->pixel[idx];

    uint32_t rmask = 255;
    uint32_t gmask = rmask << 8;
    uint32_t bmask = rmask << 16;

    float r = float(pixel & rmask) / 255.f;
    float g = float((pixel & gmask) >> 8) / 255.f;
    float b = float((pixel & bmask) >> 16) / 255.f;

    rval = vec3f(r, g, b);
    return rval;
}

vec4f TriangleMesh::sampleSurfaceParams(std::vector<Texture*>& textures, vec3f point, vec3i index)
{
    vec4f rval(0.f);

    vec3f v1 = this->vertex[index.x];
    vec3f v2 = this->vertex[index.y];
    vec3f v3 = this->vertex[index.z];

    vec2f bary = getBarycentrics(v1, v2, v3, point);

    vec2f uv = bary.y * this->texcoord[index.x]
        + bary.x * this->texcoord[index.z]
        + (1.f - bary.x - bary.y) * this->texcoord[index.y];

    uv.x = max(min(1.f, uv.x), 0.f);
    uv.y = max(min(1.f, uv.y), 0.f);
    
    if (this->diffuseTextureID == -1) {
        rval.x = this->diffuse.x;
        rval.y = this->diffuse.y;
        rval.z = this->diffuse.z;
    }
    else {
        vec3f col = textures[this->diffuseTextureID]->sample(uv);
        rval.x = col.x;
        rval.y = col.y;
        rval.z = col.z;
    }

    if (this->alphaTextureID == -1) {
        rval.w = this->alpha * this->alpha;
    }
    else {
        vec3f alpha = textures[this->alphaTextureID]->sample(uv);
        rval.w = 0.33f * (alpha.x + alpha.y + alpha.z);
        rval.w = rval.w * rval.w;
    }

    return rval;
}

/*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
int addVertex(TriangleMesh* mesh,
    tinyobj::attrib_t& attributes,
    const tinyobj::index_t& idx,
    std::map<tinyobj::index_t, int>& knownVertices)
{
    if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
    const vec3f* normal_array = (const vec3f*)attributes.normals.data();
    const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();

    int newID = (int)mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
        while (mesh->normal.size() < mesh->vertex.size())
            mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
        while (mesh->texcoord.size() < mesh->vertex.size())
            mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
        mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
        mesh->normal.resize(mesh->vertex.size());

    return newID;
}

/* Load environment light texture */
bool loadEnvTexture(std::string& path, Texture *texture)
{
    float* out; // width * height * RGBA
    float* flipped; // width * height * RGBA
    int width;
    int height;
    const char* err = nullptr; // or nullptr in C++11

    int ret = LoadEXR(&out, &width, &height, path.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
        LOG("Could not load environment map from " << path);
        return false;
    }
    else {
        vec2i res(width, height);
        texture->resolution = res;
        texture->pixel_float = out;

        return true;
    }

    return false;
}

/*! load a texture (if not already loaded), and return its ID in the
    model's textures[] vector. Textures that could not get loaded
    return -1 */
int loadTexture(Model* model,
    std::map<std::string, int>& knownTextures,
    const std::string& inFileName)
{
    if (inFileName == "")
        return -1;

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto& c : fileName)
        if (c == '\\') c = '/';

    vec2i res;
    int   comp;
    unsigned char* image = stbi_load(fileName.c_str(),
        &res.x, &res.y, &comp, STBI_rgb_alpha);
    int textureID = -1;
    if (image) {
        textureID = (int)model->textures.size();
        Texture* texture = new Texture;
        texture->resolution = res;
        texture->pixel = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
            mirrored along the y axis - mirror them here */
        for (int y = 0; y < res.y / 2; y++) {
            uint32_t* line_y = texture->pixel + y * res.x;
            uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
            int mirror_y = res.y - 1 - y;
            for (int x = 0; x < res.x; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        model->textures.push_back(texture);
    }
    else {
        LOG("Could not load texture from " << fileName);
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

Model* loadOBJ(const std::string& objFile)
{
    Model* model = new Model;

    const std::string modelDir
        = objFile.substr(0, objFile.rfind('/') + 1);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
        = tinyobj::LoadObj(&attributes,
            &shapes,
            &materials,
            &err,
            &err,
            objFile.c_str(),
            modelDir.c_str(),
            /* triangulate */true);
    if (!readOK) {
        throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
    const vec3f* normal_array = (const vec3f*)attributes.normals.data();
    const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    std::map<std::string, int>      knownTextures;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        std::map<tinyobj::index_t, int> knownVertices;

        for (int materialID : materialIDs) {
            TriangleMesh* mesh = new TriangleMesh;

            float totalArea = 0.f;
            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                vec3f v1 = vertex_array[idx0.vertex_index];
                vec3f v2 = vertex_array[idx1.vertex_index];
                vec3f v3 = vertex_array[idx2.vertex_index];

                vec3i vidx(mesh->vertex.size(), mesh->vertex.size() + 1, mesh->vertex.size() + 2);
                mesh->vertex.push_back(v1);
                mesh->vertex.push_back(v2);
                mesh->vertex.push_back(v3);
                mesh->index.push_back(vidx);

                vec3f s1 = v2 - v1;
                vec3f s2 = v3 - v1;
                vec3f cp = cross(s1, s2);
                float area = 0.5f * abs(length(cp));
                mesh->triangleArea.push_back(area);
                totalArea += area;

                vec3i nidx(mesh->normal.size(), mesh->normal.size() + 1, mesh->normal.size() + 2);
                mesh->normal.push_back(normal_array[idx0.normal_index]);
                mesh->normal.push_back(normal_array[idx1.normal_index]);
                mesh->normal.push_back(normal_array[idx2.normal_index]);
                // mesh->index.push_back(nidx);

                vec3i tidx(mesh->texcoord.size(), mesh->texcoord.size() + 1, mesh->texcoord.size() + 2);
                mesh->texcoord.push_back(texcoord_array[idx0.texcoord_index]);
                mesh->texcoord.push_back(texcoord_array[idx1.texcoord_index]);
                mesh->texcoord.push_back(texcoord_array[idx2.texcoord_index]);
                // mesh->index.push_back(tidx);

                mesh->minBound = min(mesh->minBound, v1);
                mesh->maxBound = max(mesh->maxBound, v1);

                mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
                mesh->diffuseTextureID = loadTexture(model,
                    knownTextures,
                    materials[materialID].diffuse_texname);

                mesh->alpha = 1.f;
                mesh->alphaTextureID = loadTexture(model,
                    knownTextures,
                    materials[materialID].specular_highlight_texname);
            }

            mesh->totalSurfaceArea = totalArea;

            if (mesh->vertex.empty()) {
                delete mesh;
            }
            else {
                for (auto idx : mesh->index) {
                    if (idx.x < 0 || idx.x >= (int)mesh->vertex.size() ||
                        idx.y < 0 || idx.y >= (int)mesh->vertex.size() ||
                        idx.z < 0 || idx.z >= (int)mesh->vertex.size()) {
                        LOG("invalid triangle indices");
                        throw std::runtime_error("invalid triangle indices");
                    }
                }

                model->meshes.push_back(mesh);
                model->totalArea += mesh->totalSurfaceArea;
            }
        }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
        for (auto vtx : mesh->vertex)
            model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
}

int saveBufferAsEXR(std::string path, float* hostBuffer, int width, int height, int numComponents)
{
    std::string savePath = path;
    return SaveEXR(hostBuffer, width, height, numComponents, 0, savePath.c_str(), nullptr);
}

int saveEXR(std::string path,  float* hostBuffer, int accumId, int width, int height)
{
    std::vector<float> inverted;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = x + (height-1-y) * width;

            inverted.push_back(hostBuffer[idx * 4 + 0]);
            inverted.push_back(hostBuffer[idx * 4 + 1]);
            inverted.push_back(hostBuffer[idx * 4 + 2]);
            inverted.push_back(hostBuffer[idx * 4 + 3]);
        }
    }

    LOG("(EXR) Accum buffer saved to file!");

    return saveBufferAsEXR(path, inverted.data(), width, height, 4);
}

float* loadBufferFromEXR(std::string path, int *width, int *height, int numComponents)
{
    std::string loadPath = path;
    const char* err = (const char*) malloc(100*sizeof(char));

    float* hostBufferRead;
    float* hostBuffer;
    int rval = LoadEXR(&hostBufferRead, width, height, loadPath.c_str(), &err);

    std::string errSt(err);
    std::cout << rval << ": " << errSt << std::endl;

    free((void*)err);

    if(numComponents == 4)
        return hostBufferRead;
    else {
        int numEl = (*width) * (*height);
        hostBuffer = (float*)malloc(numEl * 3 * sizeof(float));

        for (int i = 0; i < numEl; i++) {
            hostBuffer[i * 3] = hostBufferRead[i * 4];
            hostBuffer[i * 3 + 1] = hostBufferRead[i * 4 + 1];
            hostBuffer[i * 3 + 2] = hostBufferRead[i * 4 + 2];
        }

        free(hostBufferRead);

        return hostBuffer;
    }
}