#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Inputs
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
layout(location = TEXCOORD_1_LOC) in vec2 texcoord_1;
layout(location = COLOR_0_LOC) in vec4 color_0;

// Outputs
out vec2 uv_0;
out vec2 uv_1;
out float layer;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);

    uv_0 = texcoord_0;
    uv_1 = texcoord_1;
    layer = color_0.r;
}
