#version 330 core

struct Material {
    sampler2D base_color_texture;
    sampler2D emissive_texture;
};

// Texture Attributes
uniform Material material;

// Inputs
in vec2 uv_0;
in vec2 uv_1;
in float layer;

// Outputs
out vec4 frag_color;

void main()
{
    if (layer > 0) {
        frag_color = texture(material.emissive_texture, uv_1);
    } else {
        frag_color = texture(material.base_color_texture, uv_0);
    }
}
