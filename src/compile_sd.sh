#/usr/bin/env sh


libigl_path="${HOME}"/soft/libigl

g++ \
    -O3 \
    -shared \
    -std=c++11 \
    -Wall \
    -fPIC \
    -I"${libigl_path}"/build/_deps/glad-src/include \
    -I"${libigl_path}"/build/_deps/glfw-src/include \
    -I/usr/include/eigen3 \
    -I"${libigl_path}"/include \
    $(python3 -m pybind11 --includes) \
    -L"${libigl_path}"/build/lib \
    -lpthread \
    -lOpenGL \
    -lrt \
    -lm \
    -ldl \
    -ligl \
    -ligl_glfw \
    -ligl_opengl \
    -lglad \
    -lglfw3 \
    fastSymDir/{fastSymDir,optimize}.{h,cpp} \
    -o fastSymDir$(python3-config --extension-suffix) \
