// gpu_memory.h — tampon de mémoire GPU minimal pour cubvh (CuMesh).
//
// Écrit de zéro pour Menu AR, sous la licence MIT de ce dépôt, afin de remplacer un fichier
// sous licence NVIDIA non commerciale. Il ne fournit que ce dont cubvh se sert : allouer,
// redimensionner, recopier depuis le CPU, donner le pointeur GPU, libérer automatiquement.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#define MENU_GPU_HD __host__ __device__
#else
#define MENU_GPU_HD
#endif

namespace menu_gpu {
inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPUMemory : ") + what + " a échoué (" + cudaGetErrorString(err) + ")");
    }
}
}  // namespace menu_gpu

// Tableau de T en mémoire GPU, propriétaire de sa mémoire (libérée à la destruction).
// Déplaçable, non copiable (une copie silencieuse dupliquerait ou libérerait deux fois).
template <typename T>
class GPUMemory {
public:
    GPUMemory() = default;
    explicit GPUMemory(size_t n) { resize(n); }
    ~GPUMemory() { release(); }

    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;
    GPUMemory(GPUMemory&& other) noexcept { swap(other); }
    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) { release(); swap(other); }
        return *this;
    }

    MENU_GPU_HD T* data() const { return m_data; }
    MENU_GPU_HD size_t size() const { return m_size; }
    size_t bytes() const { return m_size * sizeof(T); }

    // Nouvelle taille ; le contenu précédent n'est pas conservé (comme une allocation neuve).
    void resize(size_t n) {
        if (n == m_size) return;
        release();
        if (n == 0) return;
        void* p = nullptr;
        menu_gpu::check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
        m_data = static_cast<T*>(p);
        m_size = n;
    }

    void memset(int value) {
        if (m_size) menu_gpu::check(cudaMemset(m_data, value, bytes()), "cudaMemset");
    }

    void copy_from_host(const T* src, size_t n) {
        if (n > m_size) throw std::runtime_error("GPUMemory : copie plus grande que le tampon");
        if (n) menu_gpu::check(cudaMemcpy(m_data, src, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy (CPU -> GPU)");
    }
    void copy_from_host(const std::vector<T>& src) { copy_from_host(src.data(), src.size()); }

    void resize_and_copy_from_host(const std::vector<T>& src) {
        resize(src.size());
        copy_from_host(src);
    }

    void copy_to_host(T* dst, size_t n) const {
        if (n > m_size) throw std::runtime_error("GPUMemory : lecture plus grande que le tampon");
        if (n) menu_gpu::check(cudaMemcpy(dst, m_data, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (GPU -> CPU)");
    }
    void copy_to_host(std::vector<T>& dst) const {
        dst.resize(m_size);
        copy_to_host(dst.data(), m_size);
    }

private:
    void release() {
        if (m_data) cudaFree(m_data);   // pas d'exception dans un destructeur
        m_data = nullptr;
        m_size = 0;
    }
    void swap(GPUMemory& other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
    }

    T* m_data = nullptr;
    size_t m_size = 0;
};

#undef MENU_GPU_HD
