#pragma once

#include "material/Material.hpp"
#include <functional>

namespace femml {

// Neural network material model
// Uses a callback function for inference (can be Python, ONNX, etc.)
class NeuralMaterial : public Material {
public:
    using InferenceCallback = std::function<std::array<double, 6>(const std::array<double, 6>&)>;

    NeuralMaterial(const std::string& name, double rho);
    ~NeuralMaterial() override = default;

    std::array<double, 6> ComputeStressIncrement(
        const std::array<double, 6>& strain_increment,
        const std::array<double, 6>& current_strain,
        const std::array<double, 6>& current_stress) const override;

    std::string GetType() const override { return "NeuralMaterial"; }

    std::shared_ptr<Material> Clone() const override;

    // Set the neural network inference function
    void SetInferenceCallback(const InferenceCallback& callback);

    // Load ONNX model (if available)
    void LoadONNXModel(const std::string& model_path);

    // For hybrid models: set a fallback material
    void SetFallbackMaterial(const std::shared_ptr<Material>& fallback);

private:
    InferenceCallback inference_callback_;
    std::shared_ptr<Material> fallback_material_;
    bool use_fallback_;
};

} // namespace femml
