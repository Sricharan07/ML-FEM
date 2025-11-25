#include "material/NeuralMaterial.hpp"
#include <stdexcept>
#include <iostream>

namespace femml {

NeuralMaterial::NeuralMaterial(const std::string& name, double rho)
    : Material(name), use_fallback_(false) {
    density_ = rho;
}

std::array<double, 6> NeuralMaterial::ComputeStressIncrement(
    const std::array<double, 6>& strain_increment,
    const std::array<double, 6>& current_strain,
    const std::array<double, 6>& current_stress) const {

    std::array<double, 6> updated_strain = current_strain;
    for (size_t i = 0; i < updated_strain.size(); ++i) {
        updated_strain[i] += strain_increment[i];
    }

    if (inference_callback_) {
        try {
            auto updated_stress = inference_callback_(updated_strain);
            std::array<double, 6> stress_increment{};
            for (size_t i = 0; i < stress_increment.size(); ++i) {
                stress_increment[i] = updated_stress[i] - current_stress[i];
            }
            return stress_increment;
        }
        catch (const std::exception& e) {
            std::cerr << "Neural network inference failed: " << e.what() << std::endl;
            if (!(use_fallback_ && fallback_material_)) {
                throw;
            }
            std::cerr << "Using fallback material" << std::endl;
        }
    }

    if (use_fallback_ && fallback_material_) {
        return fallback_material_->ComputeStressIncrement(
            strain_increment, current_strain, current_stress);
    }

    throw std::runtime_error("Neural material: no inference callback or fallback material set");
}

std::shared_ptr<Material> NeuralMaterial::Clone() const {
    auto clone = std::make_shared<NeuralMaterial>(name_, density_);
    clone->inference_callback_ = inference_callback_;
    clone->fallback_material_ = fallback_material_;
    clone->use_fallback_ = use_fallback_;
    return clone;
}

void NeuralMaterial::SetInferenceCallback(const InferenceCallback& callback) {
    inference_callback_ = callback;
}

void NeuralMaterial::LoadONNXModel(const std::string& model_path) {
    // TODO: Implement ONNX runtime loading
    // This requires linking against ONNX runtime library
    std::cout << "ONNX model loading not yet implemented: " << model_path << std::endl;
    std::cout << "Use SetInferenceCallback() to provide a custom inference function" << std::endl;
}

void NeuralMaterial::SetFallbackMaterial(const std::shared_ptr<Material>& fallback) {
    fallback_material_ = fallback;
    use_fallback_ = true;
}

} // namespace femml
