// Copyright 2017-2019, Nicholas Sharp and the Polyscope contributors. http://polyscope.run.
#pragma once

#include "polyscope/affine_remapper.h"
#include "polyscope/curve_network.h"
#include "polyscope/gl/color_maps.h"
#include "polyscope/histogram.h"

namespace polyscope {

class CurveNetworkScalarQuantity : public CurveNetworkQuantity {
public:
  CurveNetworkScalarQuantity(std::string name, CurveNetwork& network_, std::string definedOn, DataType dataType);

  virtual void draw() override;
  virtual void buildCustomUI() override;
  virtual std::string niceName() override;
  virtual void geometryChanged() override;

  // === Members
  const DataType dataType;

  // === Get/set visualization parameters

  // The color map
  CurveNetworkScalarQuantity* setColorMap(gl::ColorMapID val);
  gl::ColorMapID getColorMap();

  // Data limits mapped in to colormap
  CurveNetworkScalarQuantity* setMapRange(std::pair<double, double> val);
  std::pair<double, double> getMapRange();
  CurveNetworkScalarQuantity* resetMapRange(); // reset to full range

protected:
  // === Visualization parameters

  // Affine data maps and limits
  std::pair<float, float> vizRange;
  std::pair<double, double> dataRange;
  Histogram hist;

  // UI internals
  PersistentValue<gl::ColorMapID> cMap;
  const std::string definedOn;
  std::unique_ptr<gl::GLProgram> nodeProgram;
  std::unique_ptr<gl::GLProgram> edgeProgram;

  // Helpers
  virtual void createProgram() = 0;
  void setProgramUniforms(gl::GLProgram& program);
};

// ========================================================
// ==========             Node Scalar            ==========
// ========================================================

class CurveNetworkNodeScalarQuantity : public CurveNetworkScalarQuantity {
public:
  CurveNetworkNodeScalarQuantity(std::string name, std::vector<double> values_, CurveNetwork& network_,
                                 DataType dataType_ = DataType::STANDARD);

  virtual void createProgram() override;

  void buildNodeInfoGUI(size_t nInd) override;

  // === Members
  std::vector<double> values;
};


// ========================================================
// ==========            Edge Scalar             ==========
// ========================================================

class CurveNetworkEdgeScalarQuantity : public CurveNetworkScalarQuantity {
public:
  CurveNetworkEdgeScalarQuantity(std::string name, std::vector<double> values_, CurveNetwork& network_,
                                 DataType dataType_ = DataType::STANDARD);

  virtual void createProgram() override;

  void buildEdgeInfoGUI(size_t edgeInd) override;


  // === Members
  std::vector<double> values;
};


} // namespace polyscope
