// A placeholder MLIR module for the relu_f32 demo case.
//
// The demo uses *stub* IREE tools (see ./tools), so this file is never
// actually compiled by a real `iree-compile`; it exists so the case has a
// concrete source path and the pipeline can content-address it. With a real
// IREE install you would replace this with a genuine linalg/func module.
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "demo.relu"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
