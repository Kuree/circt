add_circt_dialect_library(CIRCTCalyxTransforms
  CompileControl.cpp
  GoInsertion.cpp
  RemoveGroups.cpp

  DEPENDS
  CIRCTCalyxTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
