add_circt_library(CIRCTDebug
  HWDebug.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTranslation
  )