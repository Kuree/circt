// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_apply_mod(
// CHECK-SAME:                                     %[[VAL_0:.*]]: index,
// CHECK-SAME:                                     %[[VAL_1:.*]]: none, ...) -> (index, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:3 = fork [3] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = arith.remsi %[[VAL_2]], %[[VAL_5]]#0 : index
// CHECK:           %[[VAL_7:.*]]:3 = fork [3] %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_3]]#0 {value = 0 : index} : index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]]#2, %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_7]]#1, %[[VAL_5]]#1 : index
// CHECK:           %[[VAL_11:.*]] = select %[[VAL_9]], %[[VAL_7]]#0, %[[VAL_10]] : index
// CHECK:           return %[[VAL_11]], %[[VAL_3]]#2 : index, none
// CHECK:         }
func @affine_apply_mod(%arg0: index) -> index {
    %c42 = arith.constant 42 : index
    %0 = arith.remsi %arg0, %c42 : index
    %c0 = arith.constant 0 : index
    %1 = arith.cmpi slt, %0, %c0 : index
    %2 = arith.addi %0, %c42 : index
    %3 = select %1, %2, %0 : index
    return %3 : index
  }
