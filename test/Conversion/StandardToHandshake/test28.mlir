// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) {id = 1 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_2]]#2 : none
// CHECK:           %[[VAL_7:.*]]:2 = memory[ld = 1, st = 0] (%[[VAL_8:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (index) -> (f32, none)
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_10:.*]]:4 = fork [4] %[[VAL_1]] : none
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_10]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_10]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_13:.*]] = constant %[[VAL_10]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_9]] : index
// CHECK:           %[[VAL_15:.*]] = br %[[VAL_10]]#3 : none
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_11]] : index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_12]] : index
// CHECK:           %[[VAL_18:.*]] = br %[[VAL_13]] : index
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_20:.*]]#3 {{\[}}%[[VAL_21:.*]], %[[VAL_17]]] : index, index
// CHECK:           %[[VAL_22:.*]]:2 = fork [2] %[[VAL_19]] : index
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_20]]#2 {{\[}}%[[VAL_24:.*]], %[[VAL_14]]] : index, index
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_20]]#1 {{\[}}%[[VAL_26:.*]], %[[VAL_18]]] : index, index
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = control_merge %[[VAL_29:.*]], %[[VAL_15]] : none
// CHECK:           %[[VAL_20]]:4 = fork [4] %[[VAL_28]] : index
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_20]]#0 {{\[}}%[[VAL_31:.*]], %[[VAL_16]]] : index, index
// CHECK:           %[[VAL_32:.*]]:2 = fork [2] %[[VAL_30]] : index
// CHECK:           %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]]#1, %[[VAL_22]]#1 : index
// CHECK:           %[[VAL_34:.*]]:5 = fork [5] %[[VAL_33]] : i1
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = cond_br %[[VAL_34]]#4, %[[VAL_22]]#0 : index
// CHECK:           sink %[[VAL_36]] : index
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = cond_br %[[VAL_34]]#3, %[[VAL_23]] : index
// CHECK:           sink %[[VAL_38]] : index
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = cond_br %[[VAL_34]]#2, %[[VAL_25]] : index
// CHECK:           sink %[[VAL_40]] : index
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = cond_br %[[VAL_34]]#1, %[[VAL_27]] : none
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = cond_br %[[VAL_34]]#0, %[[VAL_32]]#0 : index
// CHECK:           sink %[[VAL_44]] : index
// CHECK:           %[[VAL_45:.*]] = merge %[[VAL_43]] : index
// CHECK:           %[[VAL_46:.*]]:2 = fork [2] %[[VAL_45]] : index
// CHECK:           %[[VAL_47:.*]] = merge %[[VAL_37]] : index
// CHECK:           %[[VAL_48:.*]]:2 = fork [2] %[[VAL_47]] : index
// CHECK:           %[[VAL_49:.*]] = merge %[[VAL_39]] : index
// CHECK:           %[[VAL_50:.*]]:2 = fork [2] %[[VAL_49]] : index
// CHECK:           %[[VAL_51:.*]] = merge %[[VAL_35]] : index
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = control_merge %[[VAL_41]] : none
// CHECK:           %[[VAL_54:.*]]:4 = fork [4] %[[VAL_52]] : none
// CHECK:           %[[VAL_55:.*]]:2 = fork [2] %[[VAL_54]]#3 : none
// CHECK:           %[[VAL_56:.*]] = join %[[VAL_55]]#1, %[[VAL_7]]#1, %[[VAL_6]]#1, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_53]] : index
// CHECK:           %[[VAL_57:.*]] = arith.addi %[[VAL_46]]#1, %[[VAL_48]]#1 : index
// CHECK:           %[[VAL_58:.*]] = constant %[[VAL_55]]#0 {value = 7 : index} : index
// CHECK:           %[[VAL_59:.*]] = arith.addi %[[VAL_57]], %[[VAL_58]] : index
// CHECK:           %[[VAL_60:.*]]:3 = fork [3] %[[VAL_59]] : index
// CHECK:           %[[VAL_61:.*]], %[[VAL_8]] = load {{\[}}%[[VAL_60]]#2] %[[VAL_7]]#0, %[[VAL_54]]#2 : index, f32
// CHECK:           %[[VAL_62:.*]] = arith.addi %[[VAL_46]]#0, %[[VAL_50]]#1 : index
// CHECK:           %[[VAL_63:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_60]]#1] %[[VAL_2]]#0, %[[VAL_54]]#1 : index, f32
// CHECK:           %[[VAL_64:.*]] = arith.addf %[[VAL_61]], %[[VAL_63]] : f32
// CHECK:           %[[VAL_65:.*]] = join %[[VAL_54]]#0, %[[VAL_6]]#0 : none
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_60]]#0] %[[VAL_64]], %[[VAL_65]] : index, f32
// CHECK:           %[[VAL_24]] = br %[[VAL_48]]#0 : index
// CHECK:           %[[VAL_26]] = br %[[VAL_50]]#0 : index
// CHECK:           %[[VAL_21]] = br %[[VAL_51]] : index
// CHECK:           %[[VAL_29]] = br %[[VAL_56]] : none
// CHECK:           %[[VAL_31]] = br %[[VAL_62]] : index
// CHECK:           %[[VAL_66:.*]], %[[VAL_67:.*]] = control_merge %[[VAL_42]] : none
// CHECK:           sink %[[VAL_67]] : index
// CHECK:           return %[[VAL_66]] : none
// CHECK:         }
  func @affine_load(%arg0: index) {
    %0 = memref.alloc() : memref<10xf32>
    %10 = memref.alloc() : memref<10xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %c10 : index
    cond_br %2, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    %3 = arith.addi %1, %arg0 : index
    %c7 = arith.constant 7 : index
    %4 = arith.addi %3, %c7 : index
    %5 = memref.load %0[%4] : memref<10xf32>
    %6 = arith.addi %1, %c1 : index
    %7 = memref.load %10[%4] : memref<10xf32>
    %8 = arith.addf %5, %7 : f32
    memref.store %8, %10[%4] : memref<10xf32>
    br ^bb1(%6 : index)
  ^bb3: // pred: ^bb1
    return
  }
