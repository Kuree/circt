// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:2 = memory[ld = 1, st = 0] (%[[VAL_3:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (index) -> (f32, none)
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_5:.*]]:4 = fork [4] %[[VAL_1]] : none
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_5]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_5]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_5]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_5]]#3 : none
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_7]] : index
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_8]] : index
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_15:.*]]#3 {{\[}}%[[VAL_16:.*]], %[[VAL_12]]] : index, index
// CHECK:           %[[VAL_17:.*]]:2 = fork [2] %[[VAL_14]] : index
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_15]]#2 {{\[}}%[[VAL_19:.*]], %[[VAL_9]]] : index, index
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_15]]#1 {{\[}}%[[VAL_21:.*]], %[[VAL_13]]] : index, index
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = control_merge %[[VAL_24:.*]], %[[VAL_10]] : none
// CHECK:           %[[VAL_15]]:4 = fork [4] %[[VAL_23]] : index
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_15]]#0 {{\[}}%[[VAL_26:.*]], %[[VAL_11]]] : index, index
// CHECK:           %[[VAL_27:.*]]:2 = fork [2] %[[VAL_25]] : index
// CHECK:           %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_27]]#1, %[[VAL_17]]#1 : index
// CHECK:           %[[VAL_29:.*]]:5 = fork [5] %[[VAL_28]] : i1
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_29]]#4, %[[VAL_17]]#0 : index
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_29]]#3, %[[VAL_18]] : index
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = cond_br %[[VAL_29]]#2, %[[VAL_20]] : index
// CHECK:           sink %[[VAL_35]] : index
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = cond_br %[[VAL_29]]#1, %[[VAL_22]] : none
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = cond_br %[[VAL_29]]#0, %[[VAL_27]]#0 : index
// CHECK:           sink %[[VAL_39]] : index
// CHECK:           %[[VAL_40:.*]] = merge %[[VAL_38]] : index
// CHECK:           %[[VAL_41:.*]]:2 = fork [2] %[[VAL_40]] : index
// CHECK:           %[[VAL_42:.*]] = merge %[[VAL_32]] : index
// CHECK:           %[[VAL_43:.*]]:2 = fork [2] %[[VAL_42]] : index
// CHECK:           %[[VAL_44:.*]] = merge %[[VAL_34]] : index
// CHECK:           %[[VAL_45:.*]]:2 = fork [2] %[[VAL_44]] : index
// CHECK:           %[[VAL_46:.*]] = merge %[[VAL_30]] : index
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = control_merge %[[VAL_36]] : none
// CHECK:           %[[VAL_49:.*]]:2 = fork [2] %[[VAL_47]] : none
// CHECK:           %[[VAL_50:.*]]:2 = fork [2] %[[VAL_49]]#1 : none
// CHECK:           %[[VAL_51:.*]] = join %[[VAL_50]]#1, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_48]] : index
// CHECK:           %[[VAL_52:.*]] = arith.addi %[[VAL_41]]#1, %[[VAL_43]]#1 : index
// CHECK:           %[[VAL_53:.*]] = constant %[[VAL_50]]#0 {value = 7 : index} : index
// CHECK:           %[[VAL_54:.*]] = arith.addi %[[VAL_52]], %[[VAL_53]] : index
// CHECK:           %[[VAL_55:.*]], %[[VAL_3]] = load {{\[}}%[[VAL_54]]] %[[VAL_2]]#0, %[[VAL_49]]#0 : index, f32
// CHECK:           sink %[[VAL_55]] : f32
// CHECK:           %[[VAL_56:.*]] = arith.addi %[[VAL_41]]#0, %[[VAL_45]]#1 : index
// CHECK:           %[[VAL_19]] = br %[[VAL_43]]#0 : index
// CHECK:           %[[VAL_21]] = br %[[VAL_45]]#0 : index
// CHECK:           %[[VAL_16]] = br %[[VAL_46]] : index
// CHECK:           %[[VAL_24]] = br %[[VAL_51]] : none
// CHECK:           %[[VAL_26]] = br %[[VAL_56]] : index
// CHECK:           %[[VAL_57:.*]], %[[VAL_58:.*]] = control_merge %[[VAL_37]] : none
// CHECK:           sink %[[VAL_58]] : index
// CHECK:           return %[[VAL_57]] : none
// CHECK:         }
  func @affine_load(%arg0: index) {
    %0 = memref.alloc() : memref<10xf32>
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
    br ^bb1(%6 : index)
  ^bb3: // pred: ^bb1
    return
  }
