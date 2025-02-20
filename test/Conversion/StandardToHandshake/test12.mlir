// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @more_imperfectly_nested_loops(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = fork [4] %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_5:.*]] = br %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_3]] : index
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_9:.*]] = mux %[[VAL_10:.*]]#2 {{\[}}%[[VAL_11:.*]], %[[VAL_7]]] : index, index
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_9]] : index
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_10]]#1 {{\[}}%[[VAL_14:.*]], %[[VAL_8]]] : index, index
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_17:.*]], %[[VAL_5]] : none
// CHECK:           %[[VAL_10]]:3 = fork [3] %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_10]]#0 {{\[}}%[[VAL_19:.*]], %[[VAL_6]]] : index, index
// CHECK:           %[[VAL_20:.*]]:2 = fork [2] %[[VAL_18]] : index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]]#1, %[[VAL_12]]#1 : index
// CHECK:           %[[VAL_22:.*]]:4 = fork [4] %[[VAL_21]] : i1
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_22]]#3, %[[VAL_12]]#0 : index
// CHECK:           sink %[[VAL_24]] : index
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = cond_br %[[VAL_22]]#2, %[[VAL_13]] : index
// CHECK:           sink %[[VAL_26]] : index
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = cond_br %[[VAL_22]]#1, %[[VAL_15]] : none
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = cond_br %[[VAL_22]]#0, %[[VAL_20]]#0 : index
// CHECK:           sink %[[VAL_30]] : index
// CHECK:           %[[VAL_31:.*]] = merge %[[VAL_29]] : index
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_25]] : index
// CHECK:           %[[VAL_33:.*]] = merge %[[VAL_23]] : index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_27]] : none
// CHECK:           %[[VAL_36:.*]]:4 = fork [4] %[[VAL_34]] : none
// CHECK:           sink %[[VAL_35]] : index
// CHECK:           %[[VAL_37:.*]] = constant %[[VAL_36]]#2 {value = 7 : index} : index
// CHECK:           %[[VAL_38:.*]] = constant %[[VAL_36]]#1 {value = 56 : index} : index
// CHECK:           %[[VAL_39:.*]] = constant %[[VAL_36]]#0 {value = 2 : index} : index
// CHECK:           %[[VAL_40:.*]] = br %[[VAL_31]] : index
// CHECK:           %[[VAL_41:.*]] = br %[[VAL_32]] : index
// CHECK:           %[[VAL_42:.*]] = br %[[VAL_33]] : index
// CHECK:           %[[VAL_43:.*]] = br %[[VAL_36]]#3 : none
// CHECK:           %[[VAL_44:.*]] = br %[[VAL_37]] : index
// CHECK:           %[[VAL_45:.*]] = br %[[VAL_38]] : index
// CHECK:           %[[VAL_46:.*]] = br %[[VAL_39]] : index
// CHECK:           %[[VAL_47:.*]] = mux %[[VAL_48:.*]]#5 {{\[}}%[[VAL_49:.*]], %[[VAL_45]]] : index, index
// CHECK:           %[[VAL_50:.*]]:2 = fork [2] %[[VAL_47]] : index
// CHECK:           %[[VAL_51:.*]] = mux %[[VAL_48]]#4 {{\[}}%[[VAL_52:.*]], %[[VAL_46]]] : index, index
// CHECK:           %[[VAL_53:.*]] = mux %[[VAL_48]]#3 {{\[}}%[[VAL_54:.*]], %[[VAL_40]]] : index, index
// CHECK:           %[[VAL_55:.*]] = mux %[[VAL_48]]#2 {{\[}}%[[VAL_56:.*]], %[[VAL_41]]] : index, index
// CHECK:           %[[VAL_57:.*]] = mux %[[VAL_48]]#1 {{\[}}%[[VAL_58:.*]], %[[VAL_42]]] : index, index
// CHECK:           %[[VAL_59:.*]], %[[VAL_60:.*]] = control_merge %[[VAL_61:.*]], %[[VAL_43]] : none
// CHECK:           %[[VAL_48]]:6 = fork [6] %[[VAL_60]] : index
// CHECK:           %[[VAL_62:.*]] = mux %[[VAL_48]]#0 {{\[}}%[[VAL_63:.*]], %[[VAL_44]]] : index, index
// CHECK:           %[[VAL_64:.*]]:2 = fork [2] %[[VAL_62]] : index
// CHECK:           %[[VAL_65:.*]] = arith.cmpi slt, %[[VAL_64]]#1, %[[VAL_50]]#1 : index
// CHECK:           %[[VAL_66:.*]]:7 = fork [7] %[[VAL_65]] : i1
// CHECK:           %[[VAL_67:.*]], %[[VAL_68:.*]] = cond_br %[[VAL_66]]#6, %[[VAL_50]]#0 : index
// CHECK:           sink %[[VAL_68]] : index
// CHECK:           %[[VAL_69:.*]], %[[VAL_70:.*]] = cond_br %[[VAL_66]]#5, %[[VAL_51]] : index
// CHECK:           sink %[[VAL_70]] : index
// CHECK:           %[[VAL_71:.*]], %[[VAL_72:.*]] = cond_br %[[VAL_66]]#4, %[[VAL_53]] : index
// CHECK:           %[[VAL_73:.*]], %[[VAL_74:.*]] = cond_br %[[VAL_66]]#3, %[[VAL_55]] : index
// CHECK:           %[[VAL_75:.*]], %[[VAL_76:.*]] = cond_br %[[VAL_66]]#2, %[[VAL_57]] : index
// CHECK:           %[[VAL_77:.*]], %[[VAL_78:.*]] = cond_br %[[VAL_66]]#1, %[[VAL_59]] : none
// CHECK:           %[[VAL_79:.*]], %[[VAL_80:.*]] = cond_br %[[VAL_66]]#0, %[[VAL_64]]#0 : index
// CHECK:           sink %[[VAL_80]] : index
// CHECK:           %[[VAL_81:.*]] = merge %[[VAL_79]] : index
// CHECK:           %[[VAL_82:.*]] = merge %[[VAL_69]] : index
// CHECK:           %[[VAL_83:.*]]:2 = fork [2] %[[VAL_82]] : index
// CHECK:           %[[VAL_84:.*]] = merge %[[VAL_67]] : index
// CHECK:           %[[VAL_85:.*]] = merge %[[VAL_71]] : index
// CHECK:           %[[VAL_86:.*]] = merge %[[VAL_73]] : index
// CHECK:           %[[VAL_87:.*]] = merge %[[VAL_75]] : index
// CHECK:           %[[VAL_88:.*]], %[[VAL_89:.*]] = control_merge %[[VAL_77]] : none
// CHECK:           sink %[[VAL_89]] : index
// CHECK:           %[[VAL_90:.*]] = arith.addi %[[VAL_81]], %[[VAL_83]]#1 : index
// CHECK:           %[[VAL_52]] = br %[[VAL_83]]#0 : index
// CHECK:           %[[VAL_49]] = br %[[VAL_84]] : index
// CHECK:           %[[VAL_54]] = br %[[VAL_85]] : index
// CHECK:           %[[VAL_56]] = br %[[VAL_86]] : index
// CHECK:           %[[VAL_58]] = br %[[VAL_87]] : index
// CHECK:           %[[VAL_61]] = br %[[VAL_88]] : none
// CHECK:           %[[VAL_63]] = br %[[VAL_90]] : index
// CHECK:           %[[VAL_91:.*]] = merge %[[VAL_72]] : index
// CHECK:           %[[VAL_92:.*]] = merge %[[VAL_74]] : index
// CHECK:           %[[VAL_93:.*]] = merge %[[VAL_76]] : index
// CHECK:           %[[VAL_94:.*]], %[[VAL_95:.*]] = control_merge %[[VAL_78]] : none
// CHECK:           %[[VAL_96:.*]]:4 = fork [4] %[[VAL_94]] : none
// CHECK:           sink %[[VAL_95]] : index
// CHECK:           %[[VAL_97:.*]] = constant %[[VAL_96]]#2 {value = 18 : index} : index
// CHECK:           %[[VAL_98:.*]] = constant %[[VAL_96]]#1 {value = 37 : index} : index
// CHECK:           %[[VAL_99:.*]] = constant %[[VAL_96]]#0 {value = 3 : index} : index
// CHECK:           %[[VAL_100:.*]] = br %[[VAL_91]] : index
// CHECK:           %[[VAL_101:.*]] = br %[[VAL_92]] : index
// CHECK:           %[[VAL_102:.*]] = br %[[VAL_93]] : index
// CHECK:           %[[VAL_103:.*]] = br %[[VAL_96]]#3 : none
// CHECK:           %[[VAL_104:.*]] = br %[[VAL_97]] : index
// CHECK:           %[[VAL_105:.*]] = br %[[VAL_98]] : index
// CHECK:           %[[VAL_106:.*]] = br %[[VAL_99]] : index
// CHECK:           %[[VAL_107:.*]] = mux %[[VAL_108:.*]]#5 {{\[}}%[[VAL_109:.*]], %[[VAL_105]]] : index, index
// CHECK:           %[[VAL_110:.*]]:2 = fork [2] %[[VAL_107]] : index
// CHECK:           %[[VAL_111:.*]] = mux %[[VAL_108]]#4 {{\[}}%[[VAL_112:.*]], %[[VAL_106]]] : index, index
// CHECK:           %[[VAL_113:.*]] = mux %[[VAL_108]]#3 {{\[}}%[[VAL_114:.*]], %[[VAL_100]]] : index, index
// CHECK:           %[[VAL_115:.*]] = mux %[[VAL_108]]#2 {{\[}}%[[VAL_116:.*]], %[[VAL_101]]] : index, index
// CHECK:           %[[VAL_117:.*]] = mux %[[VAL_108]]#1 {{\[}}%[[VAL_118:.*]], %[[VAL_102]]] : index, index
// CHECK:           %[[VAL_119:.*]], %[[VAL_120:.*]] = control_merge %[[VAL_121:.*]], %[[VAL_103]] : none
// CHECK:           %[[VAL_108]]:6 = fork [6] %[[VAL_120]] : index
// CHECK:           %[[VAL_122:.*]] = mux %[[VAL_108]]#0 {{\[}}%[[VAL_123:.*]], %[[VAL_104]]] : index, index
// CHECK:           %[[VAL_124:.*]]:2 = fork [2] %[[VAL_122]] : index
// CHECK:           %[[VAL_125:.*]] = arith.cmpi slt, %[[VAL_124]]#1, %[[VAL_110]]#1 : index
// CHECK:           %[[VAL_126:.*]]:7 = fork [7] %[[VAL_125]] : i1
// CHECK:           %[[VAL_127:.*]], %[[VAL_128:.*]] = cond_br %[[VAL_126]]#6, %[[VAL_110]]#0 : index
// CHECK:           sink %[[VAL_128]] : index
// CHECK:           %[[VAL_129:.*]], %[[VAL_130:.*]] = cond_br %[[VAL_126]]#5, %[[VAL_111]] : index
// CHECK:           sink %[[VAL_130]] : index
// CHECK:           %[[VAL_131:.*]], %[[VAL_132:.*]] = cond_br %[[VAL_126]]#4, %[[VAL_113]] : index
// CHECK:           %[[VAL_133:.*]], %[[VAL_134:.*]] = cond_br %[[VAL_126]]#3, %[[VAL_115]] : index
// CHECK:           %[[VAL_135:.*]], %[[VAL_136:.*]] = cond_br %[[VAL_126]]#2, %[[VAL_117]] : index
// CHECK:           %[[VAL_137:.*]], %[[VAL_138:.*]] = cond_br %[[VAL_126]]#1, %[[VAL_119]] : none
// CHECK:           %[[VAL_139:.*]], %[[VAL_140:.*]] = cond_br %[[VAL_126]]#0, %[[VAL_124]]#0 : index
// CHECK:           sink %[[VAL_140]] : index
// CHECK:           %[[VAL_141:.*]] = merge %[[VAL_139]] : index
// CHECK:           %[[VAL_142:.*]] = merge %[[VAL_129]] : index
// CHECK:           %[[VAL_143:.*]]:2 = fork [2] %[[VAL_142]] : index
// CHECK:           %[[VAL_144:.*]] = merge %[[VAL_127]] : index
// CHECK:           %[[VAL_145:.*]] = merge %[[VAL_131]] : index
// CHECK:           %[[VAL_146:.*]] = merge %[[VAL_133]] : index
// CHECK:           %[[VAL_147:.*]] = merge %[[VAL_135]] : index
// CHECK:           %[[VAL_148:.*]], %[[VAL_149:.*]] = control_merge %[[VAL_137]] : none
// CHECK:           sink %[[VAL_149]] : index
// CHECK:           %[[VAL_150:.*]] = arith.addi %[[VAL_141]], %[[VAL_143]]#1 : index
// CHECK:           %[[VAL_112]] = br %[[VAL_143]]#0 : index
// CHECK:           %[[VAL_109]] = br %[[VAL_144]] : index
// CHECK:           %[[VAL_114]] = br %[[VAL_145]] : index
// CHECK:           %[[VAL_116]] = br %[[VAL_146]] : index
// CHECK:           %[[VAL_118]] = br %[[VAL_147]] : index
// CHECK:           %[[VAL_121]] = br %[[VAL_148]] : none
// CHECK:           %[[VAL_123]] = br %[[VAL_150]] : index
// CHECK:           %[[VAL_151:.*]] = merge %[[VAL_132]] : index
// CHECK:           %[[VAL_152:.*]] = merge %[[VAL_134]] : index
// CHECK:           %[[VAL_153:.*]]:2 = fork [2] %[[VAL_152]] : index
// CHECK:           %[[VAL_154:.*]] = merge %[[VAL_136]] : index
// CHECK:           %[[VAL_155:.*]], %[[VAL_156:.*]] = control_merge %[[VAL_138]] : none
// CHECK:           sink %[[VAL_156]] : index
// CHECK:           %[[VAL_157:.*]] = arith.addi %[[VAL_151]], %[[VAL_153]]#1 : index
// CHECK:           %[[VAL_14]] = br %[[VAL_153]]#0 : index
// CHECK:           %[[VAL_11]] = br %[[VAL_154]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_155]] : none
// CHECK:           %[[VAL_19]] = br %[[VAL_157]] : index
// CHECK:           %[[VAL_158:.*]], %[[VAL_159:.*]] = control_merge %[[VAL_28]] : none
// CHECK:           sink %[[VAL_159]] : index
// CHECK:           return %[[VAL_158]] : none
// CHECK:         }
  func @more_imperfectly_nested_loops() {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : index
    %c1 = arith.constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%0: index):      // 2 preds: ^bb0, ^bb8
    %1 = arith.cmpi slt, %0, %c42 : index
    cond_br %1, ^bb2, ^bb9
  ^bb2: // pred: ^bb1
    %c7 = arith.constant 7 : index
    %c56 = arith.constant 56 : index
    %c2 = arith.constant 2 : index
    br ^bb3(%c7 : index)
  ^bb3(%2: index):      // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %c56 : index
    cond_br %3, ^bb4, ^bb5
  ^bb4: // pred: ^bb3
    %4 = arith.addi %2, %c2 : index
    br ^bb3(%4 : index)
  ^bb5: // pred: ^bb3
    %c18 = arith.constant 18 : index
    %c37 = arith.constant 37 : index
    %c3 = arith.constant 3 : index
    br ^bb6(%c18 : index)
  ^bb6(%5: index):      // 2 preds: ^bb5, ^bb7
    %6 = arith.cmpi slt, %5, %c37 : index
    cond_br %6, ^bb7, ^bb8
  ^bb7: // pred: ^bb6
    %7 = arith.addi %5, %c3 : index
    br ^bb6(%7 : index)
  ^bb8: // pred: ^bb6
    %8 = arith.addi %0, %c1 : index
    br ^bb1(%8 : index)
  ^bb9: // pred: ^bb1
    return
  }
