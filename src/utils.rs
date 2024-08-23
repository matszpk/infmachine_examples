use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

// Utilities for machine.
// General utitities for machine creation. Utilities designed to be generic and usable
// on machine with any value of parameters - cell_len_bits and data_part_len, proc_num...
// 
// Basic utilities:
// * load and determine max position at proc_id.
// * move back to start position in any data.
// * move data to another data (example: proc_id to mem_address or temp_buffer).
// * process data with any function.
// * any data operation includes stride (number of movements to next data part).
// * process integer in memory in ENDFORM form: [NP0, END0, NP1, END1, NP2, END2, ..].
//
// Number in ENDFORM: [NP0, END0, NP1, END1, NP2, END2, ..].
// NPx - part of number from lowest to highest.
// ENDx - mark last number part if value is not zero.
// Form: configurable circuit stage with specified part of stage indicator.
//       [STAGE_INDICATOR, STAGE_STATE, UNUSED]

// Algorithm to process number from memory start:
// 1. Load (NPx) number part from memory. Process with storing internal state.
// 2. Increase memory address without moving to start memory addres position.
// 3. Load (ENDx) end indicator from memory.
// 4. If ENDx is not zero:
// 4.1. End algorithm.
// 5. Move memory

// InitProcIdPosState - initialize proc id position from memory.
// Format in memory: start address is 0 in ENDFORM.
// Form of temp buffer: [PROC_ID_POS_LEN0, END0, PROC_ID_POS_LEN1, END1, ....]
//      ENDx - if not zero then mark end of temp buffer.
//      PROC_ID_POS_LEN0 - length of proc id position number.
// Algorithm:
// 1. Decrease number ENDFORM in memory address 0 and set memory address to 0.
//    Store information whether is zero to if_zero stage field.
// 2. Move temp buffer position forward.
// 3. If if_zero is true then:
// 3.1. Store 1 to current temp buffer part (at current temp buffer position).
// 3.2. End.
// 4. Otherwise go to 1.

#[derive(Clone)]
pub struct InitProcIdPosStage {
    stage: UDynVarSys,  // stage indicator
    int_stage: U3VarSys, // internal stage indicator - only for this stage.
    state: U2VarSys,  // state for stage
}

impl InitProcIdPosStage {
    pub fn new(stage_len: usize) -> Self {
        Self {
            stage: UDynVarSys::var(stage_len),
            int_stage: U3VarSys::var(),
            state: U2VarSys::var(),
        }
    }
}
