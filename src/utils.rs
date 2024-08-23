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
// END_DP_FORM: [NP0_0, NP0_1, .., NP0_T, END0, NP1_0, NP1_1, .., NP1_T, END1, .....].
// T - (data_part_len + cell_len - 1) / cell_len.  Number of required cells to store data part.

// InitMemAddressEndPosState - initialize memory address end position from memory.
// Information about MemAddressEndPos in memory:
// At memory address 0: sequences of zero cells and cell with value 1 under
// memory address which is MemAddressPosEndPos value.
// Example: [0, 0, 0, 0, 0, 1] - MemAddressPosEndPos is 5

#[derive(Clone)]
pub struct InitMemAddressEndPosStage {
    stage: UDynVarSys,   // stage indicator
    int_stage: U3VarSys, // internal stage indicator - only for this stage.
    state: U2VarSys,     // state for stage
}

impl InitMemAddressEndPosStage {
    pub fn new(cell_len_bits: usize, data_part_len: usize, stage_len: usize) -> Self {
        Self {
            stage: UDynVarSys::var(stage_len),
            int_stage: U3VarSys::var(),
            state: U2VarSys::var(),
        }
    }
}

// InitProcIdEndPosState - initialize proc id end position from memory.
// Information about ProcIdEndPos in memory:
// At memory address after first cell 1 in memory:
// sequences of zero cells and cell with value 1 under memory address
// which is ProcIdPosEndPos value.

#[derive(Clone)]
pub struct InitProcIdEndPosStage {
    stage: UDynVarSys,   // stage indicator
    int_stage: U3VarSys, // internal stage indicator - only for this stage.
    state: U2VarSys,     // state for stage
}

impl InitProcIdEndPosStage {
    pub fn new(stage_len: usize) -> Self {
        Self {
            stage: UDynVarSys::var(stage_len),
            int_stage: U3VarSys::var(),
            state: U2VarSys::var(),
        }
    }
}
